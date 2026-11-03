# How it works

[← back to the README](../README.md)

The board is a Sipeed Tang Nano 20K. Its FPGA (Gowin GW2AR-18) has about 20K LUTs, 48
multipliers, 46 block RAMs, and 8 MB of SDRAM inside the same package. That's tiny for a
language model, so everything here is about fitting the model in that 8 MB and getting it out
of the SDRAM fast enough.

## The model

It's a Llama-style transformer, sized to fit the board:

| | |
|-|-|
| Layers | 16 |
| Width | 256 (704 in the feed-forward part) |
| Attention | 8 query heads sharing 2 key/value heads, rotary positions |
| Vocabulary | 3,072 word pieces (byte-level BPE, with slots for emoji) |
| Context | 512 word pieces |
| Weights | 4-bit, with an 8-bit scale per group of 32 |
| Word table and chat memory | 8-bit |
| Everything in between | 16-bit integers |

There's no floating point on the FPGA. Softmax, RMSNorm and SiLU use small lookup tables
instead, so all the math is integer math. `board.py` has the same math in Python, and the
hardware has to match it exactly.

## How the roast bot is trained

**1. A base model.** It's trained from scratch on about 320 million word pieces of everyday
conversation (the SODA dataset plus Reddit reply chains from casual subreddits), 1.6 billion
word pieces in total. This part only teaches it English and how chats go back and forth. The
last 15% of training uses 4-bit weights, so the model gets used to them before it's packed
for the board.

**2. Roast chats.** The model is too small to learn roasting from a few examples, so a bigger
model writes them. A 4-billion-parameter model (`wesjos/Qwen3.5-4B-Fuck`, 4-bit, run locally)
writes around 10,000 short text chats where someone talks to the bot or argues with it, and
the bot roasts back. It's shown funny TikTok comments and a bank of comebacks for the tone.
No slurs, nothing about groups of people, no suicide jokes.

**3. Filtering.** The same writer model reads every chat again and scores how much sense each
bot line makes after what was said. The worst ones get dropped. Lines that copy an example
word for word get cut. The emoji at the end of each line get re-picked so they fit what was
said. "Your mom" comebacks are only kept right after an insult.

**4. Fine-tuning.** The base model is trained on the roast chats, with 5% everyday chat mixed in
so it doesn't forget how to talk. Training stops when the validation loss stops going down.
Then the model is quantized and packed into one 8 MB image for the SDRAM.

`scripts/train_roastbot.sh` runs all of it. You need a GPU with 8 GB for the writer model.

## How a reply works

1. **Reading your message.** The laptop splits it into word pieces and sends them over the UART.
   For each one the board runs all 16 layers and saves its keys and values in the chat memory
   in SDRAM (about 24 ms per piece).
2. **Writing.** After the 16 layers, the board multiplies the result by the whole word table
   to get a score for all 3,072 pieces, picks one at random from the likely ones, and feeds it
   back in (about 30 ms per piece).
3. **Scoring.** For every piece it writes, the board also works out how likely the model
   thought it was (a log-probability), plus a small bonus for words that are rare in normal
   texting, so "lol ok" doesn't win every time. A reply's score is the average.
4. **Picking the best.** The board writes up to 6 candidate replies and keeps track of the
   best average score. The laptop drops candidates that loop or repeat themselves, and
   "your mom" lines that don't follow an insult or came too soon after the last one.
5. **Remembering.** The winner goes back into the chat memory, so the bot remembers what it
   actually said.

The board only has to read your new message each time. Everything from earlier in the chat is
already in the chat memory. When the 512-piece context fills up, the oldest messages get
dropped and the last 256 pieces are read in again.

### The candidate score

At each step there's a logit for all 3,072 pieces. The sampler makes two passes over them.
The first finds the biggest one. The second looks up 2^(logit − max) in a table for every
piece and adds them up into a total, which is the softmax denominator. Picking a word needs
that total anyway, so the score comes almost for free:

```
log2 p(piece) = −(max − logit[piece]) × log2(e) − log2(total)
score         = log2 p(piece) + bonus[piece]
```

The log2 is another lookup table, and everything is fixed point with 8 fraction bits. The
probability is the plain one (temperature 1), not the one used for sampling, so it measures
how natural the model found the word, not how lucky the draw was. The bonus comes from a table
in the SDRAM image: −log of how often the piece shows up in everyday texting, so words that
pick up what you said ("toaster", "motherboard") beat filler.

The board keeps the sum and the count for the current candidate, and the sum, count and number
of the best one so far. To check if a candidate beats the best, it compares the averages
without dividing: `sum × best_count > best_sum × count`. Ties go to the earlier candidate.

### Picking the word

Sampling uses the Gumbel-max trick: add Gumbel noise (from a 1,024-entry table, with an
xorshift32 random number) to each scaled logit and take the biggest. Only pieces within a
cutoff of the top logit can win, which keeps out the really unlikely ones. Emoji are a
separate class: first the sampler decides "emoji or not" from the emoji share of the total
(scaled by a gain the laptop sets, so a chat ends up with the usual amount of emoji), then it
picks within that class.

## How the hardware works

The model is way too big for the FPGA's own memory, so all 8 MB gets streamed out of the SDRAM
for every word piece. Almost every design choice is about using that stream well.

The model is a program. `pack.py` turns it into one SDRAM image: the weights, the word
table, the lookup tables, room for the chat memory, and a short list of operations (matrix
multiply, normalize, attention for layer 5, add, ...). The engine reads that list and runs it
once per word piece.

| Block | What it does |
|-|-|
| `engine` | reads the operation list and runs the other blocks |
| `mv_unit` | matrix times vector: 16 4-bit weights per clock straight out of the SDRAM stream, plus the word-table scores |
| `vec_unit` | everything element-wise: looking up words, RMSNorm, rotary positions, SiLU, adds, writing the chat memory |
| `attn_unit` | attention over the 8-bit chat memory, 4 query heads per key/value head at once |
| `sampler` | softmax, the emoji control, picking the word, and the candidate score |
| `sdram_ctrl`, `stream_rd`, `sdram_cdc` | the SDRAM controller at 108 MHz, the read stream, and the crossing to 54 MHz |
| `host_link`, `uart` | the USB UART commands: upload the image, run a word piece, read the result |

### The matrix unit

Every weight matrix is stored row by row. A row starts with a 16-bit multiplier and a shift
for the whole row, then one 8-bit scale per group of 32 weights, then the weights themselves,
16 4-bit values per 64-bit word. For each word that comes out of the SDRAM:

1. 16 multipliers multiply the 16 weights by their 16-bit inputs, and the products are added
2. after every 32 inputs, that partial sum is multiplied by the group's scale and added to the
   row total
3. at the end of the row, the total is multiplied by the row's multiplier, shifted, rounded,
   and saturated back to 16 bits

The inputs sit in a small RAM with 16 lanes, so all 16 come out in one clock. The word table
uses the same unit with 8-bit weights, 8 per word.

### Attention

The chat memory stores the keys and values as 8-bit numbers, with a shift per layer picked
when the model is packed. Each key or value head is shared by 4 query heads, so every word
read from the SDRAM gets used 4 times, once per query head, over 4 clocks. The keys are stored
position by position. The values are stored transposed, so the weighted sum over all
positions can be built up one column at a time. The softmax uses the same 2^x table as the
sampler, and there's one division per output at the end.

### Twice the SDRAM clock

The SDRAM bus is 32 bits wide but the matrix unit wants 64 bits every clock, so at 54 MHz it
sat idle about half the time. Now the SDRAM controller runs at 108 MHz, which is exactly one
64-bit word per 54 MHz clock. The PLL makes both clocks from the board's 27 MHz oscillator.
This took a reply from 7.6 s to 4.8 s, with the same replies.

**Crossing between the clocks.** Requests and read data go through an async FIFO in block RAM.
The read and write pointers cross between the clocks in Gray code, where only one bit
changes at a time, so a pointer caught halfway through a change is still off by at most one.
Control signals cross with a toggle handshake.

**Timing the SDRAM.** Above 100 MHz the SDRAM needs a CAS latency of 3 instead of 2. Its
internal access takes about 18–20 ns, and 2 clocks at 108 MHz is only 18.5 ns. All the other
SDRAM timings (row open, precharge, refresh) are counted again in the faster clock.

**Catching the read data.** At 108 MHz a clock is 9.3 ns. The data comes back almost a full
clock late (the clock going out through a pin, the SDRAM's access time, the data coming back
through the pins), and each word is only steady on the wires for part of the clock. The data
is captured in registers right at the pins. The PLL can shift the phase of the clock going to
the SDRAM, which slides that window earlier or later. At power-up the controller writes a test
pattern and reads it back with every read latency, then picks the middle of the ones that
work. For each new build, `scripts/sdram_diag.py` sweeps the phase with 130 million words at
each setting, and the build uses the middle of the phases with zero errors (3, 4 and 5 for the
current one, so it uses 4). `scripts/repack_phase.sh` changes the phase in a finished build
without running place and route again.

### Fitting it in

Every multiplier has registers right at its inputs and outputs, because nextpnr doesn't time
the paths through the DSP blocks. Without them, some builds passed timing and still got wrong
answers on the board. That costs about 1% more clocks per word piece. The design is at 75% of
the LUTs, so about half the placement seeds don't place, and each new build tries a few.

## Results

| Board build (`roastbot_top`) | |
|-|-|
| FPGA usage | LUT4 75%, DFF 63%, BSRAM 43/46, 34/48 multipliers |
| Max clock after routing | 68.9 MHz for the engine (runs at 54 MHz), 126.5 MHz for the SDRAM side (runs at 108 MHz) |
| SDRAM image | 8,038 KiB of 8,188 (weights 6,144, word table 792, chat memory 1,024) |
| Speed | reading 24 ms and writing 30 ms per word piece, about 4.8 s per reply |

| Test | |
|-|-|
| cocotb (Verilator) | `link`, `mv` and `engine` all bit-exact with `board.py` |
| On the board | 14 of 14 replies identical to the CPU (`scripts/board_check.py --best`) |

### Why not bigger

16 layers is the limit for this board. Each extra layer costs 448 KiB (384 for the weights and
64 for its chat memory), and there's only 150 KiB left. An 18-layer model would only fit with
a 4-bit word table and 4-bit keys and values. Measured on the 16-layer model, those cost
about 1.5% and 1.6% in perplexity, which is about as much as two more layers would gain. More
layers would also make every word piece about 6% slower each.

## Testing

`board.py` is the reference. It does the exact integer math of the hardware, including the
rounding, the lookup tables and the random numbers, so the same seed gives the same reply on
the CPU and on the board.

- `scripts/sim.sh link|mv|engine` runs the cocotb tests in Verilator. The engine test runs a
  whole word piece, then samples and scores two candidates, and checks every register against
  `board.py`.
- `scripts/board_check.py --best` sends 14 messages to the board and to the CPU and checks that
  every reply, every candidate score and every pick is the same. Any change to the RTL has to
  pass it.
- `./chat.sh` without `--board` runs the reference model, so you can try a model before it goes
  on the board.

The first time, `chat.sh --board` uploads the model image to the SDRAM (about a minute). It
checks the checksum first, so it only does that once per power-up.
