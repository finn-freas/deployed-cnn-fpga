# roastbot-fpga

A small language model that roasts you over text, running on a $50 FPGA board (Sipeed Tang
Nano 20K, Gowin GW2AR-18).

You type a message on your laptop, it goes to the board over USB, and the board writes the
reply one word piece at a time. The whole model runs on the FPGA: the layers, attention, the
chat memory, picking each word, and scoring the candidate replies. The laptop just sends your
message and shows the answer. Everything is built with the open-source tools (Yosys, nextpnr,
Verilator, cocotb), no vendor IDE.

- the model is a 16-layer transformer with 12 million parameters, and it fills the board's
  8 MB SDRAM
- a reply takes about **4.8 seconds** (around 25 word pieces per second)
- the replies on the board match a Python model of the hardware bit for bit
- the SDRAM runs at 108 MHz and the rest of the design at 54 MHz

```
you> fuck you
roast bot> you're crying because a tiny board laughed at your ass
you> ur just a bot
roast bot> so you lost to a tiny little piece of plastic and still can't get a joke? fuckin pathetic 😤
```

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

**Twice the SDRAM clock.** The SDRAM bus is 32 bits wide but the matrix unit wants 64 bits
every clock, so at 54 MHz it sat idle about half the time. Now the SDRAM controller runs at
108 MHz, which is exactly one 64-bit word per 54 MHz clock. The two clock domains are
connected by an async FIFO in block RAM with Gray-code pointers. At 108 MHz the read data
comes back almost a full clock period late, so the SDRAM clock phase can be shifted in the
PLL, and at power-up the controller tries every read latency on a test pattern and picks the
middle of the ones that work. This took a reply from 7.6 s to 4.8 s.

**Fitting it in.** Every multiplier has registers right at its inputs and outputs, because
nextpnr doesn't time the paths through the DSP blocks. Without that, some builds passed
timing and still got wrong answers on the board.

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

16 layers is the limit for this board. An 18-layer model would only fit with 4-bit chat memory
and a 4-bit word table, and the quality lost to that is about as much as the extra layers add.

## Running it

```bash
scripts/build_board.sh                                     # bitstream -> build/board54/roastbot_top.fs
openFPGALoader -b tangnano20k -f build/board54/roastbot_top.fs   # write it to flash
./chat.sh --board                                          # chat with it
```

The first time, `chat.sh --board` uploads the model image to the SDRAM (it checks the
checksum first, so it only does that once per power-up).

`./chat.sh` without `--board` runs the same integer model on the CPU, with the same replies.
`scripts/sim.sh link|mv|engine` runs the RTL tests.
