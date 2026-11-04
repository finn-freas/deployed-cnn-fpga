# roastbot-fpga

A chatbot that roasts you, running entirely on a $50 FPGA board (Sipeed Tang Nano 20K). I originally made a chatbot trained on my friend's texts, and after realizing how funny it was,
I decided to make a bot that only roasts you (since my friend kind of does that anyway). I wrote the hardware to contain it on an fpga, so I can carry this little thing with me wherever I go! 

Still being improved upon!

```
you> fuck you
roast bot> you're crying because a tiny board laughed at your ass
you> ur just a bot
roast bot> so you lost to a tiny little piece of plastic and still can't get a joke? fuckin pathetic 😤
```

It's a 16-layer, 12-million-parameter language model, and the FPGA does all of it.
The laptop just sends your message over USB and prints the answer (but you don't need a laptop!)

- a reply takes about **6 seconds**
- the model fills the board's whole 8 MB of memory
- the board's replies match a Python model of the hardware bit for bit
- built with the open-source tools (Yosys, nextpnr, Verilator, cocotb), no vendor IDE

**[How it works →](docs/how-it-works.md)** (the model, how it was trained, the hardware, and results)

## Running it

The trained model and a tested bitstream are in the repo, so you don't need to train or build
anything:

```bash
pip install numpy torch tokenizers
./chat.sh                                                    # chat with it on your computer
openFPGALoader -b tangnano20k -f bitstream/roastbot.fs       # or write the bitstream to the board's flash
./chat.sh --board                                            # and chat with it on the board
```

`./chat.sh` runs the exact same integer math as the board, so you get the same replies
without one. Every chat starts from seed 1, so the same messages always get the same
replies. `./chat.sh --seed random` picks a new seed and prints it, so you can replay a chat later
with `--seed <number>`. To build the bitstream yourself, run `scripts/build_board.sh`. Not every
placement seed places, so try a few (`SEED=6 scripts/build_board.sh`), and check the SDRAM
timing on the board with `scripts/sdram_diag.py` before you trust it.

## What's in here

| Folder | What it has |
|-|-|
| `src/` | the Python side. `board.py` is the reference model: the exact integer math the hardware does, used to check the board bit for bit. `pack.py` packs a model into the 8 MB SDRAM image. `engine.py` runs that image the way the hardware engine does, op by op, and counts the clock cycles. `link.py` talks to the board over USB. `chat.py` is the chat program (picking replies, "your mom" rules, the block list). `tok.py` and `corpus.py` build the tokenizer and the everyday-chat data, `train.py` trains the base model, and `finetune.py` teaches it to roast |
| `rtl/` | the hardware in SystemVerilog: the engine and its units (matrix, vector, attention, sampler, SDRAM read stream) |
| `rtl/board/` | the parts specific to the Tang Nano 20K: the top level, SDRAM controller, clock crossing, PLLs, UART, pin constraints and clock constraints |
| `rtl/sim/` | simulation tops and an SDRAM model, for the tests in `tb/` |
| `tb/` | cocotb testbenches that run the hardware in Verilator and check it against `src/board.py` (run with `scripts/sim.sh link`, `mv` or `engine`) |
| `tests/` | plain Python unit tests (pytest) for the USB link framing and the tokenizer |
| `scripts/` | everything else you run: building the bitstream (`build_board.sh`), checking the board (`board_check.py`, `sdram_diag.py`), the whole training recipe (`train_roastbot.sh`) and its steps (writing roast chats, filtering them, emoji, distillation targets), and evaluating replies |
| `models/` | the trained models: `pr_board.pkl.xz` is the quantized model the board runs (unpacked the first time you chat), `pr.pt` is the same model before quantizing, and `pre_b16.pt` is the base model before it learned to roast |
| `data/people/pr/` | the ~8,100 roast chats it was fine-tuned on, all written by the writer model |
| `data/tok/` | the tokenizer |
| `data/style/` | the TikTok comments and comebacks shown to the writer model for the tone |
| `bitstream/` | `roastbot.fs`, the tested bitstream (14 of 14 replies identical to the CPU, plus 14 of 14 insults, SDRAM clock phase 4) |
| `docs/` | [how it works](docs/how-it-works.md): the model, the training, the hardware, and the results |

`chat.sh` at the top just runs `src/chat.py`.

To train your own from scratch, run `scripts/train_roastbot.sh` (needs a GPU with 8 GB).
