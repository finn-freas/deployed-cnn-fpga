# roastbot-fpga

A chatbot that roasts you, running entirely on a $50 FPGA board (Sipeed Tang Nano 20K). I originally made a chatbot trained on my friend's texts, and after realizing how funny it was,
I decided to make a bot that only roasts you (since my friend kind of does that anyway). I wrote the hardware to contain it on an fpga, so I can carry this little thing with me wherever I go!

```
you> fuck you
roast bot> you're crying because a tiny board laughed at your ass
you> ur just a bot
roast bot> so you lost to a tiny little piece of plastic and still can't get a joke? fuckin pathetic 😤
```

It's a 16-layer, 12-million-parameter language model, and the FPGA does all of it: every
layer, the attention, the chat memory, picking each word and choosing the best of 6 replies.
The laptop just sends your message over USB and prints the answer.

- a reply takes about **4.8 seconds**
- the model fills the board's whole 8 MB of memory
- the board's replies match a Python model of the hardware bit for bit
- built with the open-source tools (Yosys, nextpnr, Verilator, cocotb), no vendor IDE

**[How it works →](docs/how-it-works.md)** (the model, how it was trained, the hardware, and results)

## Running it

The trained model is in the repo, so you don't need to train anything:

```bash
pip install numpy torch tokenizers
./chat.sh                                                        # chat with it on your computer
scripts/build_board.sh                                           # or build the bitstream
openFPGALoader -b tangnano20k -f build/board54/roastbot_top.fs   # write it to flash
./chat.sh --board                                                # and chat with it on the board
```

`./chat.sh` runs the exact same integer math as the board, so you get the same replies
without one.

## What's in here

| | |
|-|-|
| `models/pr_board.pkl.xz` | the quantized model the board runs (unpacked the first time you chat) |
| `models/pr.pt` | the same model before quantizing, in PyTorch |
| `models/pre_b16.pt` | the base model, before it learned to roast |
| `data/people/pr/` | the ~7,800 roast chats it was fine-tuned on |
| `data/tok/` | the tokenizer |
| `data/style/` | the TikTok comments and comebacks shown to the writer model |
| `rtl/` | the hardware |

To train your own from scratch, run `scripts/train_roastbot.sh` (needs a GPU with 8 GB).
