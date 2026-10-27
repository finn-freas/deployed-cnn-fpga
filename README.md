# deployed-cnn-fpga

[![CI](https://github.com/finn-freas/deployed-cnn-fpga/actions/workflows/ci.yml/badge.svg)](https://github.com/finn-freas/deployed-cnn-fpga/actions/workflows/ci.yml)

My own INT8 neural network accelerator, running a person detector on a $25 FPGA board
(Sipeed Tang Nano 20K, Gowin GW2AR-18).

The model is the MLPerf Tiny Visual Wake Words reference (MobileNetV1 0.25, 96x96 RGB in,
"person" / "no person" out). The NPU's output matches TensorFlow Lite's int8 reference
kernels bit for bit, on every layer, not just the final class. Everything is built with the
open-source flow (Yosys, nextpnr, Verilator, cocotb), no vendor IDE.

- VWW takes 243,602 cycles with ideal memory and 256,642 with the weights in the board's
  SDRAM, so **4.75 ms per image at 54 MHz**
- 8x8 MAC array with a depthwise mode and 2 requantizers, using 77% of the LUTs and 64 DSP
  multipliers
- the board build closes at 64 MHz after routing and runs at 54

Everything passes in simulation, including the whole board system with flash and SDRAM
models. I'm bringing it up on the real board now ([docs/bringup.md](docs/bringup.md)).

## How it fits together

```mermaid
flowchart LR
    subgraph host["Host (Python)"]
        tfl["model.tflite"] --> ir[("IR")]
        ir --> comp["compiler"]
        comp --> sim["cycle-exact<br/>simulator"]
        comp --> img["program + weights<br/>flash image"]
    end
    subgraph fpga["Tang Nano 20K (GW2AR-18)"]
        flash[("SPI flash")] -->|boot| sys["npu_system<br/>boot · UART · SDRAM ctrl"]
        sys <--> sdram[("SDRAM<br/>weights")]
        sys --> core
        subgraph core["npu_core"]
            seq["sequencer"] --> te["tile engine<br/>im2col · transposer"]
            te --> arr["8×8 MAC array<br/>+ depthwise mode"]
            arr --> rq["2× requant<br/>TFLite rounding"]
            rq --> act[("activation<br/>memory 56 KiB")]
            act --> te
        end
        core --> hdmi["HDMI overlay<br/>thumbnail · verdict · latency"]
    end
    img --> flash
    cam["webcam<br/>sw/board/demo.py"] -->|UART| sys
```

## How the NPU works

The NPU runs one layer per instruction. There are four: CONV (which also covers the packed
first layer and fully connected layers), DWCONV, MAXPOOL and ARGMAX. The compiler turns
average pooling into a DWCONV with all-ones weights and 1x1 depthwise layers into a CONV.
The encoding is in [docs/isa.md](docs/isa.md).

Feature maps stay on chip in a 56 KiB activation memory. The compiler places each layer's
input and output so they fit there together. The weights and per-channel parameters (bias, multiplier, shift) are
copied from flash into SDRAM at boot and streamed in while each layer runs.

- **Tile engine.** For a CONV it gathers the input windows from activation memory (im2col)
  and passes them through a transposer into the array, one job at a time. A job is 8 output
  pixels × 8 output channels. In the first layer a pixel's RGB bytes are packed into one
  word, so the 3-channel input still keeps the array busy.
- **MAC array.** 8x8 and output-stationary: each PE accumulates one output value, and the
  inputs enter with a row skew. When the last product of a job comes through, every PE
  copies its sum into a shadow register and starts on the next job, while the drain reads
  the shadow registers out.
- **Depthwise mode.** For DWCONV a job is 8 neighbouring output pixels of one row × 8
  channels. The array keeps its row skew, so each PE row uses exactly the word arriving from
  activation memory, and every tap is issued together with the read of its word. That way
  depthwise layers need no line buffer.
- **Requantization.** Two requant pipelines turn the sums back into int8 the same way
  TFLite does: double rounding for convolutions, single rounding for fully connected layers,
  picked per instruction. The results go back into activation memory for the next layer.
- **Area.** Address math uses counters and shift-add instead of multipliers, and the
  transposer and parameter store are in LUT RAM. On Gowin a flip-flop can only take its
  input from the LUT in its own slice, so a register fed by another register costs an extra
  pass-through LUT. I avoid register copies wherever I can for that reason.
  [docs/notes.md](docs/notes.md) has the details.

## The compiler and simulator

`sw/model/export_ir.py` converts an int8 TFLite model into a simple IR (JSON + npz, committed
for both models in [`models/ir`](models/ir), so you don't need TensorFlow).
`sw/npu/compiler.py` maps the IR onto the ISA: weight layouts, padding, where each feature
map goes in activation memory, average-pool lowering and the rounding mode per layer.
`sw/npu/sim.py` runs the program with the same integer math as the RTL and predicts the
cycle count of every instruction exactly.

```bash
cd sw && python -m npu.compiler ../models/ir/vww --out ../build/npu_vww
```

## Results

| Model | Cycles | Latency at 54 MHz | Accuracy |
|-|-:|-:|-|
| MLPerf Tiny VWW (96x96x3, 29 layers, 7.5M MACs) | 243,602 | 4.5 ms (4.75 ms from SDRAM) | = TFLite int8 |
| MNIST CNN | 16,612 | 0.31 ms | = TFLite int8 |

| Board build (`npu_board_top`) | |
|-|-|
| Utilisation | LUT4 77%, DFF 55%, BSRAM 45/46, 64 DSP multipliers |
| Fmax after routing | 64.4 MHz NPU clock (runs at 54 MHz), 83.5 MHz HDMI clock (runs at 25.2 MHz) |

The first version that ran VWW took 705k cycles. [docs/notes.md](docs/notes.md) lists each
optimization step from there to 243k, plus the ones I left out.

## Running it on the board

For the first time, go through the [bring-up checklist](docs/bringup.md). There's a small
test bitstream plus `sw/board/board_check.py` that check the UART, SDRAM and flash one at a
time before the NPU goes on.

```bash
make -C fpga/tangnano20k TOP=npu_board_top FREQ_MHZ=54        # bitstream (or grab it from CI)
make -C fpga/tangnano20k TOP=npu_board_top prog               # write it to flash
cd sw && python -m npu.board ../models/ir/vww --image photo.jpg --program   # model + demo image
python sw/board/demo.py --port COM5                           # webcam -> board -> verdict
```

At power-up the board copies the weights from flash into SDRAM and classifies the demo
image. After that, any image sent over the UART (1 Mbaud) gets classified. The HDMI output
shows the image, the verdict, the score and the latency. S1 resets, S2 reruns the demo image.

## Building and testing

```bash
make env                 # .venv (simulation) and .venv-tf (TFLite export)
make test-sw             # ISA, requant, compiler and simulator tests
make test-rtl            # cocotb + Verilator (needs OSS CAD Suite in ~/oss-cad-suite)
```

The RTL tests compare against TFLite's tensors layer by layer and against the simulator's
cycle counts, run random networks (odd sizes, strides, padding), and simulate the whole board
system with SPI flash and SDRAM models. CI runs all of it and builds the bitstreams.

## Layout

```
rtl/npu/      the NPU: core, tile engine, MAC array, requant, pool
rtl/board/    board system: flash boot, SDRAM controller, UART, HDMI overlay, bring-up test
sw/npu/       ISA, compiler, cycle-exact simulator, TFLite-exact quantization, flash images
sw/board/     board check and webcam demo
sw/model/     model export (TFLite -> IR), training script
models/       the VWW and MNIST TFLite models and their IR
tb/           cocotb testbenches
fpga/         Tang Nano 20K constraints and build
docs/         ISA reference, bring-up checklist, design notes
```

## License

MIT, see [LICENSE](LICENSE). The MLPerf Tiny VWW model files in `models/vww/` are from
[mlcommons/tiny](https://github.com/mlcommons/tiny) (Apache License 2.0).
