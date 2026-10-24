# deployed-cnn-fpga

A custom INT8 neural-network accelerator ("NPU") and its compiler, running the MLPerf Tiny
**Visual Wake Words** person detector on a $25 Sipeed Tang Nano 20K (Gowin GW2AR-18), built
entirely with open-source tools (Yosys, nextpnr-himbaechel, Verilator, cocotb).

- **Bit-exact with TensorFlow Lite**: all 29 layers of the int8 MobileNet-style model match
  TFLite's reference kernels on every tested image (logits on 1000 images).
- **~273k cycles per inference** with weights in the board's SDRAM: about 5 ms at the
  board's 54 MHz NPU clock. The compute core alone (ideal memory) needs 258,808 cycles.
- **Standalone on the board**: boots from the SPI flash, copies the weights into the
  in-package SDRAM, and classifies images sent over USB-UART (e.g. a webcam), showing the
  input, verdict, score and latency on HDMI.

## Architecture

```
            ┌─────────────── npu_core ────────────────┐
 UART ──►   │ sequencer ─► tile engine ─► 8x8 MAC array ─► 2x requant ─► activation memory (56 KB)
 flash ──►  │      │            ▲  ▲                                         │
 SDRAM ◄─►  │      │      im2col/transposer    weights/params (streamed from SDRAM)
            │      └─► MAXPOOL unit (optional), ARGMAX                       │
            └─────────────────────────────────────────┘    HDMI overlay ◄─────┘ (thumbnail, result)
```

- **ISA**: CONV (incl. packed first layer and FC), DWCONV, MAXPOOL, ARGMAX; AVGPOOL and 1x1
  depthwise layers are lowered by the compiler ([docs/isa.md](docs/isa.md)).
- **Array**: 8x8 output-stationary MAC array with row skew only; each PE's shadow register
  latches the finished sum as a job's last value passes, so jobs stream back to back while
  the drain requantizes the previous one (TFLite double/single rounding, one shared datapath).
- **Depthwise mode** without a line buffer: the array works row-staggered, so each PE row
  uses exactly the word arriving from activation memory, one tap per read.
- **Area discipline** for a 20K-LUT part: no address multipliers (counters and shift-add),
  LUT-RAM transposer/param store, and attention to Gowin's register packing (a register
  fed by another register costs a pass-through LUT).

## Results

| | |
|-|-|
| VWW latency (compute) | 258,808 cycles (ideal memory) |
| VWW on the board | ~273k cycles, ~5.1 ms at 54 MHz; weights from SDRAM (over 1-bit SPI flash: 4.2M) |
| Utilisation (board design) | LUT4 76%, DFF 55%, BSRAM 45/46, 64 DSP multipliers |
| Fmax (post-route) | ~66 MHz NPU domain |
| Accuracy | identical to TFLite int8 |

The optimisation history (O0–O4, depthwise, place-and-route) is logged in
[`.gateflow/plans/vww_npu_compiler.md`](.gateflow/plans/vww_npu_compiler.md).

## Running on the board

```bash
make -C fpga/tangnano20k TOP=npu_board_top FREQ_MHZ=54   # bitstream
make -C fpga/tangnano20k TOP=npu_board_top prog          # write it to flash
cd sw && python -m npu.board ../build/vww --image photo.jpg --out ../build/board/flash.bin --program
python sw/board/demo.py --port COM5                      # webcam -> board -> verdict
```

S1 resets, S2 reruns the demo image; LEDs show person / no person / busy / booted / error.

## Layout

```
rtl/npu/    the NPU: core, tile engine, array, requant, pool
rtl/board/  board system: flash boot, SDRAM controller, UART, HDMI overlay
sw/npu/     ISA, compiler, cycle-exact simulator, TFLite-exact quantization, flash images
sw/model/   model export (TFLite -> IR), training scripts
tb/         cocotb testbenches (core vs TFLite golden data, random networks, board system)
fpga/       Tang Nano 20K constraints and build
```

## Tests

```bash
.venv/bin/python -m pytest tb sw/tests      # Verilator + cocotb
```

The RTL is checked layer by layer against TFLite golden tensors, cycle by cycle against
the simulator's timing model, and on randomly generated networks (odd sizes, strides,
padding, group-mode convolutions).
