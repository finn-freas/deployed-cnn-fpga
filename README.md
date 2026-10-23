# deployed-cnn-fpga

A standalone INT8 vision accelerator ("mini NPU") and its compiler, running on a
Sipeed Tang Nano 20K. A camera feeds live video in, an 8×8 systolic array runs a
quantized CNN on every frame, and the prediction is drawn over the video on an
HDMI monitor. No PC is needed at runtime.

> Status: early development. The design plan is in [`.gateflow/plans/vision_npu.md`](.gateflow/plans/vision_npu.md).

## Layout

```
rtl/        SystemVerilog (npu/, video/, camera/, common/, top/)
tb/         cocotb testbenches (Verilator)
formal/     SymbiYosys proofs
sw/         model training, golden reference, compiler
fpga/       board constraints and build scripts
docs/       ISA and design notes
```

## Environments

| Env | Python | Used for |
|-----|--------|----------|
| `.venv` | 3.14 | cocotb simulation, NumPy reference |
| `.venv-tf` | 3.12 | TensorFlow training + TFLite export only |
| `~/oss-cad-suite` | — | Verilator, Yosys, nextpnr-himbaechel, openFPGALoader |

```bash
make env        # create both Python environments
```
