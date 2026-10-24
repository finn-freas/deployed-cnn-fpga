# Design Plan: MLPerf Tiny Person Detection on a Custom NPU + ONNX→FPGA Compiler

**Created:** 2026-10-23
**Author:** GateFlow Planner
**Status:** Active (supersedes `yolo_trash_npu.md` and `vision_npu.md`; their Phase 0–1 work is reused)

## Goals

1. **Run the MLPerf Tiny Visual Wake Words (VWW) reference model** (person / no
   person) on our own NPU on a Sipeed Tang Nano 20K, **bit-exact against TFLite**, and
   **optimize it as far as possible**, reporting latency, throughput, utilization,
   resources and accuracy the way MLPerf Tiny does.
2. **Build `npuc`, an ONNX → FPGA compiler.** Given an ONNX model (float + calibration
   images, or pre-quantized QDQ), it produces a program and weight image for the NPU
   and loads it onto the board.

> Resume line: *designed an INT8 NPU and an ONNX compiler for it; MLPerf Tiny VWW runs
> bit-exact on a $30 FPGA at N fps / M ms, X% MAC utilization, after K measured
> optimizations.*

## The benchmark model (measured from `models/vww/vww_96_int8.tflite`, MD5 verified)

| | |
|---|---|
| Source | `mlcommons/tiny` `benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite` (Apache-2.0) |
| Architecture | MobileNetV1, width 0.25, 96×96×3 RGB input (int8, scale 1/255, zp −128) |
| Layers | 1 × CONV 3×3/s2, 13 × (DEPTHWISE 3×3 + CONV 1×1), AVGPOOL 3×3, FC 256→2, SOFTMAX |
| Params | 210,850 (≈ 216 KB with int32 biases) |
| Compute | **7.49 M MACs**: pointwise 1×1 ≈ 86%, depthwise ≈ 11%, first conv ≈ 7% |
| Peak live activations | 54 KB (layer 2: 48×48×8 in + 48×48×16 out) |
| Quantization | per-channel conv/depthwise, per-tensor FC, ReLU fused everywhere |
| Accuracy set | MLPerf `vww01`: 1000 COCO images (`models/vww/y_labels.csv`) |

SOFTMAX doesn't change the argmax, so the NPU outputs logits plus argmax and
SOFTMAX runs off-device only for score reporting. This is noted in results.

## Architecture deltas from the MNIST NPU (`vision_npu.md`)

| Area | Change |
|------|--------|
| **ISA** | Add `DWCONV` (depthwise, per-channel), `AVGPOOL` (TFLite int8 rounding), 3-channel `LOAD_INPUT`; widen fields for 256 channels and larger memories |
| **Weights** | 216 KB doesn't fit on-chip (~100 KB BSRAM total), so weights stream per layer from **SPI flash** (baseline: simple, no SDRAM risk) or **SDRAM** preloaded at boot (faster). Both get measured |
| **Activations** | On-chip, ~56 KB ping-pong. **BSRAM is the tightest resource.** Layer fusion (dw+pw with line buffers) is an optimization lever |
| **Input** | Live camera: OV7670 RGB565 → center crop → 96×96 RGB888 → NPU input buffer, or images over UART for benchmarking |
| **Output** | HDMI overlay "PERSON 93%" on live video, plus a UART benchmark mode (MLPerf-style host runner) |

## Optimization roadmap (each step measured: cycles, utilization, resources)

| Step | Lever | Why |
|------|-------|-----|
| O0 | Baseline: sequential gather → compute → drain | Reference point |
| O1 | Depthwise mode on the array (per-column A) or a dedicated DW unit | Depthwise is 11% of MACs but maps at ≤1/8 utilization |
| O2 | First-layer packing (3 channels → dense im2col words) | C=3 wastes 5/8 of each word |
| O3 | Overlap: double-buffered A banks, shadow accumulators | Hide gather and drain |
| O4 | Weight prefetch overlapped with compute; flash vs SDRAM | Hide weight-streaming latency |
| O5 | Clock: PLL to 50–75 MHz, pipelined PE / requant | Raw speed |
| O6 | dw+pw layer fusion | Cuts activation memory and traffic |
| O7 | (Open division) model-level: resolution / width / pruning search | Report separately from the closed-division reference |

Metrics: latency per inference, fps (live), MAC utilization, LUT/FF/BSRAM/DSP, accuracy on
vww01, and board power from a USB power meter (energy per inference).

## `npuc`: ONNX → FPGA compiler

```
 ONNX (float + calib images | QDQ)     TFLite (int8)
          │                                 │
   onnx frontend ──────────┐      ┌── tflite frontend (exists: sw/model/export_ir.py)
                           ▼      ▼
                       NPU IR (graph)
                           │  passes: BN folding, Conv+ReLU/Clip fusion, Flatten/Reshape
                           │  elimination, NCHW→channel-blocked HWC, quantization (PTQ),
                           │  FC→CONV, op legality checks
                           ▼
                    scheduling + tiling + memory allocation (act / weights / params)
                           ▼
                 codegen: program + weight/param images (existing backend, extended)
                           ▼
            npuc run/flash: load over UART or write to SPI flash, run, read result
```

| Feature | Detail |
|---------|--------|
| Inputs | Float ONNX + calibration images (npuc does per-channel PTQ itself), or QDQ ONNX from onnxruntime quantization |
| Op set v1 | Conv, depthwise Conv (group = C), Relu/Clip, MaxPool, AveragePool / GlobalAveragePool, Gemm/MatMul, Flatten/Reshape, BatchNorm (folded), Add (later) |
| Verification | NPU sim ≡ RTL bit-exact; vs onnxruntime: identical argmax/accuracy, int8 outputs within ±1 LSB (ORT requantizes in float, the NPU in TFLite fixed point; mismatch statistics get reported) |
| UX | `npuc compile model.onnx --calib imgs/ -o build/m` · `npuc sim` · `npuc run --board` |
| Tests | MNIST (ONNX export of our CNN), VWW (ONNX from the float reference), a small zoo of MobileNet-style models |

## Progress log

- **V1 done (2026-10-23):** ISA v2 (DWCONV, AVGPOOL, packed-input CONV). All 29 VWW layers
  bit-exact vs TFLite BUILTIN_REF on 50 images, logits bit-exact on 1000; accuracy 85.0% =
  TFLite on the dev vww01 set. That set was rebuilt from COCO val2017 because the official
  Silabs tarball returns HTTP 403 from this network, so a browser download is needed for
  exact MLPerf images. Depthwise uses double rounding. v1 cycle model baseline:
  550,776 cycles = 21.9 ms @ 25.2 MHz, 21.3% MAC utilization. Time split: pointwise 57.6%
  (30.5% util, drain-bound), depthwise 31.5% (7.2% util), packed first conv 10.7%.
  So the optimization order is O1 (depthwise), then O3 (overlap the drain), then O2.

## Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| **V1** | VWW in the NumPy simulator | TFLite frontend + ISA v2 (DWCONV, AVGPOOL, RGB input); sim bit-exact vs TFLite (BUILTIN_REF) on every layer; vww01 accuracy reproduced |
| **V2** | NPU RTL core (old Phases 2–4) | MNIST, then VWW, bit-exact in RTL sim (Verilator/cocotb) |
| **V3** | Weight streaming + memory system | SPI-flash (and/or SDRAM) weight path in RTL sim; full VWW in RTL sim |
| **V4** | Board: UART benchmark mode | Images over UART → result; latency measured on hardware |
| **V5** | Board: live camera + HDMI overlay | Standalone person detector |
| **V6** | Optimization (O1–O6) | Each step measured; final numbers vs published MLPerf Tiny results |
| **C1** | `npuc` ONNX frontend + PTQ | MNIST and VWW from ONNX compile and run bit-exact in sim |
| **C2** | `npuc` board loader | `npuc run --board model.onnx` works end to end |
| **C3** | Write-up | README: architecture, ISA, compiler, verification, results tables, demo video |

C1 can start in parallel with V2/V3, since it only needs the simulator.

## Risks

| ID | Risk | Prob. | Impact | Mitigation |
|----|------|-------|--------|------------|
| R1 | BSRAM: 56 KB activations + param memory + A banks + video exceed 46 blocks | Med | High | Allocation per layer, dw+pw fusion, video frame buffer kept tiny (or in SDRAM) |
| R2 | Weight streaming: flash pins usable as user I/O in the open flow / SDRAM in the open flow | Med | High | SPI flash first (simpler); SDRAM pins already accepted by nextpnr; Gowin EDA fallback |
| R3 | TFLite rounding of AVGPOOL / DWCONV / per-tensor FC | Med | Med | The same per-op measurement method that found the conv/FC rounding difference |
| R4 | vww01 preprocessing must match MLPerf exactly to reproduce accuracy | Med | Med | Use the mlcommons conversion scripts verbatim |
| R5 | ONNX ↔ TFLite quantization differences | High | Low | Bit-exact target is our own sim; ORT compared on accuracy + LSB statistics |
| R6 | Camera pins / OV7670 bring-up | Med | Med | UART benchmark mode first (V4) needs no camera |

---
*Generated by GateFlow Planner*
