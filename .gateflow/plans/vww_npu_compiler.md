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
- **V2 done (2026-10-23):** NPU core RTL (array, requant, tile engine, pool, sequencer).
  Requant is bit-exact on 100K random vectors. The core runs MNIST (5/5 layers) and VWW
  (29/29 layers) bit-exact vs TFLite in Verilator, step-mode checked layer by layer.
  Measured v1 latency: **VWW 832,611 cycles = 33.0 ms @ 25.2 MHz, 14.1% MAC utilization**.
  The simulator's cycle model now matches the RTL exactly (asserted in test_npu_core).
  **Synthesis probe (GW2AR-18):** logic ≈27% LUT, ≈36% FF, DSP ≈44/48 fit; **BSRAM needs
  79 of 46 blocks.** The big consumers are activation memory (8K × 64b), the 27 KB input
  buffer, and the 8 A-banks (64-bit wide × 128 deep, so 2 blocks each). V3 must restructure
  memory: byte-wide or transposed A-banks, tighter activation allocation / fusion, and the
  input and weights moved to SDRAM- or flash-backed streaming.
- **V3 (memory system) (2026-10-24):** ISA v3. The packed input lives in activation memory
  (7168 words); weights and params share one external store. The tile engine uses one
  transposed im2col buffer (A_T, 1K × 64) fed by a double-buffered 8×8 transposer, a 1K-word
  weight buffer (whole layer when it fits, otherwise per output group) and a 256-entry
  param buffer. Back-ends: `ext_mem_model` (SDRAM-like) and the real `spi_flash_reader`
  (board flash pins CS 60, CLK 59, MOSI 61, MISO 62; `gowin_pack --mspi_as_gpio`).
  **Bit-exact through both back-ends** (MNIST 5/5, VWW 29/29). VWW latency: 1,006,914 cycles
  with the SDRAM-like model (blocking loads; prefetch is O4), 5.0M cycles = 199 ms over 1-bit
  SPI flash. Cycle model exact again. **BSRAM: 79 → 43 of 46.** The transposer had to be a
  packed array (an unpacked one was inferred as 534 LUT-RAM cells). Full PnR timing pending.
- **Optimization (2026-10-24):** O3 overlapped drain (705,711), O3b double-buffered A_T
  (462,990), O4a group-mode weight prefetch (419,791), O2 packed first-layer gather (376,047),
  O3c dual requant drain (335,657), O3d gather-tail overlap (320,682). Each step bit-exact
  on both back-ends with the cycle model exact.
- **O1 depthwise datapath (2026-10-24):** DW units are 8 adjacent output pixels of one row ×
  one channel block. The input rows stream through a 17-word shift register; the array's
  depthwise mode feeds each PE (pixel × channel lane) directly. Depthwise 128,044 → 71,516
  cycles; **VWW 264,161 cycles = 10.5 ms @ 25.2 MHz**. Stride-1 units are drain-bound
  (34 cycles for 64 outputs through two requant pipelines); stride-2 units are read-bound
  (51).
- **PnR hang diagnosed (2026-10-24):** with `--placer-heap-cell-placement-timeout` nextpnr
  stops with "Unable to find legal placement for cell … MULT9X9" (an address-math multiplier
  in the pool). The default timeout is cells²/8 attempts, which looks like a hang. All
  address arithmetic is now incremental or shift-and-add, so the DSPs are only the 64 PEs
  and the two requant multipliers.
- **Second wall: LUT positions.** On Gowin a flip-flop's D input comes only from its own
  slice's LUT, so nextpnr gives every register fed by a register (or by a RAM, or by a LUT
  that already drives another register) a pass-through LUT. The utilisation report leaves
  these out. Real demand = LUT + ALU + such registers was 21.4k of 20.7k, which is why the
  placer couldn't legalise (confirmed on the routed memdp probe: every DFF sits behind its
  own LUT). Reductions: PE shadows latch the adder output on a group's *last* value
  (no capture wave); a shared single/double-rounding requant path; 26-bit accumulators;
  params, transposer (diagonal byte RAMs) and row tables in LUT RAM; AVGPOOL lowered to a
  ones-weight DWCONV (pool unit MAXPOOL-only); units read the sequencer's instruction;
  and a depthwise mode with *no line buffer*: the array keeps only its row skew, so row r
  needs exactly the word arriving from activation memory, and each tap is issued with the
  read of its word (stride 2 reads even columns before odd ones). Demand 21.4k → 15.6k
  (75%); the analytical placer now converges.
- **Full core places and routes (2026-10-24):** GW2AR-18, LUT4 14,759 (71%), ALU 4,582,
  DFF 8,263 (53%), BSRAM 43/46, DSP 64 MULT9 + 2 MULT36, LUT RAM 74. **Fmax 63.3 MHz**
  post-route (target 25.2). VWW 264,036 cycles = 10.5 ms @ 25.2 MHz, 4.2 ms at 63 MHz.
  Next: board top (flash loader, run, LEDs/HDMI), then clock up with a PLL.

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
