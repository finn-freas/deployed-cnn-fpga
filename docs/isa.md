# NPU Instruction Set (v2)

The NPU executes a short, static program once per image. Instructions run strictly in
order; each one completes before the next starts, so there are no hazards. The machine
parameters and field layout are defined in [`sw/npu/isa.py`](../sw/npu/isa.py), which
generates [`rtl/npu/npu_pkg.sv`](../rtl/npu/npu_pkg.sv). A test fails if the two drift apart.

**v2 changes:** `DWCONV` and `AVGPOOL` added, `CONV` can read the packed input buffer
directly (`in_src = 1`), `LOAD_INPUT` removed, and fields widened for MobileNet-sized
models (160-bit instructions).

## Machine state

| Memory | Geometry | Contents |
|--------|----------|----------|
| Instruction | 64 × 160 b | Program |
| Input buffer | 32 KiB, bytes | Image, uint8, packed HWC (`addr = (y·W + x)·C + c`), written by the camera or host |
| Activation | 8192 × 64 b (address space) | Feature maps, channel-blocked (below) |
| Weight store | 65536 × 64 b (address space) | Per layer; streamed from flash/SDRAM in hardware |
| Param store | 4096 × 70 b (address space) | Per output channel `{shift[5:0], mult[31:0], bias[31:0]}`; streamed with the weights, with only the current layer buffered on chip |
| Result register | `{class[7:0], score[7:0]}` | Written by `ARGMAX` |

Physical on-chip sizes are a hardware decision (BSRAM is the tight resource). The
compiler checks every address against these spaces, and the simulator checks every access.

### Data layout

**Activations: channel-blocked HWC.** A 64-bit word holds 8 consecutive channels of
one pixel, with byte *i* = channel `8·cb + i`. A tensor with C channels uses
`Cb = ⌈C/8⌉` words per pixel:

```
addr(y, x, cb) = base + (y·W + x)·Cb + cb
```

Padding channels hold arbitrary values. Their weights are zero, so they never matter.

**CONV weights.** For output group *g* (channels `8g … 8g+7`) and reduction index *k*,
one word holds the group's 8 weights (byte *i* = output channel `8g + i`), at
`w_base + g·K + k`:

| Input | K | k order |
|-------|---|---------|
| activation memory (`in_src = 0`) | `ksize² · in_cb · 8` | `((ky·ksize + kx)·Cb + cb)·8 + c` |
| packed input (`in_src = 1`, `in_cb` = C) | `ksize · KR`, `KR = ⌈ksize · C / 8⌉ · 8` | `ky·KR + kx·C + c`; each kernel row zero-padded to KR, so every 8-value chunk is one contiguous input-row window (the gather reads it as two aligned words) |

The *k* order matches TFLite's NHWC flatten order, so a fully-connected layer is just a
CONV whose kernel covers its whole input.

**DWCONV weights.** For channel block *cb* and tap *t* = `ky·ksize + kx`, one word holds
the 8 channels' weights, at `w_base + cb·ksize² + t`.

**Params.** Entry `p_base + 8g + i` belongs to output channel `8g + i`. The compiler folds
the input zero point into the bias, `bias' = bias − in_zp · Σw`, which is why
convolutions fill padding with `in_zp` rather than 0.

## Instruction format

160 bits, fields packed LSB first:

| Field | Width | Signed | Meaning |
|-------|-------|--------|---------|
| op | 4 | | opcode |
| in_base | 16 | | input tensor base (activation memory) |
| out_base | 16 | | output tensor base |
| w_base | 16 | | weight store base |
| p_base | 12 | | param store base |
| in_h, in_w | 7, 7 | | input size; ARGMAX: `in_w` = count |
| in_cb | 6 | | input channel blocks; for `in_src = 1`, the channel count C |
| out_h, out_w | 7, 7 | | output size |
| out_g | 6 | | output channel blocks (words per output pixel) |
| ksize | 3 | | kernel / window size |
| stride | 2 | | stride |
| pad | 3 | | top/left padding; bottom/right follow from the bounds (handles TFLite SAME) |
| in_zp, out_zp | 8, 8 | ✓ | zero points (in_zp is also the padding value) |
| act_min, act_max | 8, 8 | ✓ | output clamp (ReLU = `act_min = max(−128, out_zp)`) |
| rmode | 1 | | requant rounding: 0 = double, 1 = single |
| in_src | 1 | | CONV input: 0 = activation memory, 1 = packed input buffer |
| (reserved) | 14 | | zero |

## Instructions

### `CONV` (0x1)
2-D convolution, square kernel, int8 × int8 → int32, then requantize. For each output
pixel (oy, ox) and output channel o = 8g + i:

```
acc = Σ_k A[k] · W[g][k][i]       A = input at (oy·stride − pad + ky, ox·stride − pad + kx, c),
                                  or in_zp outside the input
y   = clamp(MBQM(acc + bias'[o], mult[o], shift[o], rmode) + out_zp, act_min, act_max)
```

With `in_src = 1`, input bytes come from the packed buffer and are converted with
`int8 = (u8 + in_zp) mod 256`. With TFLite's usual input quantization (scale 1/255,
zp −128) that is exactly `u8 − 128`, and padding (u8 0) becomes `in_zp`. Outputs are
written in raster order, `out_g` words per pixel.

### `DWCONV` (0x3)
Depthwise convolution (depth multiplier 1): each channel convolves only with its own
`ksize × ksize` filter. Same padding, requantization and output layout as CONV, with
`out_g = in_cb`. The depthwise datapath supports `ksize ≤ 3` and `stride` 1 or 2.

### `MAXPOOL` (0x2) / `AVGPOOL` (0x5)
`ksize × ksize` window, stride `stride`, top/left padding `pad`, per channel; then clamp.
Input and output share quantization, so no requantization is needed.
- MAXPOOL: max over in-bounds taps.
- AVGPOOL (TFLite reference int8): `s = Σ` in-bounds taps, `n` = their count;
  `q = s > 0 ? (s + n/2) / n : (s − n/2) / n` with C truncating division, i.e. round
  half away from zero. **Not implemented in hardware:** the compiler lowers it to a DWCONV
  with all-ones weights, zero bias and zero points, and single-rounding requant by `1/n`.
  That is exact when every window lies inside the input (constant `n`) and `n` is odd
  (round-half-up then has no ties); the compiler checks the multiplier exhaustively over
  all reachable sums and rejects other pools.

### `ARGMAX` (0x4)
Over `count = in_w` int8 values stored consecutively from `in_base` (value *i* is byte
`i mod 8` of word `in_base + ⌊i/8⌋`), write the index of the **first** maximum and its
value to the result register.

### `END` (0xF) / `NOP` (0x0)
Stop and publish the result register and perf counters. / No operation.

## Requantization (`MBQM`), exactly as TFLite

| rmode | Form | Definition | TFLite op (measured) |
|-------|------|------------|----------------------|
| 0 | double rounding | `RoundingDivideByPOT(SRDHM(x << max(shift,0), mult), max(−shift,0))` | CONV_2D, DEPTHWISE_CONV_2D |
| 1 | single rounding | `(x · mult + 2^(30−shift)) >> (31 − shift)` | FULLY_CONNECTED |

Measured on TF 2.21 reference kernels (BUILTIN_REF): each form gives zero mismatches on
its own op types and thousands of mismatches (conv) or a handful (FC) on the others.
See `sw/npu/quant.py`. Single rounding requires `shift ≤ 0` (effective scale < 1), so both
forms can share one rounding datapath in hardware (see `rtl/npu/npu_requant.sv`); the
compiler rejects layers that violate this.

## Example: MLPerf Tiny VWW (MobileNetV1-0.25), first instructions

```
 0: CONV   in_base=0    out_base=5888 in_h=96 in_w=96 in_cb=3 out_h=48 out_w=48 out_g=1 ksize=3 stride=2 in_zp=-128 out_zp=-128 in_src=1
 1: DWCONV in_base=5888 out_base=0    w_base=32 p_base=8 in_h=48 in_w=48 in_cb=1 out_g=1 ksize=3 stride=1 pad=1 ...
 2: CONV   in_base=0    out_base=3584 w_base=41 p_base=16 in_cb=1 out_g=2 ksize=1 ...
 ...
27: AVGPOOL in_base=7904 out_base=0 in_h=3 in_w=3 in_cb=32 out_h=1 out_w=1 out_g=32 ksize=3 stride=3
28: CONV   in_base=0    out_base=8191 w_base=25955 p_base=2736 in_cb=32 out_g=1 ksize=1 out_zp=-5 rmode=1   (FC)
29: ARGMAX in_base=8191 in_w=2
30: END
```

31 instructions; weights 26,211 words (204.8 KiB), params 2,744.
