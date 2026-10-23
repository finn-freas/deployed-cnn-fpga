# NPU Instruction Set (v1)

The NPU executes a short, static program once per camera frame. Instructions run
strictly in order; each one completes before the next starts, so there are no
hazards. The machine parameters and field layout are defined in
[`sw/npu/isa.py`](../sw/npu/isa.py), which generates
[`rtl/npu/npu_pkg.sv`](../rtl/npu/npu_pkg.sv). A test fails if the two drift apart.

## Machine state

| Memory | Geometry | Contents |
|--------|----------|----------|
| Instruction | 32 × 128 b | Program |
| Activation | 2048 × 64 b | Feature maps, channel-blocked (below) |
| Weight | 2048 × 64 b | Per 8-output-channel group, k-major |
| Param | 256 × 70 b | Per output channel: `{shift[5:0], mult[31:0], bias[31:0]}` |
| Input buffer | H × W × 8 b | Camera crop, uint8, raster order (written by the camera subsystem) |
| Result register | `{class[7:0], score[7:0]}` | Written by ARGMAX |

### Data layout

**Activations: channel-blocked HWC.** A 64-bit word holds 8 consecutive
channels of one pixel, with byte *i* = channel `8·cb + i`. A tensor with C channels
uses `Cb = ⌈C/8⌉` words per pixel:

```
addr(y, x, cb) = base + (y·W + x)·Cb + cb
```

Padding channels (C < 8·Cb) hold arbitrary values. Their weights are zero, so they never affect results.

**Weights.** For output group *g* (channels `8g … 8g+7`) and reduction index *k*,
one word holds the 8 weights for that group, with byte *i* = output channel `8g + i`:

```
addr(g, k) = w_base + g·K + k,        K = ksize² · in_cb · 8
k = ((ky·ksize + kx)·in_cb + cb)·8 + c_in
```

The *k* order (ky, kx, channel) matches TFLite's NHWC flatten order, so a
fully-connected layer is just a CONV whose kernel covers its whole input.

**Params.** Entry `p_base + 8g + i` belongs to output channel `8g + i`. The compiler
folds the input zero point into the bias, `bias' = bias − in_zp · Σw`, which is why
CONV fills padding with `in_zp` rather than 0.

## Instruction format

128 bits, fields packed LSB first:

| Bits | Field | Width | Signed | Used by |
|------|-------|-------|--------|---------|
| 3:0 | op | 4 | | all |
| 14:4 | in_base | 11 | | CONV, MAXPOOL, ARGMAX |
| 25:15 | out_base | 11 | | CONV, MAXPOOL, LOAD_INPUT |
| 36:26 | w_base | 11 | | CONV |
| 44:37 | p_base | 8 | | CONV |
| 51:45 | in_h | 7 | | CONV, MAXPOOL, LOAD_INPUT |
| 58:52 | in_w | 7 | | CONV, MAXPOOL, LOAD_INPUT; ARGMAX: count |
| 64:59 | in_cb | 6 | | CONV, MAXPOOL |
| 71:65 | out_h | 7 | | CONV, MAXPOOL |
| 78:72 | out_w | 7 | | CONV, MAXPOOL |
| 84:79 | out_g | 6 | | CONV (output groups = output Cb), MAXPOOL (= in_cb) |
| 87:85 | ksize | 3 | | CONV, MAXPOOL (window) |
| 89:88 | stride | 2 | | CONV, MAXPOOL |
| 92:90 | pad | 3 | | CONV |
| 100:93 | in_zp | 8 | ✓ | CONV (padding value), LOAD_INPUT (offset) |
| 108:101 | out_zp | 8 | ✓ | CONV |
| 116:109 | act_min | 8 | ✓ | CONV, MAXPOOL |
| 124:117 | act_max | 8 | ✓ | CONV, MAXPOOL |
| 125 | rmode | 1 | | CONV: 0 = double rounding, 1 = single rounding |
| 127:126 | — | 2 | | reserved, zero |

## Instructions

### `LOAD_INPUT` (0x3)
For every pixel *p* of the `in_h × in_w` input buffer, write `act[out_base + p]` with
byte 0 = `(u8[p] + in_zp) mod 256` and bytes 1–7 = 0. With TFLite's usual input
quantization (scale 1/255, zp −128) this is exactly `u8 − 128`.

### `CONV` (0x1)
2-D convolution, square kernel, int8 × int8 → int32, then requantize to int8.
For each output pixel (oy, ox) and output channel o = 8g + i:

```
acc  = Σ_k A[k] · W[g][k][i]          A[k] = input at (oy·stride − pad + ky, ox·stride − pad + kx, c),
                                      or in_zp if outside the input
x    = acc + bias'[o]
y    = MBQM(x, mult[o], shift[o], rmode) + out_zp
out  = clamp(y, act_min, act_max)     (ReLU is act_min = max(−128, out_zp))
```

Output pixels are written in raster order, `out_g` words per pixel.

**Requantization (`MBQM`)** follows TFLite exactly:

| rmode | Form | Definition |
|-------|------|------------|
| 0 | double rounding | `RoundingDivideByPOT(SRDHM(x << max(shift,0), mult), max(−shift,0))` |
| 1 | single rounding | `(x · mult + 2^(30−shift)) >> (31 − shift)` |

Two forms are needed because TF 2.21's reference kernels use double rounding for
CONV_2D but single rounding for FULLY_CONNECTED. This was measured on 1000 MNIST
images: each form gives zero mismatches on its own op type, while the other form
gives thousands of mismatches (conv) or 6 (FC). See `sw/npu/quant.py`.

**Limits:** `K = ksize²·in_cb·8 ≤ 1024`. A fully-connected layer over an H×W×C input
is CONV with `ksize = H = W`, `pad = 0`, and a 1×1 output.

### `MAXPOOL` (0x2)
VALID-padded max over `ksize × ksize` windows with stride `stride`, per channel
(byte-wise on whole words), then clamp to `[act_min, act_max]`. Input and output
share quantization, so no requantization is needed.

### `ARGMAX` (0x4)
Over `count = in_w` int8 values stored consecutively from `in_base` (value *i* is
byte `i mod 8` of word `in_base + ⌊i/8⌋`), write the index of the **first** maximum
and its value to the result register.

### `END` (0xF)
Stop, publish the result register and perf counters, and wait for the next frame.

### `NOP` (0x0)
No operation.

## Example: the Stage 2 MNIST program

```
 0: LOAD_INPUT in_h=28 in_w=28 in_zp=-128
 1: CONV       out_base=1264 in_h=28 in_w=28 in_cb=1 out_h=28 out_w=28 out_g=1 ksize=3 stride=1 pad=1 in_zp=-128 out_zp=-128
 2: MAXPOOL    in_base=1264 in_h=28 in_w=28 in_cb=1 out_h=14 out_w=14 out_g=1 ksize=2 stride=2
 3: CONV       out_base=1656 w_base=72 p_base=8 in_h=14 in_w=14 in_cb=1 out_h=14 out_w=14 out_g=2 ksize=3 stride=1 pad=1 ...
 4: MAXPOOL    in_base=1656 in_h=14 in_w=14 in_cb=2 out_h=7 out_w=7 out_g=2 ksize=2 stride=2
 5: CONV       out_base=2046 w_base=216 p_base=24 in_h=7 in_w=7 in_cb=2 out_h=1 out_w=1 out_g=2 ksize=7 stride=1 ... rmode=1
 6: ARGMAX     in_base=2046 in_w=10
 7: END
```

Weights use 1784 / 2048 words, params 40 / 256, instructions 8 / 32.
