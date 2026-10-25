# Exported IR

`sw/model/export_ir.py` output for the two TFLite models (`make ir`, `make vww` refresh these
copies): `ir.json` (layers, shapes, quantization) and `ir.npz` (int8 weights, int32 biases,
per-channel scales). They let the compiler, the board tools and the tests run without
TensorFlow:

    cd sw && python -m npu.compiler ../models/ir/vww --out ../build/npu_vww

| | Source |
|-|-|
| `vww/` | `models/vww/vww_96_int8.tflite` (MLPerf Tiny, Apache License 2.0) |
| `mnist/` | `models/mnist_int8.tflite` |

The golden tensors used for the bit-exactness tests need TensorFlow and stay in `build/`.
