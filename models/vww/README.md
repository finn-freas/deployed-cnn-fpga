# MLPerf Tiny Visual Wake Words reference model

Files copied unmodified from [mlcommons/tiny](https://github.com/mlcommons/tiny)
(Apache License 2.0), `benchmark/training/visual_wake_words/`:

| File | Upstream path |
|------|---------------|
| `vww_96_int8.tflite` | `trained_models/vww_96_int8.tflite` (MD5 `f0b011416abee0343a5d130cb1f4c18f`, matches `vww_96_int8.md5`) |
| `vww_96_int8.md5` | `trained_models/vww_96_int8.md5` |
| `vww_96_float.tflite` | `trained_models/vww_96_float.tflite` |
| `vww_model.py.ref` | `vww_model.py` (model definition, for reference) |
| `y_labels.csv` | `benchmark/evaluation/datasets/vww01/y_labels.csv` (1000-image accuracy set: file, classes, label) |

MobileNetV1 (width 0.25), 96×96×3 int8 input, 210,850 parameters, 7.49 M MACs.
