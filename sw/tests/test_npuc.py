"""npuc: ONNX import, compilation and agreement with ONNX Runtime."""

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from npu.compiler import compile_ir, load_ir  # noqa: E402
from npuc.__main__ import main as npuc_main  # noqa: E402
from npuc.onnx_frontend import UnsupportedModel, import_onnx  # noqa: E402
from npuc.testnets import SPECS, calibration_images, float_model, quantize  # noqa: E402
from npuc.verify import verify  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MNIST_ONNX = ROOT / "models" / "onnx" / "mnist_cnn_int8.onnx"
VWW_ONNX = ROOT / "models" / "vww" / "vww_96_int8.onnx"


def _quantized(tmp_path, spec, shape, fmt="QDQ", act="uint8", seed=0):
    f = tmp_path / "float.onnx"
    onnx.save(float_model(spec, shape, seed), f)
    q = tmp_path / f"{fmt}_{act}.onnx"
    quantize(f, q, calibration_images(shape), fmt, act)
    return q


@pytest.mark.parametrize("fmt", ["QDQ", "QOperator"])
@pytest.mark.parametrize("act", ["uint8", "int8"])
@pytest.mark.parametrize("net", sorted(SPECS))
def test_testnets_match_onnxruntime(tmp_path, net, fmt, act):
    spec, shape = SPECS[net]
    q = _quantized(tmp_path, spec, shape, fmt, act)
    ir, arrays = import_onnx(q)
    prog = compile_ir(ir, arrays)
    v = verify(prog, ir, q, calibration_images(shape, n=6, seed=7))
    assert v.argmax_agree == v.images
    # single rounding vs ONNX's float rounding: only exact ties may differ
    assert all(d.max_diff <= 1 for d in v.layers), [(d.name, d.max_diff) for d in v.layers]
    exact = sum(d.exact for d in v.layers) / sum(d.values for d in v.layers)
    assert exact > 0.999, exact


@pytest.mark.skipif(not MNIST_ONNX.exists(), reason="run sw/model/make_onnx_mnist.py")
def test_mnist_onnx_accuracy():
    ir, arrays = import_onnx(MNIST_ONNX)
    prog = compile_ir(ir, arrays)
    d = np.load(MNIST_ONNX.parent / "mnist_test.npz")
    v = verify(prog, ir, MNIST_ONNX, d["images"][:300], d["labels"][:300])
    assert v.argmax_agree == v.images
    assert v.npu_correct / v.labelled > 0.95
    assert sum(d.exact for d in v.layers) / sum(d.values for d in v.layers) > 0.9999


@pytest.mark.skipif(not VWW_ONNX.exists(), reason="no converted VWW model")
def test_vww_onnx_equals_tflite_path():
    """tf2onnx's conversion of the VWW TFLite model, with TFLite rounding, compiles to the
    same memory images as the TFLite path (which is bit-exact with TFLite)."""
    ir, arrays = import_onnx(VWW_ONNX, rounding="tflite")
    prog = compile_ir(ir, arrays)
    ref_dir = ROOT / "build" / "vww"
    if (ref_dir / "ir.json").exists():
        ref = compile_ir(*load_ir(ref_dir))
        assert [i.encode() for i in prog.instrs] == [i.encode() for i in ref.instrs]
        assert prog.ext_words() == ref.ext_words()
    img = np.random.default_rng(3).integers(0, 256, (2, 96, 96, 3)).astype(np.uint8)
    assert verify(prog, ir, VWW_ONNX, img).argmax_agree == 2


# ---------------------------------------------------------------- unsupported models

def _float_graph(nodes, inits, in_shape, out="y"):
    from onnx import TensorProto, helper, numpy_helper
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("image", TensorProto.FLOAT, in_shape)],
                          [helper.make_tensor_value_info(out, TensorProto.FLOAT, None)],
                          [numpy_helper.from_array(v.astype(np.float32), k) for k, v in inits.items()])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    return m


def _quantize_model(tmp_path, m, shape):
    f = tmp_path / "f.onnx"
    onnx.save(m, f)
    q = tmp_path / "q.onnx"
    quantize(f, q, calibration_images(shape))
    return q


def test_rejects_residual(tmp_path):
    from onnx import helper
    rng = np.random.default_rng(0)
    m = _float_graph([helper.make_node("Conv", ["image", "w"], ["c"], kernel_shape=[3, 3], pads=[1] * 4),
                      helper.make_node("Add", ["c", "image"], ["y"])],
                     {"w": rng.normal(0, 0.3, (2, 2, 3, 3))}, [1, 2, 8, 8])
    with pytest.raises(UnsupportedModel, match="sequential|residual"):
        import_onnx(_quantize_model(tmp_path, m, (8, 8, 2)))


def test_rejects_grouped_conv(tmp_path):
    from onnx import helper
    rng = np.random.default_rng(0)
    m = _float_graph([helper.make_node("Conv", ["image", "w"], ["y"], kernel_shape=[3, 3], group=2)],
                     {"w": rng.normal(0, 0.3, (4, 2, 3, 3))}, [1, 4, 8, 8])
    with pytest.raises(UnsupportedModel, match="group"):
        import_onnx(_quantize_model(tmp_path, m, (8, 8, 4)))


def test_rejects_float_model(tmp_path):
    spec, shape = SPECS["odd"]
    with pytest.raises(UnsupportedModel, match="QuantizeLinear|quantize"):
        import_onnx(float_model(spec, shape))


def test_rejects_normalized_input(tmp_path):
    """Calibrating on mean/std-normalized inputs gives a non-zero input zero point."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static
    spec, shape = SPECS["odd"]
    f = tmp_path / "f.onnx"
    onnx.save(float_model(spec, shape), f)
    imgs = calibration_images(shape)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(imgs)

        def get_next(self):
            i = next(self.it, None)
            return None if i is None else {"image": ((i.transpose(2, 0, 1)[None] / 255.0 - 0.5) / 0.25).astype(np.float32)}

    q = tmp_path / "q.onnx"
    quantize_static(str(f), str(q), Reader(), quant_format=QuantFormat.QDQ)
    with pytest.raises(UnsupportedModel, match="zero point"):
        import_onnx(q)


# ---------------------------------------------------------------- command line

def test_cli_writes_outputs(tmp_path):
    spec, shape = SPECS["mobile"]
    q = _quantized(tmp_path, spec, shape)
    img = tmp_path / "demo.npy"
    np.save(img, calibration_images(shape, n=1)[0])
    out = tmp_path / "out"
    npuc_main([str(q), "-o", str(out), "--verify", "3", "--flash-image", str(img), "-q"])
    for f in ("program.hex", "ext.hex", "meta.json", "ir.json", "ir.npz", "report.md", "flash.bin"):
        assert (out / f).exists(), f
    report = (out / "report.md").read_text()
    assert "class agrees on 3/3" in report and "DWCONV" in report
    # the written IR recompiles to the same program (what `python -m npu.board` does)
    assert [i.encode() for i in compile_ir(*load_ir(out)).instrs] == \
        [int(x, 16) for x in (out / "program.hex").read_text().split()]
