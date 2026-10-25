"""cocotb: ONNX models compiled by npuc, on the NPU core vs the simulator.

The MNIST ONNX example plus small onnxruntime-quantized networks (QDQ and QOperator,
uint8 and int8 activations: max pooling, a flattened spatial FC, a 4x4 global average
pool lowered to CONV). Layer outputs and cycle counts must match the simulator exactly;
sw/tests/test_npuc.py checks the simulator against onnxruntime.
"""

import sys
import tempfile
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
sys.path.insert(0, str(ROOT / "tb"))
from npu.compiler import compile_ir  # noqa: E402
from test_npu_core import SOURCES  # noqa: E402
from test_npu_random import check_on_rtl  # noqa: E402

MNIST_ONNX = ROOT / "models" / "onnx" / "mnist_cnn_int8.onnx"


def _models(tmp):
    import onnx
    from npuc.testnets import SPECS, calibration_images, float_model, quantize
    if MNIST_ONNX.exists():
        d = np.load(MNIST_ONNX.parent / "mnist_test.npz")
        yield "mnist_cnn", MNIST_ONNX, d["images"][:2]
    for net, fmt, act in [("lenet", "QDQ", "uint8"), ("odd", "QOperator", "int8"),
                          ("mobile", "QDQ", "int8")]:
        spec, shape = SPECS[net]
        f = Path(tmp) / f"{net}.onnx"
        onnx.save(float_model(spec, shape), f)
        q = Path(tmp) / f"{net}_{fmt}_{act}.onnx"
        quantize(f, q, calibration_images(shape), fmt, act)
        yield f"{net} {fmt} {act}", q, calibration_images(shape, n=1, seed=5)


@cocotb.test()
async def onnx_models(dut):
    from npuc.onnx_frontend import import_onnx
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    with tempfile.TemporaryDirectory() as tmp:
        for name, path, images in _models(tmp):
            prog = compile_ir(*import_onnx(path))
            for i, img in enumerate(images):
                cycles = await check_on_rtl(dut, prog, img, f"{name} image {i}")
                dut._log.info(f"{name} image {i}: bit-exact, {sum(cycles)} cycles")


def test_npu_onnx():
    from sim import run
    run("npu_sim_top", SOURCES, "test_npu_onnx")
