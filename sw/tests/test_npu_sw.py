"""Tests for the NPU software stack: ISA encoding, TFLite-exact requant, compiler + simulator."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from npu import isa
from npu.compiler import compile_ir, load_ir
from npu.isa import FIELDS, Instr, Op
from npu.quant import (INT32_MAX, INT32_MIN, multiply_by_quantized_multiplier,
                       quantize_multiplier, rounding_divide_by_pot, srdhm)
from npu.sim import NpuSim

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "build" / "model"


# ---------------------------------------------------------------- ISA

def test_instr_roundtrip():
    rng = random.Random(0)
    for _ in range(2000):
        kw = {}
        for name, width, signed in FIELDS:
            if name == "op":
                kw[name] = rng.choice(list(Op))
            elif signed:
                kw[name] = rng.randint(-(1 << (width - 1)), (1 << (width - 1)) - 1)
            else:
                kw[name] = rng.randint(0, (1 << width) - 1)
        ins = Instr(**kw)
        word = ins.encode()
        assert word < (1 << isa.INSTR_W)
        assert Instr.decode(word) == ins


def test_param_roundtrip():
    rng = random.Random(1)
    for _ in range(2000):
        p = (rng.randint(INT32_MIN, INT32_MAX), rng.randint(INT32_MIN, INT32_MAX), rng.randint(-32, 31))
        assert isa.decode_param(isa.encode_param(*p)) == p


def test_field_overflow_rejected():
    with pytest.raises(ValueError):
        Instr(Op.CONV, in_h=128).encode()
    with pytest.raises(ValueError):
        Instr(Op.CONV, out_zp=128).encode()


# ---------------------------------------------------------------- quantization
# Scalar re-implementations with Python ints, written straight from the TFLite C++ source,
# to cross-check the vectorized numpy versions.

def _srdhm_ref(a, b):
    if a == b == INT32_MIN:
        return INT32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else (1 - (1 << 30))
    s = ab + nudge
    return int(s / (1 << 31)) if abs(s) < (1 << 52) else (abs(s) >> 31) * (1 if s >= 0 else -1)


def _rdbp_ref(x, e):
    mask = (1 << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)


def _mbqm_ref(x, m, shift):
    left, right = max(shift, 0), max(-shift, 0)
    xs = ((x << left) + (1 << 31)) % (1 << 32) - (1 << 31)
    return _rdbp_ref(_srdhm_ref(xs, m), right)


def test_requant_matches_scalar_reference():
    rng = np.random.default_rng(2)
    n = 200_000
    x = rng.integers(INT32_MIN, INT32_MAX, n, endpoint=True)
    x[:1000] = rng.integers(-(1 << 20), 1 << 20, 1000)        # typical accumulator range
    m = rng.integers(1 << 30, INT32_MAX, n, endpoint=True)
    sh = rng.integers(-31, 8, n)
    x[0], m[0], sh[0] = INT32_MIN, INT32_MIN, 0                 # SRDHM saturation case
    got = multiply_by_quantized_multiplier(x, m, sh)
    for i in list(range(2000)) + list(rng.integers(0, n, 3000)):
        assert got[i] == _mbqm_ref(int(x[i]), int(m[i]), int(sh[i])), (x[i], m[i], sh[i])


def _single_ref(x, m, shift):
    total = 31 - shift
    return (x * m + (1 << (total - 1))) >> total


def test_single_rounding_matches_scalar_reference():
    from npu.quant import multiply_by_quantized_multiplier_single
    rng = np.random.default_rng(3)
    x = rng.integers(-(1 << 24), 1 << 24, 100_000)
    m = rng.integers(1 << 30, INT32_MAX, 100_000, endpoint=True)
    sh = rng.integers(-31, 8, 100_000)
    got = multiply_by_quantized_multiplier_single(x, m, sh)
    for i in range(0, 100_000, 7):
        assert got[i] == _single_ref(int(x[i]), int(m[i]), int(sh[i]))


def test_rounding_divide_ties():
    # Ties round away from zero: 5/2 -> 3, -5/2 -> -3, 3/2 -> 2, -3/2 -> -2
    xs = np.array([5, -5, 3, -3, 4, -4, 1, -1])
    assert list(rounding_divide_by_pot(xs, 1)) == [3, -3, 2, -2, 2, -2, 1, -1]
    assert srdhm(np.array([INT32_MIN]), np.array([INT32_MIN]))[0] == INT32_MAX


def test_quantize_multiplier():
    for real in [0.5, 0.25, 0.75, 1e-3, 0.999999999, 1.5, 3.0]:
        m, s = quantize_multiplier(real)
        assert (1 << 30) <= m < (1 << 31)
        assert abs(m * 2.0 ** (s - 31) - real) / real < 1e-9
    assert quantize_multiplier(0.0) == (0, 0)
    m, s = quantize_multiplier(0.99999999999999)      # rounds up to 2^31 -> renormalized
    assert m == 1 << 30 and s == 1


# ---------------------------------------------------------------- compiler + simulator vs TFLite

needs_model = pytest.mark.skipif(not (MODEL / "golden.npz").exists(),
                                 reason="run `make model` first")


@needs_model
def test_sim_bit_exact_vs_tflite():
    ir, arrays = load_ir(MODEL)
    prog = compile_ir(ir, arrays)
    gold = np.load(MODEL / "golden.npz")
    sim = NpuSim.from_program(prog)
    n = len(gold["labels"])
    correct = 0
    for i in range(n):
        cls, score = sim.run(gold["images"][i])
        # Earlier layers have been overwritten by ping-pong allocation; only the final
        # logits are still resident. test_every_layer_bit_exact checks each layer.
        last = prog.meta["layers"][-1]
        logits = sim.read_tensor(last["base"], 1, 1, last["cb"], last["c"]).reshape(-1)
        assert np.array_equal(logits, gold[last["name"]][i].reshape(-1)), f"image {i}: logits"
        assert cls == int(np.argmax(gold[last["name"]][i]))
        correct += cls == gold["labels"][i]
    print(f"sim accuracy {correct / n:.4f} on {n} images")


@needs_model
def test_every_layer_bit_exact():
    """Run instruction by instruction and compare each layer's output while it is resident."""
    ir, arrays = load_ir(MODEL)
    prog = compile_ir(ir, arrays)
    gold = np.load(MODEL / "golden.npz")
    by_instr = {L["instr"]: L for L in prog.meta["layers"]}
    sim = NpuSim.from_program(prog)
    for i in range(200):
        sim.input_buffer = gold["images"][i]
        for pc, ins in enumerate(prog.instrs):
            if ins.op == Op.END:
                break
            getattr(sim, "_" + ins.op.name.lower())(ins)
            if ins.op == Op.LOAD_INPUT:
                inp = prog.meta["input"]
                got = sim.read_tensor(inp["base"], inp["h"], inp["w"], 1, 1)[..., 0]
                assert np.array_equal(got, gold["input"][i]), f"image {i}: input"
            if pc in by_instr:
                L = by_instr[pc]
                got = sim.read_tensor(L["base"], L["h"], L["w"], L["cb"], L["c"])
                exp = gold[L["name"]][i].reshape(L["h"], L["w"], L["c"])
                assert np.array_equal(got, exp), (
                    f"image {i} {L['name']}: {np.count_nonzero(got != exp)} mismatches")


@needs_model
def test_cycle_estimate_report():
    ir, arrays = load_ir(MODEL)
    prog = compile_ir(ir, arrays)
    sim = NpuSim.from_program(prog)
    sim.run(np.zeros((28, 28), np.uint8))
    total = sum(sim.stats.cycles)
    macs = sum(sim.stats.macs)
    for ins, c, m in zip(prog.instrs, sim.stats.cycles, sim.stats.macs):
        print(f"{ins.op.name:10s} {ins.comment:22s} {c:7d} cycles {m:8d} MACs")
    print(f"total {total} cycles = {total / 25.2:.0f} us @ 25.2 MHz; {macs} MACs; "
          f"utilization {macs / (total * isa.N * isa.N):.1%}")
    assert total / 25.2e6 < 0.033, "must fit in one camera frame"
