import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
from npu.quant import INT32_MAX, requantize  # noqa: E402

LATENCY = 4
N_VECTORS = 100_000

def _vectors(rng, n):
    acc = rng.integers(-(1 << 24), 1 << 24, n)
    bias = rng.integers(-(1 << 24), 1 << 24, n)
    mult = rng.integers(1 << 30, INT32_MAX, n, endpoint=True)
    mult[rng.random(n) < 0.01] = 0
    shift = rng.integers(-31, 8, n)
    rmode = rng.integers(0, 2, n)
    zp = rng.integers(-128, 128, n)
    lo = rng.integers(-128, 128, n)
    hi = np.maximum(lo, rng.integers(-128, 128, n))
    lo[:2000], hi[:2000] = -128, 127
    mult[:4], shift[:4] = INT32_MAX, [0, -31, 7, -1]
    shift[rmode == 1] = np.minimum(shift[rmode == 1], 0)
    return acc, bias, mult, shift, rmode, zp, lo, hi

@cocotb.test()
async def requant_random(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = np.random.default_rng(7)
    acc, bias, mult, shift, rmode, zp, lo, hi = _vectors(rng, N_VECTORS)
    expected = np.array([requantize(a, b, m, s, z, l, h, r) for a, b, m, s, z, l, h, r in
                         zip(acc, bias, mult, shift, zp, lo, hi, rmode)], dtype=np.int64)

    mismatches = 0
    for i in range(N_VECTORS + LATENCY):
        await FallingEdge(dut.clk)
        if i >= LATENCY:
            j = i - LATENCY
            got = dut.out.value.to_signed()
            if got != expected[j]:
                mismatches += 1
                if mismatches <= 5:
                    dut._log.error(f"vec {j}: acc={acc[j]} bias={bias[j]} mult={mult[j]} shift={shift[j]} "
                                   f"rmode={rmode[j]} zp={zp[j]} [{lo[j]},{hi[j]}] got {got} exp {expected[j]}")
        if i < N_VECTORS:
            dut.acc.value = int(acc[i])
            dut.bias.value = int(bias[i])
            dut.mult.value = int(mult[i])
            dut.shift.value = int(shift[i])
            dut.rmode.value = int(rmode[i])
            dut.out_zp.value = int(zp[i])
            dut.act_min.value = int(lo[i])
            dut.act_max.value = int(hi[i])
    assert mismatches == 0, f"{mismatches} mismatches out of {N_VECTORS}"
    dut._log.info(f"{N_VECTORS} vectors bit-exact")

def test_npu_requant():
    from sim import run
    run("npu_requant", ["npu/npu_pkg.sv", "npu/npu_requant.sv"], "test_npu_requant")
