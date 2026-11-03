import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import board as B
import pack as P


def make_case(rng, rows, cols):
    w = rng.normal(0, 1, (rows, cols)) * rng.uniform(0.2, 3, (rows, 1)) * rng.uniform(0.3, 1, (1, cols))
    mat = B.quant4_matrix(w, B.F_U, int(rng.integers(8, 13)))
    x = rng.integers(-32768, 32768, cols) if rng.random() < 0.3 else rng.integers(-3000, 3000, cols)
    return mat, x


async def run_case(dut, rng, mat, x, gap):
    rows, cols = mat["q4"].shape
    for i in range(cols // 16):
        word = 0
        for k in range(16):
            word |= (int(x[16 * i + k]) & 0xFFFF) << (16 * k)
        dut.xw_en.value, dut.xw_addr.value, dut.xw_data.value = 1, i, word
        await RisingEdge(dut.clk)
    dut.xw_en.value = 0
    dut.rows.value, dut.cols.value, dut.start.value = rows, cols, 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    words = P.matrix_words(mat)
    got = []

    async def collect():
        while len(got) < rows:
            await RisingEdge(dut.clk)
            if dut.o_valid.value:
                got.append(dut.o_data.value.to_signed())

    col = cocotb.start_soon(collect())
    i = 0
    while i < len(words):
        if rng.random() < gap:
            dut.s_valid.value = 0
        else:
            dut.s_valid.value, dut.s_data.value = 1, words[i]
            await RisingEdge(dut.clk)
            if dut.s_ready.value:
                i += 1
            continue
        await RisingEdge(dut.clk)
    dut.s_valid.value = 0
    await col
    want = B.matvec(mat, x)
    assert got == [int(v) for v in want], f"mismatch at rows {[r for r in range(rows) if got[r] != want[r]][:5]}"


@cocotb.test()
async def test_matvec(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value, dut.start.value, dut.s_valid.value, dut.xw_en.value = 1, 0, 0, 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    rng = np.random.default_rng(1)
    n = 0
    for rows, cols in [(16, 256), (24, 704), (64, 256), (8, 704), (40, 256)]:
        for gap in (0.0, 0.5):
            mat, x = make_case(rng, rows, cols)
            await run_case(dut, rng, mat, x, gap)
            n += rows
    dut._log.info(f"{n} rows bit-exact")
