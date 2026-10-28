import re
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, with_timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
from npu.board import BOOT_W, flash_words, image_sum  # noqa: E402
from npu.compiler import compile_ir, load_ir  # noqa: E402

BIT = 8
MEM_WORDS = 4096

async def uart_send(dut, data: bytes):
    for byte in data:
        for bit in [0] + [(byte >> i) & 1 for i in range(8)] + [1]:
            dut.uart_rx.value = bit
            await ClockCycles(dut.clk, BIT)

async def uart_line(dut):
    out = bytearray()
    while True:
        await FallingEdge(dut.uart_tx)
        await ClockCycles(dut.clk, BIT + BIT // 2)
        v = 0
        for i in range(8):
            v |= int(dut.uart_tx.value) << i
            await ClockCycles(dut.clk, BIT)
        if v == 0x0A:
            return out.decode().strip()
        out.append(v)

async def ask(dut, cmd: bytes, timeout_ms=50):
    line = cocotb.start_soon(uart_line(dut))
    await uart_send(dut, cmd)
    return await with_timeout(line, timeout_ms, "ms")

async def load_flash(dut, words, base):
    await FallingEdge(dut.clk)
    for i, w in enumerate(words):
        dut.load_we.value = 1
        dut.load_addr.value = base + i
        dut.load_data.value = int(w)
        await FallingEdge(dut.clk)
    dut.load_we.value = 0

@cocotb.test()
async def bringup(dut):
    cocotb.start_soon(Clock(dut.clk, 125, unit="ns").start())
    dut.rst.value = 1
    dut.run_all.value = 0
    dut.uart_rx.value = 1
    dut.inject.value = 0
    dut.load_we.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0

    lines = []
    while not (lines and lines[-1].startswith("SDRAM")):
        lines.append(await with_timeout(uart_line(dut), 20, "ms"))
    assert any(line.startswith("NPU bring-up") for line in lines), lines
    sd = lines[-1]
    assert re.fullmatch(r"SDRAM: ready, read latency CAS-1\+[0-3]", sd), sd
    dut._log.info(sd)

    assert await ask(dut, b"p") == "pong"
    assert await ask(dut, b"s") == sd
    assert (await ask(dut, b"x")).startswith("? commands")

    r = await ask(dut, b"m", 200)
    assert r == f"mem: PASS, 0x{MEM_WORDS:05x} words x 2 patterns", r
    assert int(dut.status.value) & 0b01000

    dut.inject.value = 1
    r = await ask(dut, b"m", 200)
    dut.inject.value = 0
    m = re.fullmatch(r"mem: FAIL, 0x([0-9a-f]{8}) errors, first @([0-9a-f]{5}) got ([0-9a-f]{16}) exp ([0-9a-f]{16})", r)
    assert m, r
    got, exp = int(m[3], 16), int(m[4], 16)
    assert int(m[2], 16) == 0 and got ^ exp == (1 << 5) | (1 << 37), r
    assert int(dut.status.value) & 0b10000
    dut._log.info(r)

    r = await ask(dut, b"f", 50)
    assert r.startswith("flash: no model image at 0x00480000"), r

    prog = compile_ir(*load_ir(ROOT / "models" / "ir" / "mnist"))
    img = np.zeros((28, 28, 1), np.uint8)
    words = flash_words(prog, img)
    ext = prog.ext_words()
    await load_flash(dut, ext, 0)
    await load_flash(dut, words[BOOT_W:BOOT_W + 2], BOOT_W)
    r = await ask(dut, b"f", 2000)
    m = re.fullmatch(r"flash: image found, header ([0-9a-f]{16}), 0x([0-9a-f]{5}) words, sum ([0-9a-f]{8})", r)
    assert m, r
    assert int(m[1], 16) == words[BOOT_W] and int(m[2], 16) == len(ext), r
    assert int(m[3], 16) == image_sum(ext), (r, hex(image_sum(ext)))
    dut._log.info(r)

    lines = cocotb.start_soon(_two_lines(dut))
    await uart_send(dut, b"a")
    first, second = await with_timeout(lines, 3000, "ms")
    assert first.startswith("mem: PASS") and second.startswith("flash: image found"), (first, second)
    assert int(dut.status.value) & 0b01000 and not int(dut.status.value) & 0b10000

async def _two_lines(dut):
    return await uart_line(dut), await uart_line(dut)

SOURCES = ["npu/npu_pkg.sv", "common/spi_flash_reader.sv", "board/uart.sv", "board/sdram_ctrl.sv",
           "board/bringup.sv", "sim/spi_flash_model.sv", "sim/sdram_model.sv", "sim/bringup_tb_top.sv"]

def test_bringup():
    from sim import run
    run("bringup_tb_top", SOURCES, "test_bringup")
