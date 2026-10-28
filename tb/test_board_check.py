import os
import subprocess
import sys
import tty
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
sys.path.insert(0, str(ROOT / "tb"))
from npu.board import BOOT_W, boot_words, flash_words  # noqa: E402
from npu.compiler import compile_ir, load_ir  # noqa: E402

BIT = 8
MODEL = ROOT / "models" / "ir" / "mnist"

async def load_flash(dut, words, base):
    await FallingEdge(dut.clk)
    for i, w in enumerate(words):
        dut.load_we.value = 1
        dut.load_addr.value = base + i
        dut.load_data.value = int(w)
        await FallingEdge(dut.clk)
    dut.load_we.value = 0

async def uart_to_pty(dut, fd):
    while True:
        await FallingEdge(dut.uart_tx)
        await ClockCycles(dut.clk, BIT + BIT // 2)
        v = 0
        for i in range(8):
            v |= int(dut.uart_tx.value) << i
            await ClockCycles(dut.clk, BIT)
        os.write(fd, bytes([v]))

async def run_host(dut, args):
    master, slave = os.openpty()
    tty.setraw(master)
    os.set_blocking(master, False)
    cmd = [sys.executable, str(ROOT / "sw" / "board" / "board_check.py"), "--port", os.ttyname(slave),
           "--timeout", "600"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    cocotb.start_soon(uart_to_pty(dut, master))
    while proc.poll() is None:
        try:
            data = os.read(master, 4096)
        except (BlockingIOError, OSError):
            data = b""
        for byte in data:
            for bit in [0] + [(byte >> i) & 1 for i in range(8)] + [1]:
                dut.uart_rx.value = bit
                await ClockCycles(dut.clk, BIT)
        if not data:
            await ClockCycles(dut.clk, 200)
    out = proc.stdout.read()
    os.close(master)
    os.close(slave)
    for line in out.splitlines():
        dut._log.info("host: " + line)
    return proc.returncode, out

def _mnist_flash():
    prog = compile_ir(*load_ir(MODEL))
    return prog, flash_words(prog, np.zeros((28, 28, 1), np.uint8))

@cocotb.test()
async def board_check(dut):
    top = os.environ["BOARD_CHECK_TOP"]
    period = 125 if top == "bringup_tb_top" else 10
    cocotb.start_soon(Clock(dut.clk, period, unit="ns").start())
    dut.rst.value = 1
    dut.uart_rx.value = 1
    dut.load_we.value = 0
    if top == "bringup_tb_top":
        dut.run_all.value = 0
        dut.inject.value = 0
    else:
        dut.run.value = 0
        dut.thumb_raddr.value = 0
    prog, words = _mnist_flash()
    await load_flash(dut, prog.ext_words(), 0)
    await load_flash(dut, words[BOOT_W:], BOOT_W)
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0

    if top == "bringup_tb_top":
        await ClockCycles(dut.clk, 20_000)
        code, out = await run_host(dut, ["--model", str(MODEL)])
        assert code == 0 and "all checks passed" in out, out
        for name in ("UART", "SDRAM calibration", "SDRAM test", "flash read", "flash holds"):
            assert f"[PASS] {name}" in out, (name, out)
    else:
        while not int(dut.res_valid.value):
            await ClockCycles(dut.clk, 10_000)
        code, out = await run_host(dut, ["--npu", "--model", str(MODEL), "--images", "2"])
        assert code == 0 and out.count("[PASS] inference") == 2, out

def test_board_check_bringup():
    from sim import run
    from test_bringup import SOURCES
    run("bringup_tb_top", SOURCES, "test_board_check", extra_env={"BOARD_CHECK_TOP": "bringup_tb_top"})

def test_board_check_npu():
    from sim import run
    from test_npu_system import SOURCES
    run("npu_system_tb_top", SOURCES, "test_board_check", parameters={"EXT_SDRAM": 1},
        extra_env={"BOARD_CHECK_TOP": "npu_system_tb_top"})
