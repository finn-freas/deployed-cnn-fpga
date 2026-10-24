"""cocotb: the board-level system (npu_system) with a behavioural SPI flash.

Boots the VWW program and demo image from flash, checks the first result and the
thumbnail, then sends another image over the UART and checks the reply.
NPU_SYSTEM_MODEL=mnist|vww picks the model (default vww).
"""

import os
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, with_timeout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
from npu.board import BOOT_W, boot_words  # noqa: E402
from npu.compiler import compile_ir, load_ir  # noqa: E402

MODEL = os.environ.get("NPU_SYSTEM_MODEL", "vww")
BIT = 8                                    # UART clocks per bit (tb top: 8 MHz / 1 Mbaud)


async def load_flash(dut, words, base):
    for i, w in enumerate(words):
        dut.load_we.value = 1
        dut.load_addr.value = base + i
        dut.load_data.value = int(w)
        await FallingEdge(dut.clk)
    dut.load_we.value = 0


async def uart_send(dut, data: bytes):
    for byte in data:
        for bit in [0] + [(byte >> i) & 1 for i in range(8)] + [1]:
            dut.uart_rx.value = bit
            await ClockCycles(dut.clk, BIT)


async def uart_recv(dut, n):
    out = []
    for _ in range(n):
        await FallingEdge(dut.uart_tx)
        await ClockCycles(dut.clk, BIT + BIT // 2)          # middle of bit 0
        v = 0
        for i in range(8):
            v |= int(dut.uart_tx.value) << i
            await ClockCycles(dut.clk, BIT)
        out.append(v)
    return bytes(out)


def thumbnail(img):
    """What npu_system computes: every second pixel of every second row, (r+2g+b)>>6."""
    s = img[0:96:2, 0:96:2].astype(np.int64)
    if img.shape[2] == 1:
        return (s[..., 0] >> 4).reshape(-1)
    return ((s[..., 0] + 2 * s[..., 1] + s[..., 2]) >> 6).reshape(-1)


@cocotb.test()
async def system_boot_and_uart(dut):
    model_dir = ROOT / "build" / ("model" if MODEL == "mnist" else MODEL)
    if not (model_dir / "golden.npz").exists():
        dut._log.warning(f"no golden data in {model_dir}, skipping")
        return
    prog = compile_ir(*load_ir(model_dir))
    gold = np.load(model_dir / "golden.npz")
    last = prog.meta["layers"][-1]["name"]
    img0, img1 = gold["images"][0], gold["images"][1]

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.run.value = 0
    dut.uart_rx.value = 1
    dut.load_we.value = 0
    dut.thumb_raddr.value = 0
    await FallingEdge(dut.clk)
    await load_flash(dut, prog.ext_words(), 0)
    await load_flash(dut, boot_words(prog, img0), BOOT_W)
    dut._log.info(f"{MODEL}: flash loaded ({len(prog.ext_words())} weight/param words)")
    dut.rst.value = 0

    # ---- boot run on the demo image
    await with_timeout(RisingEdge(dut.res_valid), 200, "ms")
    await FallingEdge(dut.clk)
    assert int(dut.boot_err.value) == 0
    logits = gold[last][0].reshape(-1)
    cls, score = int(dut.res_class.value), dut.res_score.value.to_signed()
    assert cls == int(np.argmax(logits)) and score == int(logits.max()), "boot run result"
    cyc = int(dut.res_cycles.value)
    dut._log.info(f"boot run: class {cls} score {score}, {cyc} cycles with weights over 1-bit SPI")

    if img0.shape[1] == 96:
        exp = thumbnail(img0)
        for a in [0, 1, 47, 48, 1000, 2303]:
            dut.thumb_raddr.value = a
            await FallingEdge(dut.clk)
            await FallingEdge(dut.clk)
            assert int(dut.thumb_rdata.value) == exp[a], f"thumbnail[{a}]"

    # ---- an image over the UART
    in_words = -(-img1.size // 8)
    payload = np.zeros(in_words * 8, np.uint8)
    payload[:img1.size] = img1.reshape(-1)
    recv = cocotb.start_soon(uart_recv(dut, 7))
    await uart_send(dut, b"I" + payload.tobytes())
    reply = await with_timeout(recv, 200, "ms")
    logits = gold[last][1].reshape(-1)
    assert reply[0] == 0xA5, reply
    assert reply[1] == int(np.argmax(logits)), "uart class"
    assert np.int8(reply[2]) == int(logits.max()), "uart score"
    rc = int.from_bytes(reply[3:7], "little")
    assert rc == int(dut.res_cycles.value) and int(dut.res_count.value) == 2
    dut._log.info(f"uart run: class {reply[1]} score {np.int8(reply[2])}, {rc} cycles")


SOURCES = ["npu/npu_pkg.sv", "common/sdpram.sv", "npu/npu_pe.sv", "npu/npu_array.sv",
           "npu/npu_requant.sv", "npu/npu_tile_engine.sv", "npu/npu_pool.sv", "npu/npu_core.sv",
           "common/spi_flash_reader.sv", "board/uart.sv", "board/npu_system.sv",
           "sim/spi_flash_model.sv", "sim/npu_system_tb_top.sv"]


def test_npu_system():
    from sim import run
    run("npu_system_tb_top", SOURCES, "test_npu_system")
