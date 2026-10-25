"""cocotb: the board-level system (npu_system) with a behavioural SPI flash.

Boots the VWW program and demo image from flash, checks the first result and the
thumbnail, then sends another image over the UART and checks the reply.
NPU_SYSTEM_MODEL=mnist|vww picks the model (default vww). Without the TFLite golden data
(build/, needs TensorFlow) the committed ONNX models are compiled with npuc instead and the
simulator provides the expected results.
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


def load_case():
    """-> (program, two images, expected logits per image)"""
    model_dir = ROOT / "build" / ("model" if MODEL == "mnist" else MODEL)
    if (model_dir / "golden.npz").exists():
        prog = compile_ir(*load_ir(model_dir))
        gold = np.load(model_dir / "golden.npz")
        last = prog.meta["layers"][-1]["name"]
        return prog, gold["images"][:2], [gold[last][i].reshape(-1) for i in range(2)]
    from npu.sim import NpuSim
    from npuc.onnx_frontend import import_onnx
    if MODEL == "mnist":
        prog = compile_ir(*import_onnx(ROOT / "models" / "onnx" / "mnist_cnn_int8.onnx"))
        images = np.load(ROOT / "models" / "onnx" / "mnist_test.npz")["images"][:2]
    else:   # tf2onnx's conversion of the VWW model: identical images to the TFLite path
        prog = compile_ir(*import_onnx(ROOT / "models" / "vww" / "vww_96_int8.onnx", rounding="tflite"))
        images = np.random.default_rng(0).integers(0, 256, (2, 96, 96, 3)).astype(np.uint8)
    last = prog.meta["layers"][-1]
    logits = []
    for img in images:
        sim = NpuSim.from_program(prog)
        sim.run(img)
        logits.append(sim.read_tensor(last["base"], 1, 1, last["cb"], last["c"]).reshape(-1))
    return prog, images, logits


def thumbnail(img):
    """What npu_system computes: every second pixel of every second row, (r+2g+b)>>6."""
    s = img[0:96:2, 0:96:2].astype(np.int64)
    if img.shape[2] == 1:
        return (s[..., 0] >> 4).reshape(-1)
    return ((s[..., 0] + 2 * s[..., 1] + s[..., 2]) >> 6).reshape(-1)


@cocotb.test()
async def system_boot_and_uart(dut):
    prog, (img0, img1), expect = load_case()

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
    await with_timeout(RisingEdge(dut.res_valid), 2000, "ms")
    await FallingEdge(dut.clk)
    assert int(dut.boot_err.value) == 0
    logits = expect[0]
    cls, score = int(dut.res_class.value), dut.res_score.value.to_signed()
    assert cls == int(np.argmax(logits)) and score == int(logits.max()), "boot run result"
    cyc = int(dut.res_cycles.value)
    dut._log.info(f"boot run: class {cls} score {score}, {cyc} cycles")

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
    logits = expect[1]
    assert reply[0] == 0xA5, reply
    assert reply[1] == int(np.argmax(logits)), "uart class"
    assert np.int8(reply[2]) == int(logits.max()), "uart score"
    rc = int.from_bytes(reply[3:7], "little")
    assert rc == int(dut.res_cycles.value) and int(dut.res_count.value) == 2
    dut._log.info(f"uart run: class {reply[1]} score {np.int8(reply[2])}, {rc} cycles")


SOURCES = ["npu/npu_pkg.sv", "common/sdpram.sv", "common/sdpram_dc.sv", "npu/npu_pe.sv", "npu/npu_array.sv",
           "npu/npu_requant.sv", "npu/npu_tile_engine.sv", "npu/npu_pool.sv", "npu/npu_core.sv",
           "common/spi_flash_reader.sv", "board/uart.sv", "board/sdram_ctrl.sv",
           "board/npu_system.sv", "sim/spi_flash_model.sv", "sim/sdram_model.sv",
           "sim/npu_system_tb_top.sv"]


def test_npu_system():
    """Weights over 1-bit SPI flash."""
    from sim import run
    run("npu_system_tb_top", SOURCES, "test_npu_system")


def test_npu_system_sdram():
    """Weights copied to SDRAM at boot, inference reads them from there."""
    from sim import run
    run("npu_system_tb_top", SOURCES, "test_npu_system", parameters={"EXT_SDRAM": 1})
