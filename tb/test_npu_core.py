"""cocotb: the full NPU core running compiled MNIST / VWW programs, bit-exact vs TFLite.

For each model (skipped if its golden data hasn't been generated):
  * step mode: pause after every instruction and compare that layer's output tensor,
    read back through the debug port, with the TFLite reference-kernel golden tensor
  * full speed: final class/score vs golden, plus measured cycle counts
Image counts can be raised with NPU_CORE_IMAGES=<n> (layer-checked) and NPU_CORE_FAST=<n>.
"""

import os
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, First, RisingEdge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
from npu import isa  # noqa: E402
from npu.compiler import compile_ir, load_ir  # noqa: E402
from npu.sim import NpuSim, total_cycles  # noqa: E402

MODELS = {"mnist": ROOT / "build" / "model", "vww": ROOT / "build" / "vww"}
SEL_IMEM, SEL_ACT, SEL_EXT = 0, 2, 3
LAYER_IMAGES = int(os.environ.get("NPU_CORE_IMAGES", "3"))
FAST_IMAGES = int(os.environ.get("NPU_CORE_FAST", "10"))
EXT_FLASH = os.environ.get("NPU_EXT_FLASH", "0") == "1"         # weights over SPI flash
# RTL cycles == simulator model (the model covers the ext_mem_model back-end only)
CHECK_CYCLES = os.environ.get("NPU_CHECK_CYCLES", "0" if EXT_FLASH else "1") == "1"


# ---------------------------------------------------------------- host access helpers

async def reset(dut):
    dut.rst.value = 1
    dut.start.value = 0
    dut.step_mode.value = 0
    dut.resume.value = 0
    dut.host_we.value = 0
    dut.dbg_addr.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


async def host_write_block(dut, sel, values, base=0):
    dut.host_sel.value = sel
    for i, v in enumerate(values):
        dut.host_we.value = 1
        dut.host_addr.value = base + i
        dut.host_wdata.value = int(v)
        await FallingEdge(dut.clk)
    dut.host_we.value = 0


async def load_model(dut, prog):
    await host_write_block(dut, SEL_IMEM, [ins.encode() for ins in prog.instrs])
    await host_write_block(dut, SEL_EXT, prog.ext_words())


async def load_image(dut, image_u8, base):
    """Packed uint8 HWC bytes into activation memory at `base`, 8 bytes per word."""
    flat = np.asarray(image_u8, np.uint8).reshape(-1)
    buf = np.zeros(-(-flat.size // 8) * 8, np.uint8)
    buf[:flat.size] = flat
    await host_write_block(dut, SEL_ACT, np.frombuffer(buf.tobytes(), dtype="<u8").tolist(), base)


async def read_act(dut, base, count):
    """Read `count` activation words through the debug port (1-cycle read latency)."""
    out = np.zeros((count, isa.N), np.int8)
    dut.dbg_addr.value = base
    await FallingEdge(dut.clk)
    for i in range(count):
        if i + 1 < count:
            dut.dbg_addr.value = base + i + 1
        word = int(dut.dbg_rdata.value)
        out[i] = np.frombuffer(word.to_bytes(8, "little"), dtype=np.int8)
        await FallingEdge(dut.clk)
    return out


async def pulse(sig, dut):
    sig.value = 1
    await FallingEdge(dut.clk)
    sig.value = 0


# ---------------------------------------------------------------- per-model test body

async def run_model(dut, name):
    model_dir = MODELS[name]
    if not (model_dir / "golden.npz").exists():
        dut._log.warning(f"{name}: no golden data in {model_dir}, skipping")
        return
    prog = compile_ir(*load_ir(model_dir))
    gold = np.load(model_dir / "golden.npz")
    layer_at = {L["instr"]: L for L in prog.meta["layers"]}
    last = prog.meta["layers"][-1]

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    await load_model(dut, prog)
    dut._log.info(f"{name}: loaded {len(prog.instrs)} instrs, {prog.ext_used} external words, "
                  f"{len(prog.params)} params")

    # ---- step mode: every layer
    per_instr_cycles = []
    for i in range(min(LAYER_IMAGES, len(gold["labels"]))):
        await load_image(dut, gold["images"][i], prog.meta["input"]["base"])
        dut.step_mode.value = 1
        await pulse(dut.start, dut)
        while True:
            await First(RisingEdge(dut.paused), RisingEdge(dut.done))
            await FallingEdge(dut.clk)
            if int(dut.done.value) or not int(dut.paused.value):
                break
            pc = int(dut.pc.value)
            if i == 0:
                per_instr_cycles.append((pc, int(dut.instr_cycles.value)))
            if pc in layer_at:
                L = layer_at[pc]
                words = await read_act(dut, L["base"], L["h"] * L["w"] * L["cb"])
                got = words.reshape(L["h"], L["w"], L["cb"] * isa.N)[:, :, :L["c"]]
                exp = gold[L["name"]][i].reshape(L["h"], L["w"], L["c"])
                bad = np.count_nonzero(got != exp)
                assert bad == 0, f"image {i} {L['name']}: {bad}/{exp.size} mismatches"
            await pulse(dut.resume, dut)
        exp_cls = int(np.argmax(gold[last["name"]][i]))
        assert int(dut.result_class.value) == exp_cls, f"image {i}: class"
        dut._log.info(f"{name} image {i}: all {len(layer_at)} layers bit-exact, class {exp_cls}")

    # The simulator's cycle model must reproduce the RTL exactly.
    model = NpuSim.from_program(prog)
    model.run(gold["images"][0])
    total = sum(c for _, c in per_instr_cycles)
    for (pc, c), m in zip(per_instr_cycles, model.stats.cycles):
        dut._log.info(f"  {pc:2d} {prog.instrs[pc].op.name:8s} {prog.instrs[pc].comment:26s} "
                      f"{c:9d} cycles (model {m})")
    if CHECK_CYCLES:
        assert [c for _, c in per_instr_cycles] == model.stats.cycles, "cycle model != RTL"
    expected_total = total_cycles(model.stats, len(prog.instrs))

    # ---- full speed: final result + cycles
    dut.step_mode.value = 0
    correct = 0
    n = min(FAST_IMAGES, len(gold["labels"]))
    for i in range(n):
        await load_image(dut, gold["images"][i], prog.meta["input"]["base"])
        await pulse(dut.start, dut)
        await RisingEdge(dut.done)
        await FallingEdge(dut.clk)
        cls = int(dut.result_class.value)
        score = dut.result_score.value.to_signed()
        logits = gold[last["name"]][i].reshape(-1)
        assert cls == int(np.argmax(logits)) and score == int(logits.max()), f"image {i}: result"
        correct += cls == int(gold["labels"][i])
        cycles = int(dut.cycles.value)
        assert not CHECK_CYCLES or cycles == expected_total, \
            f"image {i}: {cycles} cycles, model says {expected_total}"
    dut._log.info(f"{name}: {n} images at full speed match TFLite; accuracy {correct}/{n}; "
                  f"{cycles} cycles/inference = {cycles / 25.2:.0f} us @ 25.2 MHz "
                  f"(per-instruction sum in step mode: {total})")


@cocotb.test()
async def core_mnist(dut):
    await run_model(dut, "mnist")


@cocotb.test()
async def core_vww(dut):
    await run_model(dut, "vww")


SOURCES = ["npu/npu_pkg.sv", "common/sdpram.sv", "npu/npu_pe.sv", "npu/npu_array.sv",
           "npu/npu_requant.sv", "npu/npu_tile_engine.sv", "npu/npu_pool.sv", "npu/npu_core.sv",
           "common/spi_flash_reader.sv", "sim/spi_flash_model.sv", "sim/ext_mem_model.sv",
           "sim/npu_sim_top.sv"]


def test_npu_core():
    from sim import run
    run("npu_sim_top", SOURCES, "test_npu_core")


def test_npu_core_spi_flash():
    """Same checks with weights/params streamed from SPI flash (the board's weight path)."""
    from sim import run
    run("npu_sim_top", SOURCES, "test_npu_core", parameters={"EXT_FLASH": 1},
        extra_env={"NPU_EXT_FLASH": "1", "NPU_CORE_IMAGES": os.environ.get("NPU_CORE_IMAGES", "1"),
                   "NPU_CORE_FAST": os.environ.get("NPU_CORE_FAST", "1")})
