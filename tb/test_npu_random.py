"""cocotb: random small networks (sw/npu/randnet.py) on the NPU core vs the simulator.

The simulator is bit-exact with TFLite on the real models; these programs push the RTL
through shapes the models don't have (odd widths, stride-2 padded depthwise, k = 1/2
depthwise, group-mode CONV, pools with padding). Every layer's output and every
instruction's cycle count must match the simulator exactly.
NPU_RANDOM_NETS=<n> sets how many networks run (default 12), NPU_RANDOM_SEED the first seed.
"""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, First, RisingEdge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sw"))
sys.path.insert(0, str(ROOT / "tb"))
from npu.compiler import compile_ir  # noqa: E402
from npu.randnet import random_image, random_ir  # noqa: E402
from npu.sim import NpuSim  # noqa: E402
from test_npu_core import SOURCES, load_image, load_model, pulse, read_act, reset  # noqa: E402

N_NETS = int(os.environ.get("NPU_RANDOM_NETS", "12"))
SEED0 = int(os.environ.get("NPU_RANDOM_SEED", "1000"))


async def check_on_rtl(dut, prog, image, tag):
    """Run prog on the core in step mode: every layer's output and every instruction's
    cycle count must equal the simulator's. Returns the per-instruction cycles."""
    # reference: the simulator's output tensor after each instruction
    sim = NpuSim.from_program(prog)
    ref = {}
    layer_at = {L["instr"]: L for L in prog.meta["layers"]}

    def snap(pc, ins):
        if pc in layer_at:
            L = layer_at[pc]
            ref[pc] = sim.act[L["base"]:L["base"] + L["h"] * L["w"] * L["cb"]].copy()
    sim.run(image, trace=snap)

    await reset(dut)
    await load_model(dut, prog)
    await load_image(dut, image, prog.meta["input"]["base"])
    dut.step_mode.value = 1
    await pulse(dut.start, dut)
    cycles = []
    while True:
        await First(RisingEdge(dut.paused), RisingEdge(dut.done))
        await FallingEdge(dut.clk)
        if int(dut.done.value) or not int(dut.paused.value):
            break
        pc = int(dut.pc.value)
        cycles.append(int(dut.instr_cycles.value))
        if pc in layer_at:
            L = layer_at[pc]
            got = await read_act(dut, L["base"], L["h"] * L["w"] * L["cb"])
            bad = int((got != ref[pc]).sum())
            assert bad == 0, f"{tag} {L['name']} ({prog.instrs[pc]}): {bad} mismatches"
        await pulse(dut.resume, dut)
    assert int(dut.result_class.value) == sim.result[0], f"{tag}: class"
    assert cycles == sim.stats.cycles, f"{tag}: cycles {cycles} != model {sim.stats.cycles}"
    return cycles


@cocotb.test()
async def random_nets(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    done = 0
    seed = SEED0
    while done < N_NETS:
        seed += 1
        ir, arrays = random_ir(seed)
        try:
            prog = compile_ir(ir, arrays)
        except (MemoryError, NotImplementedError, AssertionError, ValueError):
            continue                                    # outside the machine's limits
        image = random_image(ir, seed)
        cycles = await check_on_rtl(dut, prog, image, f"seed {seed}")
        ops = " ".join(i.op.name for i in prog.instrs[:-2])
        dut._log.info(f"seed {seed}: {ir['input']['shape']} {ops}: bit-exact, {sum(cycles)} cycles")
        done += 1


def test_npu_random():
    from sim import run
    run("npu_sim_top", SOURCES, "test_npu_random")
