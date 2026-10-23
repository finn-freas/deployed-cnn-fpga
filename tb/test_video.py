"""cocotb tests for the portable video blocks: tmds_encoder, video_timing."""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from models.tmds import TmdsEncoder, tmds_decode


async def reset(dut, cycles=3):
    dut.rst.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0


# ---------------------------------------------------------------- tmds_encoder

def _stimulus(n, rng):
    """Mix of video runs and blanking runs, plus corner-case bytes."""
    corners = [0x00, 0xFF, 0x0F, 0xF0, 0x55, 0xAA, 0x01, 0x80, 0x10, 0xEF]
    out = []
    while len(out) < n:
        if rng.random() < 0.7:
            run = rng.randint(1, 200)
            for _ in range(run):
                d = rng.choice(corners) if rng.random() < 0.2 else rng.randrange(256)
                out.append((1, d, 0, 0))
        else:
            run = rng.randint(1, 40)
            c0, c1 = rng.randrange(2), rng.randrange(2)
            out.extend([(0, rng.randrange(256), c0, c1)] * run)
    return out[:n]


@cocotb.test()
async def tmds_matches_reference(dut):
    """RTL output equals the spec model bit-for-bit; decodes back; disparity stays bounded."""
    if not hasattr(dut, "de"):
        return
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.de.value, dut.d.value, dut.c0.value, dut.c1.value = 0, 0, 0, 0
    await reset(dut)

    rng = random.Random(1234)
    ref = TmdsEncoder()
    max_disp = 0
    for i, (de, d, c0, c1) in enumerate(_stimulus(20000, rng)):
        dut.de.value, dut.d.value, dut.c0.value, dut.c1.value = de, d, c0, c1
        expected = ref.encode(de, d, c0, c1)
        max_disp = max(max_disp, abs(ref.cnt))
        await FallingEdge(dut.clk)  # inputs sampled on the rising edge in between
        got = int(dut.q.value)
        assert got == expected, f"cycle {i}: de={de} d={d:#04x} got {got:010b} exp {expected:010b}"

        kind, val = tmds_decode(got)
        if de:
            assert kind == "data" and val == d, f"cycle {i}: decode {kind} {val} != {d:#04x}"
        else:
            assert kind == "ctrl" and val == (c1, c0), f"cycle {i}: ctrl decode {val}"

    assert max_disp <= 10, f"running disparity reached {max_disp}"
    dut._log.info(f"20000 symbols match; max |disparity| = {max_disp}")


# ---------------------------------------------------------------- video_timing

@cocotb.test()
async def timing_640x480(dut):
    """One full frame: active area, sync widths/counts, frame_start pulse."""
    if not hasattr(dut, "frame_start"):
        return
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    H_TOTAL, V_TOTAL = 800, 525
    # Align to the first frame_start
    while not int(dut.frame_start.value):
        await FallingEdge(dut.clk)

    active = hs_cycles = vs_cycles = hs_pulses = frames = 0
    hs_prev = 1
    for _ in range(H_TOTAL * V_TOTAL):
        if int(dut.active.value):
            active += 1
            assert int(dut.x.value) < 640 and int(dut.y.value) < 480
        hs = int(dut.hsync.value)
        if hs == 0:
            hs_cycles += 1
            if hs_prev == 1:
                hs_pulses += 1
        hs_prev = hs
        vs_cycles += int(dut.vsync.value) == 0
        frames += int(dut.frame_start.value)
        await FallingEdge(dut.clk)

    assert active == 640 * 480, active
    assert hs_pulses == V_TOTAL, hs_pulses
    assert hs_cycles == 96 * V_TOTAL, hs_cycles
    assert vs_cycles == 2 * H_TOTAL, vs_cycles
    assert frames == 1, frames
    assert int(dut.frame_start.value) == 1, "next frame_start should be exactly one frame later"


# ---------------------------------------------------------------- pytest entry points

def test_tmds_encoder():
    from sim import run
    run("tmds_encoder", ["video/tmds_encoder.sv"], "test_video")


def test_video_timing():
    from sim import run
    run("video_timing", ["video/video_timing.sv"], "test_video")
