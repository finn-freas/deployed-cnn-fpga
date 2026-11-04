import os
import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

BIT = 8


async def send(dut, data):
    for b in data:
        for bit in [0] + [(b >> i) & 1 for i in range(8)] + [1]:
            dut.uart_rx.value = bit
            await ClockCycles(dut.clk, BIT)


async def recv(dut, n, timeout=2_000_000):
    out = bytearray()
    t = 0
    while len(out) < n:
        await RisingEdge(dut.clk)
        t += 1
        assert t < timeout, f"timeout, got {bytes(out)!r}"
        if dut.uart_tx.value == 0:
            await ClockCycles(dut.clk, BIT // 2)
            b = 0
            for i in range(8):
                await ClockCycles(dut.clk, BIT)
                b |= int(dut.uart_tx.value) << i
            await ClockCycles(dut.clk, BIT)
            out.append(b)
    return bytes(out)


def hdr(cmd, addr, n):
    return cmd + addr.to_bytes(3, "little") + n.to_bytes(3, "little")


@cocotb.test()
async def test_link(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.clk_f, int(os.environ.get("FB_FCLK_PS", "5000")), unit="ps").start())
    dut.uart_rx.value, dut.rst.value = 1, 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    while not dut.sd_ready.value:
        await RisingEdge(dut.clk)
    await send(dut, b"P")
    assert await recv(dut, 3) == b"K\x03\x01"
    await send(dut, b"D")
    d = await recv(dut, 3)
    diag = int.from_bytes(d[1:3], "little")
    dut._log.info(f"calibration: passing points {diag >> 4:04b} (bit: read latency), pick {diag & 15}")
    assert d[0:1] == b"d" and (diag >> 4) >> (diag & 15) & 1
    rng = random.Random(3)
    for addr in [0, 5, 0x3FFF7, 0xFFFF0, 0x7A123]:
        n = rng.randint(1, 9)
        words = [rng.getrandbits(64) for _ in range(n)]
        await send(dut, hdr(b"W", addr, n) + b"".join(struct.pack("<Q", w) for w in words))
        assert await recv(dut, 1) == b"w"
        await send(dut, hdr(b"R", addr, n))
        got = struct.unpack(f"<{n}Q", await recv(dut, 8 * n))
        assert list(got) == words, f"readback at {addr:#x}"
        await send(dut, hdr(b"C", addr, n))
        want = sum((w & 0xFFFFFFFF) + (w >> 32) for w in words) & 0xFFFFFFFF
        assert struct.unpack("<I", await recv(dut, 4))[0] == want
    await send(dut, b"x")
    assert await recv(dut, 1) == b"?"
    await send(dut, b"X" + struct.pack("<H", 424))
    assert await recv(dut, 1) == b"x"
    assert int(dut.u_link.e_wbase.value) == 424
    dut._log.info("link: ping, write/read/checksum at 5 addresses all pass")
