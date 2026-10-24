"""cocotb: spi_flash_reader against a behavioural SPI flash."""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

BASE = 0x40_0000


@cocotb.test()
async def bursts_match_flash_contents(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    rng = random.Random(3)
    dut.rst.value = 1
    dut.req.value = 0
    dut.load_we.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0

    words = [rng.getrandbits(64) for _ in range(256)]
    for w, word in enumerate(words):
        for b in range(8):
            dut.load_we.value = 1
            dut.load_addr.value = BASE + w * 8 + b
            dut.load_data.value = (word >> (8 * b)) & 0xFF
            await FallingEdge(dut.clk)
    dut.load_we.value = 0

    for start, length in [(0, 1), (5, 3), (17, 40), (200, 56), (255, 1)]:
        dut.addr.value = start
        dut.len.value = length
        dut.req.value = 1
        await FallingEdge(dut.clk)
        while not int(dut.ack.value):
            await FallingEdge(dut.clk)
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
        dut.req.value = 0
        got = []
        while len(got) < length:
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            if int(dut.rvalid.value):
                got.append(int(dut.rdata.value))
        assert got == words[start:start + length], f"burst ({start}, {length}) mismatch"
    dut._log.info("all bursts match")


def test_spi_flash():
    from sim import run
    run("flash_reader_tb_top", ["common/spi_flash_reader.sv", "sim/spi_flash_model.sv",
                                "sim/flash_reader_tb_top.sv"], "test_spi_flash")
