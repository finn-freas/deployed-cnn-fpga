"""Host-side helpers of sw/board/board_check.py, against a fake serial port."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from board_check import ask, dq_lines, read_exact, read_line  # noqa: E402
from demo import CLK_HZ, infer  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402


class FakePort:
    """Hands out its bytes one read at a time, like a serial port with a short timeout."""

    def __init__(self, data: bytes):
        self.data = bytearray(data)

    def read(self, n=1):
        out, self.data = bytes(self.data[:n]), self.data[n:]
        return out


def test_dq_lines_names_the_differing_pins():
    assert dq_lines(0x1, 0x0) == "DQ0"
    # bits 3 and 35 are the same DQ line, on the two beats of a word
    assert dq_lines(1 << 35, 0) == "DQ3"
    assert dq_lines((1 << 35) | (1 << 3), 0) == "DQ3"
    assert dq_lines(0b101, 0) == "DQ0, DQ2"


def test_dq_lines_many_bits():
    assert dq_lines(0xFFFF_FFFF_FFFF_FFFF, 0) == "almost every bit"


def test_read_line_skips_blank_lines():
    assert read_line(FakePort(b"\r\n\npong\r\nnext\n"), 0.05) == "pong"


def test_read_line_returns_partial_text_on_timeout():
    assert read_line(FakePort(b"SDRAM: rea"), 0.05) == "SDRAM: rea"
    assert read_line(FakePort(b""), 0.05) == ""


def test_read_exact_stops_at_the_timeout():
    assert read_exact(FakePort(b"\xa5\x01\x02"), 7, 0.05) == b"\xa5\x01\x02"


class EchoPort(FakePort):
    """FakePort that records writes and the input-buffer resets."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.written = b""
        self.resets = 0

    def write(self, b):
        self.written += b

    def reset_input_buffer(self):
        self.resets += 1


def test_ask_clears_input_then_reads_the_reply_line():
    port = EchoPort(b"pong\n")
    assert ask(port, b"p", 0.05) == "pong"
    assert port.written == b"p" and port.resets == 1


def test_infer_decodes_the_reply():
    cycles = CLK_HZ // 100                       # 10 ms
    reply = bytes([0xA5, 1, 0xFE]) + cycles.to_bytes(4, "little")
    port = EchoPort(reply)
    person, score, latency_ms, _ = infer(port, np.zeros((96, 96, 3), np.uint8))
    assert person and score == -2 and latency_ms == pytest.approx(10.0)
    assert port.written[:1] == b"I" and len(port.written) == 1 + 96 * 96 * 3


def test_infer_rejects_a_short_or_wrong_reply():
    for reply in (b"", b"\xa5\x01", b"\x00" * 7):
        with pytest.raises(IOError):
            infer(EchoPort(reply), np.zeros((96, 96, 3), np.uint8))
