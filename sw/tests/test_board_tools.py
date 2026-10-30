import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from board_check import ask, dq_lines, read_exact, read_line  # noqa: E402
from demo import CLK_HZ, infer  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

class FakePort:

    def __init__(self, data: bytes):
        self.data = bytearray(data)

    def read(self, n=1):
        out, self.data = bytes(self.data[:n]), self.data[n:]
        return out

def test_dq_lines_names_the_differing_pins():
    assert dq_lines(0x1, 0x0) == "DQ0"
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
    cycles = CLK_HZ // 100
    reply = bytes([0xA5, 1, 0xFE]) + cycles.to_bytes(4, "little")
    port = EchoPort(reply)
    person, score, latency_ms, _ = infer(port, np.zeros((96, 96, 3), np.uint8))
    assert person and score == -2 and latency_ms == pytest.approx(10.0)
    assert port.written[:1] == b"I" and len(port.written) == 1 + 96 * 96 * 3

def test_infer_rejects_a_short_or_wrong_reply():
    for reply in (b"", b"\xa5\x01", b"\x00" * 7):
        with pytest.raises(IOError):
            infer(EchoPort(reply), np.zeros((96, 96, 3), np.uint8))

def _fake_program(n_instrs, ext_len=3, h=2, w=3, c=1, base=7):
    from types import SimpleNamespace
    from npu.isa import Instr, Op
    ins = [Instr(op=list(Op)[0]) for _ in range(n_instrs)]
    return SimpleNamespace(meta={"input": {"h": h, "w": w, "c": c, "base": base}}, instrs=ins,
                           ext_words=lambda: list(range(1, ext_len + 1)))

def test_boot_words_header_and_layout():
    from npu import board, isa
    prog = _fake_program(3)
    img = np.arange(6, dtype=np.uint8)
    words = board.boot_words(prog, img)
    beats = 3 * (isa.INSTR_W // 32)
    beats += beats % 2
    hdr = words[0]
    assert hdr & 0xFFFF == beats
    assert (hdr >> 16) & 0xFFFF == 7
    assert (hdr >> 32) & 0xFFFF == 1
    assert (hdr >> 48) & 0x7F == 3 and (hdr >> 55) & 0x7 == 1
    assert hdr >> 58 == board.MAGIC
    assert words[1] == 3
    assert len(words) == 2 + beats // 2 + 1
    assert words[-1].to_bytes(8, "little") == bytes(range(6)) + b"\0\0"

def test_flash_words_pads_the_external_store_to_the_boot_block():
    from npu import board
    prog = _fake_program(1)
    words = board.flash_words(prog, np.zeros(6, np.uint8))
    assert words[:3] == [1, 2, 3] and not any(words[3:board.BOOT_W])
    assert words[board.BOOT_W:] == board.boot_words(prog, np.zeros(6, np.uint8))

def test_image_sum_adds_both_halves_mod_2_32():
    from npu.board import image_sum
    assert image_sum([]) == 0
    assert image_sum([0x0000_0002_0000_0001]) == 3
    assert image_sum([0xFFFF_FFFF_FFFF_FFFF, 2]) == (0xFFFF_FFFF * 2 + 2) & 0xFFFF_FFFF

def test_load_image_reshapes_npy(tmp_path):
    from npu.board import load_image
    f = tmp_path / "x.npy"
    np.save(f, np.arange(12).reshape(2, 6))
    out = load_image(f, 2, 2, 3)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8 and out[1, 1, 2] == 11

def test_font_rom_is_in_sync_with_generator():
    from gen_font import GLYPHS, render

    rom = Path(__file__).resolve().parents[2] / "rtl" / "board" / "font_rom.sv"
    assert rom.read_text() == render(), "run python sw/board/gen_font.py"
    for c, rows in GLYPHS.items():
        assert len(rows) == 7 and all(len(r) == 5 and set(r) <= {"#", "."} for r in rows), c
    assert all(ord(c) < 128 for c in GLYPHS)

def test_load_image_rejects_a_missing_file(tmp_path):
    from npu.board import load_image
    with pytest.raises(SystemExit, match="no such image file"):
        load_image(tmp_path / "gone.npy", 2, 2, 3)

def test_load_image_rejects_a_wrongly_sized_npy(tmp_path):
    from npu.board import load_image
    f = tmp_path / "img.npy"
    np.save(f, np.zeros(5, np.uint8))
    with pytest.raises(SystemExit, match="5 values, the model input 2x2x3 needs 12"):
        load_image(f, 2, 2, 3)

def test_main_reports_a_missing_openfpgaloader(tmp_path, monkeypatch):
    from npu import board

    def missing(*a, **k):
        raise FileNotFoundError("openFPGALoader")

    monkeypatch.setattr(board.subprocess, "run", missing)
    monkeypatch.setattr(board, "compile_ir", lambda *a: SimpleNamespace(
        meta={"input": {"h": 1, "w": 1, "c": 1, "base": 0}}, instrs=[], ext_words=lambda: []))
    monkeypatch.setattr(board, "load_ir_or_exit", lambda d: ((), ()))
    monkeypatch.setattr(sys, "argv", ["board", str(tmp_path), "--program", "--out", str(tmp_path / "f.bin")])
    with pytest.raises(SystemExit, match="openFPGALoader is not installed"):
        board.main()
    assert (tmp_path / "f.bin").exists()
