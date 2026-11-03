import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import link


class FakeLink(link.Link):
    def __init__(self, replies=()):
        self.sent = bytearray()
        self.replies = bytearray(b"".join(replies))

    def drain(self, quiet=0.2):
        pass

    def send(self, data):
        self.sent += data

    def recv(self, n, timeout=5.0):
        if len(self.replies) < n:
            raise TimeoutError("short")
        out = bytes(self.replies[:n])
        del self.replies[:n]
        return out


def test_header_layout():
    assert link.Link._hdr(b"W", 0x010203, 0x0A0B0C) == b"W\x03\x02\x01\x0c\x0b\x0a"


def test_checksum_pads_to_eight_bytes():
    assert link.checksum(b"\x01\x00\x00\x00") == 1
    assert link.checksum(bytes(range(16))) == link.checksum(bytes(range(16)) + b"\x00" * 8)


def test_checksum_wraps_to_32_bits():
    data = struct.pack("<2I", 0xFFFFFFFF, 2)
    assert link.checksum(data) == 1


def test_ping_decodes_flags():
    lk = FakeLink([b"K\x07\x01"])
    assert lk.ping() == {"version": 7, "sdram_ready": True, "cal_err": False}
    assert lk.sent == b"P"


def test_ping_bad_reply_raises():
    lk = FakeLink([b"xyz", b"xyz"])
    with pytest.raises(IOError):
        lk.ping()


def test_write_frames_words():
    lk = FakeLink([b"w"])
    lk.write(5, [1, 2])
    assert lk.sent == b"W\x05\x00\x00\x02\x00\x00" + struct.pack("<2Q", 1, 2)


def test_write_requires_ack():
    with pytest.raises(IOError):
        FakeLink([b"?"]).write(0, [1])


def test_write_bytes_chunks_and_pads():
    n_words = link.CHUNK + 3
    data = bytes(8 * n_words - 5)
    lk = FakeLink([b"w", b"w"])
    seen = []
    lk.write_bytes(0, data, lambda done, n: seen.append((done, n)))
    assert seen == [(link.CHUNK, n_words), (n_words, n_words)]
    assert len(lk.sent) == 2 * 7 + 8 * n_words


def test_read_unpacks_words():
    lk = FakeLink([struct.pack("<2Q", 9, 2 ** 63)])
    assert lk.read(1, 2) == [9, 2 ** 63]
    assert lk.sent == b"R\x01\x00\x00\x02\x00\x00"


def test_seed_masks_to_32_bits():
    lk = FakeLink([b"s"])
    lk.seed(2 ** 32 + 5)
    assert lk.sent == b"S" + struct.pack("<I", 5)


def test_settings_clamps_gain():
    lk = FakeLink([b"g"])
    lk.settings(1, 2, 10 ** 6)
    assert lk.sent == b"G" + struct.pack("<HHH", 1, 2, 0xFFFF)


def test_feed_and_step():
    lk = FakeLink([b"a" + struct.pack("<H", 42)])
    assert lk.feed(300) == 42
    assert lk.sent == struct.pack("<cBH", b"F", 0, 300)
    reply = b"f" + struct.pack("<H", 7) + (1).to_bytes(6, "little") + (2).to_bytes(6, "little") + (-3).to_bytes(3, "little", signed=True)
    lk = FakeLink([reply])
    assert lk.step(1) == (7, 1, 2, -3)


def test_best_flags_and_reply():
    lk = FakeLink([b"y" + bytes([3]) + struct.pack("<i", -9) + bytes([4])])
    assert lk.best(clear=True, end=True) == (3, -9, 4)
    assert lk.sent == b"Y\x03"


def test_ack_mismatch_raises():
    with pytest.raises(IOError):
        FakeLink([b"x"]).init_engine()


def test_emoji_table_size():
    lk = FakeLink([b"e"])
    lk.emoji_table([1] * 128, [-1] * 128)
    assert len(lk.sent) == 1 + 128 * 3
