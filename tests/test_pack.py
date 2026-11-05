import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import pack as P


def test_pack_bytes_is_little_endian_and_zero_padded():
    assert P.pack_bytes([1, 2, 3]) == [0x030201]
    assert P.pack_bytes(range(9)) == [0x0706050403020100, 8]


def test_pack_bytes_stores_signed_bytes_as_twos_complement():
    assert P.pack_bytes(np.array([-1, -128], np.int8)) == [0x80FF]


def test_pack_i16_pads_to_four_values():
    assert P.pack_i16([1, -1]) == [0x0000_0000_FFFF_0001]
    assert len(P.pack_i16(range(5))) == 2


def test_pack_nibbles_orders_low_nibble_first():
    assert P.pack_nibbles([1, 2, 3] + [0] * 13) == [0x321]
    assert P.pack_nibbles([-1] * 16) == [2 ** 64 - 1]


def test_pack_nibbles_rejects_partial_words():
    with pytest.raises(AssertionError):
        P.pack_nibbles([1] * 15)


@pytest.mark.parametrize("cols", [B.G, 2 * B.G, 8 * B.G, 9 * B.G])
def test_row_words_matches_matrix_words(cols):
    rng = np.random.default_rng(cols)
    mat = {
        "m": np.array([5, 6]),
        "s": np.array([20, 21]),
        "g8": rng.integers(-128, 128, (2, cols // B.G)),
        "q4": rng.integers(-8, 8, (2, cols)),
    }
    words = P.matrix_words(mat)
    assert len(words) == 2 * P.row_words(cols)
    rw = P.row_words(cols)
    assert words[0] == 5 | 20 << 16
    assert words[rw] == 6 | 21 << 16


def test_matrix_row_layout_round_trips():
    cols = 2 * B.G
    rng = np.random.default_rng(0)
    g8 = rng.integers(-128, 128, (1, cols // B.G))
    q4 = rng.integers(-8, 8, (1, cols))
    words = P.matrix_words({"m": np.array([3]), "s": np.array([30]), "g8": g8, "q4": q4})
    gw = -(-(cols // B.G) // 8)
    raw = struct.pack(f"<{gw}Q", *words[1:1 + gw])
    assert list(np.frombuffer(raw, np.int8)[:cols // B.G]) == list(g8[0])
    nib = np.array(words[1 + gw:], np.uint64)
    shifts = np.arange(16, dtype=np.uint64) * np.uint64(4)
    vals = ((nib[:, None] >> shifts) & np.uint64(15)).astype(np.int64).reshape(-1)
    vals = np.where(vals >= 8, vals - 16, vals)
    assert list(vals) == list(q4[0])


def test_image_records_symbols_and_serialises_words():
    img = P.Image()
    img.put("a", [1, 2])
    img.put("b", [-1])
    assert img.sym == {"a": (0, 2), "b": (2, 1)}
    assert img.here() == 3
    assert img.bytes() == struct.pack("<3Q", 1, 2, 2 ** 64 - 1)
