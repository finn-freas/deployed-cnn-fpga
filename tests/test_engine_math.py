import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import engine as E
import pack as P


def test_unpack_inverts_pack_bytes():
    vals = np.array([0, 1, -1, 127, -128, 5, -77, 100, 3], np.int64)
    words = np.array(P.pack_bytes(vals), np.uint64)
    assert E.Engine.unpack(words, 8)[:len(vals)].tolist() == vals.tolist()
    assert E.Engine.unpack(words, 8, signed=False)[:3].tolist() == [0, 1, 255]


def test_unpack_inverts_pack_nibbles():
    vals = np.arange(-8, 8)
    words = np.array(P.pack_nibbles(vals), np.uint64)
    assert E.Engine.unpack(words, 4).tolist() == vals.tolist()


def test_unpack_inverts_pack_i16():
    vals = np.array([0, -1, 32767, -32768], np.int64)
    words = np.array(P.pack_i16(vals), np.uint64)
    assert E.Engine.unpack(words, 16).tolist() == vals.tolist()


def test_signed_converts_twos_complement():
    out = E.Engine._signed(np.array([0, 5, 0xFFFF, 0x8000], np.uint64), 16)
    assert out.tolist() == [0, 5, -1, -32768]
    out = E.Engine._signed(np.array([2 ** 64 - 1, 2 ** 63 - 1], np.uint64), 64)
    assert out.tolist() == [-1, 2 ** 63 - 1]


def test_sat16_clips_to_int16_range():
    assert B.sat16(np.array([-40000, -32768, 0, 32767, 40000])).tolist() == [-32768, -32768, 0, 32767, 32767]


def test_rshift_round_rounds_half_up_and_handles_left_shifts():
    assert B.rshift_round([3, 4, 5, -3, -4, -5], 1).tolist() == [2, 2, 3, -1, -2, -2]
    assert B.rshift_round([3, -3], 0).tolist() == [3, -3]
    assert B.rshift_round([3, -3], -2).tolist() == [12, -12]


def test_kv8_shift_is_smallest_shift_that_fits_int8():
    assert B.kv8_shift(127) == 0
    assert B.kv8_shift(128) == 1
    assert B.kv8_shift(10 ** 9) == B.KV8_MAX_SHIFT


def test_kv8_q_saturates():
    assert B.kv8_q(np.array([1000, -1000, 6]), 2).tolist() == [127, -128, 2]


def test_row_words_matches_matvec_row_layout():
    for cols in (32, 64, 256, 512):
        ng = cols // B.G
        assert P.row_words(cols) == 1 + -(-ng // 8) + cols // 16
