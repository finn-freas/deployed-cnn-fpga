import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B


def test_xorshift_matches_reference_sequence():
    r = B.XorShift(1)
    assert [r.next() for _ in range(3)] == [270369, 67634689, 2647435461]


def test_xorshift_zero_seed_uses_default_state():
    assert B.XorShift(0).x == 0x9E3779B9
    assert B.XorShift(1 << 32).x == 0x9E3779B9


def test_xorshift_draws_fit_gumbel_table():
    d = B.XorShift(7).draws(200)
    assert d.min() >= 0 and d.max() < 1 << B.G_BITS


def test_gumbel_table_is_monotonic():
    t = B.gumbel_table(8)
    assert len(t) == 1 << B.G_BITS
    assert np.all(np.diff(t) >= 0)


def test_exp2_table_decays_from_full_scale():
    t = B.exp2_table()
    assert len(t) == 256 and t[0] == 1 << B.F_E
    assert np.all(np.diff(t) < 0)


def test_rope_table_position_zero_is_identity():
    c, s = B.rope_table(4, 8)
    assert c.shape == s.shape == (4, 4)
    assert np.all(c[0] == 1 << B.F_ROPE) and np.all(s[0] == 0)


def test_sigmoid_table_is_monotonic_and_below_full_scale():
    t = B.sigmoid_table(8)
    assert len(t) == B.SIG_N
    assert np.all(np.diff(t) >= 0) and t.max() < 1 << B.F_SIG


def test_rsqrt_table_decreases():
    t = B.rsqrt_table()
    assert np.all(np.diff(t) <= 0)


def test_frac_bits_is_clamped_to_zero_and_fifteen():
    assert B.frac_bits(1e-9) == 15
    assert B.frac_bits(1e9) == 0
    assert B.frac_bits(1.0) == 14


def test_log2_q8_is_monotonic_and_doubles_by_one_octave():
    vals = [B.log2_q8(t) for t in range(256, 5000)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert B.log2_q8(1024) - B.log2_q8(512) == 256


def test_better_compares_by_cross_multiplication():
    assert B.better(3, 2, 1, 1)
    assert not B.better(1, 2, 1, 1)
    assert not B.better(1, 1, 1, 1)
