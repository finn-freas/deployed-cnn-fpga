import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B


def test_matvec_applies_multiplier_and_shift_with_rounding():
    mat = {"w": np.array([[1, 2], [3, 4]], np.int64), "m": np.array([1, 1], np.int64), "s": np.array([1, 2], np.int64)}
    assert B.matvec(mat, [3, 4]).tolist() == [6, 6]


def test_matvec_saturates_to_int16():
    mat = {"w": np.array([[1000]], np.int64), "m": np.array([100], np.int64), "s": np.array([1], np.int64)}
    assert B.matvec(mat, [1000]).tolist() == [32767]
    assert B.matvec(mat, [-1000]).tolist() == [-32768]


def test_rmsnorm_output_rms_does_not_depend_on_input_magnitude():
    rng = np.random.default_rng(0)
    x = rng.integers(-2000, 2000, 64)
    t = B.rsqrt_table()
    rms = [np.sqrt(np.mean(B.rmsnorm(x * k, 1 << (B.RSQ_BITS - 2), t).astype(float) ** 2)) for k in (1, 2, 5)]
    assert max(rms) / min(rms) < 1.02


def test_rmsnorm_is_scale_invariant_up_to_rounding():
    rng = np.random.default_rng(1)
    x = rng.integers(-500, 500, 64)
    t = B.rsqrt_table()
    a = B.rmsnorm(x, 256, t)
    b = B.rmsnorm(x * 4, 256, t)
    assert np.abs(a - b).max() <= 2


def test_rope_at_position_zero_leaves_vector_unchanged():
    c, s = B.rope_table(2, 8)
    v = np.arange(-8, 8, dtype=np.int64)
    assert B.rope(v, c[0], s[0]).tolist() == v.tolist()


def test_rope_preserves_pair_norms():
    c, s = B.rope_table(8, 8)
    v = np.array([1000, 0, 0, 1000, 700, 700, -300, 400], np.int64)
    out = B.rope(v, c[5], s[5])
    before = np.hypot(v[0::2], v[1::2])
    after = np.hypot(out[0::2], out[1::2])
    assert np.abs(before - after).max() <= 2


def test_attention_head_with_one_key_returns_its_value():
    q = np.array([3, -2], np.int64)
    ks = np.array([[5, 7]], np.int64)
    vs = np.array([[11, -13, 17]], np.int64)
    assert B.attention_head(q, ks, vs, B.exp2_table(), 4).tolist() == [11, -13, 17]


def test_attention_head_picks_the_dominant_key():
    q = np.array([1000, 0], np.int64)
    ks = np.array([[1000, 0], [-1000, 0]], np.int64)
    vs = np.array([[100], [-100]], np.int64)
    assert B.attention_head(q, ks, vs, B.exp2_table(), 8).tolist() == [100]


def test_kv8_shift_grows_with_the_peak_and_is_capped():
    assert [B.kv8_shift(a) for a in (0, 127, 128, 255, 256, 10 ** 6)] == [0, 0, 1, 2, 2, B.KV8_MAX_SHIFT]


def test_silu_mul_is_zero_for_zero_gate():
    t = B.sigmoid_table(8)
    out = B.silu_mul(np.zeros(4, np.int64), np.array([5, -5, 100, 7], np.int64), t, 8)
    assert out.tolist() == [0, 0, 0, 0]


def test_pick_without_noise_takes_the_largest_logit():
    table = np.zeros(1 << B.G_BITS, np.int64)
    out = B.pick([5, 90, 7, 3], 1 << B.INV_T_BITS, 1000, B.XorShift(3), table)
    assert out == 1


def test_pick_cutoff_excludes_logits_far_below_the_maximum():
    table = np.zeros(1 << B.G_BITS, np.int64)
    table[:] = np.arange(len(table)) - len(table) // 2
    seen = {B.pick([100, 99, 0, 0], 1 << B.INV_T_BITS, 5, B.XorShift(s), table) for s in range(1, 60)}
    assert seen <= {0, 1}
