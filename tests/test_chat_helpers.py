import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chat as C
import tok as T


def plain(n):
    return [t for t in range(4, T.VOCAB) if not C.EMOJI[t]][:n]


class WordTok:
    def __init__(self, words):
        self.words = words

    def decode(self, toks):
        return " ".join(self.words[t] for t in toks)


def test_make_rng_is_deterministic_and_seed_dependent():
    a = [C.make_rng(7).next() for _ in range(3)]
    assert a == [C.make_rng(7).next() for _ in range(3)]
    assert C.make_rng(7).next() != C.make_rng(8).next()


def test_make_rng_masks_seed_to_32_bits():
    assert C.make_rng(5).x == C.make_rng(5 + (1 << 32)).x


def test_pick_seed_numeric(capsys):
    assert C.pick_seed("42") == 42
    assert capsys.readouterr().out.strip() == "seed 42"


def test_pick_seed_random_in_range(capsys):
    seed = C.pick_seed("random")
    assert 0 <= seed < 1_000_000
    assert capsys.readouterr().out.strip() == f"seed {seed}"


def test_pick_seed_rejects_garbage():
    with pytest.raises(ValueError):
        C.pick_seed("abc")


def test_emoji_scale():
    assert C.emoji_scale(0.5) == 1.0
    assert C.emoji_scale(0) == 0.0
    assert C.emoji_scale(-3) == 0.0
    assert C.emoji_scale(1.0) == 4.0


def test_emoji_mask_shape():
    assert C.EMOJI.shape == (T.VOCAB,)
    assert C.EMOJI.dtype == bool


def test_loops_detects_repeated_trigram():
    base = plain(6)
    assert not C.loops(base)
    assert C.loops(base[:3] + base[:3])


def test_loops_detects_triple_repeat():
    t = plain(1)[0]
    assert C.loops([t, t, t])
    assert not C.loops([t, t])


def test_loops_ignores_him_marker():
    a, b = plain(2)
    assert not C.loops([a, T.HIM, b, T.HIM, a])
    assert C.loops([a, T.HIM, a, T.HIM, a])


def test_blocked_matches_slurs_and_groups():
    assert C.blocked(WordTok(["kys now"]), [0])
    assert C.blocked(WordTok(["you are gay"]), [0])
    assert not C.blocked(WordTok(["nice shoes"]), [0])


def test_has_mom_variants():
    tk = WordTok(["ur mom", "Your Mama", "you’re momma", "mom", "your moment"])
    assert C.has_mom(tk, [0])
    assert C.has_mom(tk, [1])
    assert C.has_mom(tk, [2])
    assert not C.has_mom(tk, [3])
    assert not C.has_mom(tk, [4])


def test_mom_bad_requires_aggression_and_no_recent_repeat():
    tk = WordTok(["ur mom", "hello"])

    class P:
        said = []

    assert C.mom_bad(P, tk, [0], "hello there")
    assert not C.mom_bad(P, tk, [0], "you stupid")
    assert not C.mom_bad(P, tk, [1], "hello there")

    class Q:
        said = [[0]]

    assert C.mom_bad(Q, tk, [0], "you stupid")


def test_mom_bad_normalises_curly_apostrophe():
    tk = WordTok(["ur mom"])

    class P:
        said = []

    assert not C.mom_bad(P, tk, [0], "shut up")
    assert C.mom_bad(P, tk, [0], "nothing")
