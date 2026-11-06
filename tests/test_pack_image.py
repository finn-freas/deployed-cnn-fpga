import copy
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import pack as P
import train as T


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(1)
    cfg = {"dim": 256, "hidden": 256, "layers": 2, "heads": 8, "kv_heads": 2, "vocab": 3072, "seq": 16, "dropout": 0.0}
    net = T.Net(cfg)
    with torch.no_grad():
        for prm in net.parameters():
            if prm.dim() == 2:
                prm.mul_(8.0)
    net.eval()
    x = torch.randint(0, cfg["vocab"], (4, cfg["seq"]))
    fmt, _ = B.calibrate(net, x)
    p = B.quantize(net, fmt)
    B.kv8_calibrate(p, [[int(t) for t in x[0]]])
    return p


def test_config_words_describe_the_model(tiny):
    img, prog = P.build(tiny)
    c = tiny["cfg"]
    w = img.words
    assert w[0] & 0xFFFF == c.dim
    assert w[0] >> 16 & 0xFFFF == c.hidden
    assert w[0] >> 32 & 0xFF == c.n_layers
    assert w[1] == c.vocab_size | c.seq_len << 16
    assert w[2] == tiny["eps"]
    assert w[4] & 0xFFFFFFFF == img.sym["ucode"][0]
    assert w[4] >> 32 == len(prog)


def test_symbols_are_contiguous_and_kv_follows_them(tiny):
    img, _ = P.build(tiny)
    pos = 0
    for name, (start, n) in img.sym.items():
        if name == "kv":
            continue
        assert start == pos
        pos += n
    assert img.sym["kv"][0] == pos == img.here()
    c = tiny["cfg"]
    assert img.sym["kv"][1] == c.n_layers * c.seq_len * 2 * c.kv_dim // 8


def test_layer_matrices_have_the_expected_size(tiny):
    img, _ = P.build(tiny)
    c = tiny["cfg"]
    assert img.sym["l0.wq"][1] == c.dim * P.row_words(c.dim)
    assert img.sym["l0.wk"][1] == c.kv_dim * P.row_words(c.dim)
    assert img.sym["l1.w2"][1] == c.dim * P.row_words(c.hidden)


def test_bytes_round_trip_the_words(tiny):
    img, _ = P.build(tiny)
    raw = img.bytes()
    assert len(raw) == img.here() * P.WORD
    assert list(struct.unpack(f"<{img.here()}Q", raw)) == img.words


def test_microcode_has_fixed_ops_per_layer(tiny):
    _, prog = P.build(tiny)
    n = tiny["cfg"].n_layers
    ops = [w & 0xF for w in prog]
    assert len(prog) == 1 + 16 * n + 5
    assert ops[0] == P.OP["EMBED"]
    assert ops[-1] == P.OP["END"]
    assert ops[-2] == P.OP["SAMPLE"]
    assert ops.count(P.OP["MV"]) == 7 * n
    assert ops.count(P.OP["ATTN"]) == n
    assert ops.count(P.OP["KVW"]) == n


def test_attn_and_kvw_carry_the_layer_index(tiny):
    _, prog = P.build(tiny)
    for kind in ("ATTN", "KVW"):
        layers = [(w >> 12 & 15) | (w >> 60) << 4 for w in prog if w & 0xF == P.OP[kind]]
        assert layers == list(range(tiny["cfg"].n_layers))


def test_build_requires_int8_kv_shifts(tiny):
    p = copy.copy(tiny)
    p.pop("kv8")
    with pytest.raises(AssertionError, match="kv8"):
        P.build(p)


def test_build_rejects_a_context_that_is_not_a_power_of_two(tiny):
    p = copy.copy(tiny)
    c = tiny["cfg"]
    p["cfg"] = type(c)(**{**vars(c), "seq_len": 12}) if not hasattr(c, "_replace") else c._replace(seq_len=12)
    with pytest.raises(AssertionError, match="power-of-two"):
        P.build(p)


def test_memory_report_mentions_free_space(tiny):
    img, _ = P.build(tiny)
    report = P.memory_report(img)
    assert "layers" in report and "KiB free" in report
    assert f"of {P.CAL_ROW}" in report
