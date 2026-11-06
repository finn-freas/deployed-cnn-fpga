import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import board as B
import engine as E
import pack as P
import train as T


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
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
    toks = [int(t) for t in x[0]]
    B.kv8_calibrate(p, [toks])
    return p, toks


def test_engine_logits_match_int_model(tiny):
    p, toks = tiny
    img, _ = P.build(p)
    eng = E.Engine(img.bytes())
    ref = B.IntModel(p)
    for pos, t in enumerate(toks[:6]):
        assert np.array_equal(eng.run(t, pos), ref.forward(t, pos))


def test_engine_reads_config_from_image(tiny):
    p, _ = tiny
    c = p["cfg"]
    eng = E.Engine(P.build(p)[0].bytes())
    assert (eng.dim, eng.hidden, eng.layers, eng.heads, eng.kv_heads) == (c.dim, c.hidden, c.n_layers, c.n_heads, c.n_kv_heads)
    assert (eng.vocab, eng.seq, eng.eps) == (c.vocab_size, c.seq_len, p["eps"])


def test_token_cycles_grow_with_position(tiny):
    p, _ = tiny
    assert E.token_cycles(p, 5) > E.token_cycles(p, 0) > 0
