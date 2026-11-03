import math
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

import board as B

WORD = 8
CAL_ROW = 0xFFE00
MAX_LAYERS = 32
OPS = ["END", "EMBED", "RMS", "MV", "ROPE", "KVW", "ATTN", "ADD", "SILU", "GAIN", "LOGITS", "SAMPLE"]
OP = {n: i for i, n in enumerate(OPS)}
REGS = ["X", "U", "Q", "K", "V", "ATT", "T", "H1", "H3", "HB", "UG", "LOGIT"]
R = {n: i for i, n in enumerate(REGS)}


class Image:
    def __init__(self):
        self.words = []
        self.sym = {}

    def here(self):
        return len(self.words)

    def put(self, name, ws):
        self.sym[name] = (self.here(), len(ws))
        self.words += [int(w) & (2 ** 64 - 1) for w in ws]

    def bytes(self):
        return struct.pack(f"<{len(self.words)}Q", *self.words)


def pack_bytes(vals):
    v = np.asarray(vals, np.uint64) & 0xFF
    v = np.concatenate([v, np.zeros((-len(v)) % 8, np.uint64)]).reshape(-1, 8)
    return [int(sum(int(b) << (8 * i) for i, b in enumerate(r))) for r in v]


def pack_i16(vals):
    v = np.asarray(vals, np.int64) & 0xFFFF
    v = np.concatenate([v, np.zeros((-len(v)) % 4, np.int64)]).reshape(-1, 4)
    return [int(sum(int(b) << (16 * i) for i, b in enumerate(r))) for r in v]


def pack_nibbles(q4):
    v = np.asarray(q4, np.int64) & 0xF
    assert len(v) % 16 == 0
    v = v.reshape(-1, 16)
    return [int(sum(int(b) << (4 * i) for i, b in enumerate(r))) for r in v]


def matrix_words(mat):
    out = []
    for r in range(mat["q4"].shape[0]):
        out.append(int(mat["m"][r]) | int(mat["s"][r]) << 16)
        out += pack_bytes(mat["g8"][r])
        out += pack_nibbles(mat["q4"][r])
    return out


def row_words(cols):
    return 1 + math.ceil(cols / B.G / 8) + cols // 16


TABLE_ROW = 33


def shifts(p):
    f = p["fmt"]
    return {"t": f["q"] + f["k"] - B.F_T, "h": 2 * f["h"] - f["hb"], "ug": B.F_U + p["f_fin"] - f["ug"]}


def build(p):
    c = p["cfg"]
    img = Image()
    f = p["fmt"]
    sh = shifts(p)
    assert all(1 <= v <= 40 for v in sh.values()), f"engine shift out of range: {sh}"
    ms = [int(v) for L in p["layers"] for k in L for v in L[k]["s"]] + [int(v) for v in p["cls_s"]]
    assert 16 <= min(ms) and max(ms) <= 47, f"requant shifts {min(ms)}..{max(ms)}: mv_unit takes 16..47"
    assert c.head_size == 32 and c.n_heads == 4 * c.n_kv_heads, "the RTL assumes 32-wide heads, 4 per K/V head"
    assert c.seq_len & (c.seq_len - 1) == 0 and c.seq_len <= 512, "the RTL assumes a power-of-two context <= 512"
    assert 1 <= c.n_layers <= MAX_LAYERS, f"{c.n_layers} layers: the RTL takes 1..{MAX_LAYERS}"
    assert p.get("kv8") and len(p["kv8"]) == c.n_layers, "no int8 KV shifts: run board.kv8_calibrate (scripts/kv8_add.py)"
    assert all(0 <= k <= B.KV8_MAX_SHIFT and 0 <= v <= B.KV8_MAX_SHIFT and sh["t"] - k >= 1 for k, v in p["kv8"])
    img.put("cfg", [c.dim | c.hidden << 16 | c.n_layers << 32 | c.n_heads << 40 | c.n_kv_heads << 48,
                    c.vocab_size | c.seq_len << 16,
                    p["eps"],
                    sh["t"] | sh["h"] << 8 | sh["ug"] << 16 | f["logit"] << 24,
                    0, 0, 0, 0])
    img.put("table", [w for tok in range(c.vocab_size) for w in
                      [int(p["emb_m"][tok]) | int(p["emb_s"][tok]) << 16 | int(p["cls_m"][tok]) << 24
                       | int(p["cls_s"][tok]) << 40] + pack_bytes(p["table"][tok])])
    for l, L in enumerate(p["layers"]):
        for k in ["wq", "wk", "wv", "wo", "w1", "w3", "w2"]:
            img.put(f"l{l}.{k}", matrix_words(L[k]))
    img.put("gfin", pack_i16(p["gfin"]))
    rope = np.stack([p["rope_cos"], p["rope_sin"]], -1).reshape(c.seq_len, -1)
    img.put("rope", [w for row in rope for w in pack_i16(row)])
    assert all(-32768 <= int(g) <= 32767 for g in p["gumbel"]), "the Gumbel LUT is int16 in the RTL: fmt logit too large"
    for name in ["rsqrt", "exp2", "sigmoid", "gumbel"]:
        v = [int(x) & (2 ** 64 - 1) for x in p[name]]
        if name == "exp2":
            assert max(v) < 1 << 16
            v = [x | int(m) << 16 for x, m in zip(v, B.LOG2_LUT)]
        img.put(name, v)
    w = np.asarray(p.get("score_w", np.zeros(c.vocab_size)), np.int64)
    assert len(w) == c.vocab_size == 3072 and np.abs(w).max() < 2 ** 15, "score table: 3072 int16"
    img.put("score_w", [int(v) & (2 ** 64 - 1) for v in w])
    prog = microcode(p, img)
    img.put("ucode", prog)
    kv_words = c.n_layers * c.seq_len * 2 * c.kv_dim // 8
    img.sym["kv"] = (img.here(), kv_words)
    a = lambda k: img.sym[k][0]
    img.words[3] |= a("score_w") << 32
    img.words[4:8] = [a("ucode") | len(prog) << 32, a("rope") | a("kv") << 32,
                      a("rsqrt") | a("exp2") << 32, a("sigmoid") | a("gumbel") << 32]
    assert img.here() + kv_words <= CAL_ROW,         f"image {img.here()} + KV {kv_words} words does not fit below the calibration row ({CAL_ROW})"
    return img, prog


def microcode(p, img):
    c, f = p["cfg"], p["fmt"]
    prog = []

    def op(name, src=0, dst=0, aux=0, rows=0, cols=0, addr=0):
        assert 0 <= aux < MAX_LAYERS
        prog.append(OP[name] | src << 4 | dst << 8 | (aux & 15) << 12 | rows << 16 | cols << 28 | addr << 40
                    | (aux >> 4) << 60)

    op("EMBED", dst=R["X"], addr=img.sym["table"][0])
    for l in range(c.n_layers):
        a = lambda k: img.sym[f"l{l}.{k}"][0]
        op("RMS", R["X"], R["U"], cols=c.dim)
        op("MV", R["U"], R["Q"], rows=c.dim, cols=c.dim, addr=a("wq"))
        op("ROPE", R["Q"], R["Q"], cols=c.dim)
        op("MV", R["U"], R["K"], rows=c.kv_dim, cols=c.dim, addr=a("wk"))
        op("ROPE", R["K"], R["K"], cols=c.kv_dim)
        op("MV", R["U"], R["V"], rows=c.kv_dim, cols=c.dim, addr=a("wv"))
        kv8 = p["kv8"][l][0] | p["kv8"][l][1] << 4
        op("KVW", R["K"], R["V"], aux=l, rows=kv8, cols=c.kv_dim)
        op("ATTN", R["Q"], R["ATT"], aux=l, rows=kv8, cols=c.dim)
        op("MV", R["ATT"], R["T"], rows=c.dim, cols=c.dim, addr=a("wo"))
        op("ADD", R["T"], R["X"], cols=c.dim)
        op("RMS", R["X"], R["U"], cols=c.dim)
        op("MV", R["U"], R["H1"], rows=c.hidden, cols=c.dim, addr=a("w1"))
        op("MV", R["U"], R["H3"], rows=c.hidden, cols=c.dim, addr=a("w3"))
        op("SILU", R["H1"], R["HB"], cols=c.hidden)
        op("MV", R["HB"], R["T"], rows=c.dim, cols=c.hidden, addr=a("w2"))
        op("ADD", R["T"], R["X"], cols=c.dim)
    op("RMS", R["X"], R["U"], cols=c.dim)
    op("GAIN", R["U"], R["UG"], cols=c.dim, addr=img.sym["gfin"][0])
    op("LOGITS", R["UG"], R["LOGIT"], rows=c.vocab_size, cols=c.dim, addr=img.sym["table"][0])
    op("SAMPLE", R["LOGIT"])
    op("END")
    return prog


def main():
    src = Path(sys.argv[1])
    p = pickle.loads(src.read_bytes())
    img, prog = build(p)
    out = src.with_suffix("")
    out.with_suffix(".sdram.bin").write_bytes(img.bytes())
    used = img.here() * WORD
    print(f"{out.name}: image {used / 2 ** 20:.2f} MB + KV {img.sym['kv'][1] * WORD / 2 ** 20:.2f} MB "
          f"of 8 MB, {len(prog)} microcode ops")
    print(memory_report(img))


def memory_report(img):
    reg = {}
    for name, (a, n) in img.sym.items():
        k = "layers" if name.startswith("l") and "." in name else name
        reg[k] = reg.get(k, 0) + n
    end = img.sym["kv"][0] + img.sym["kv"][1]
    return ", ".join(f"{k} {v * WORD / 1024:.0f}" for k, v in reg.items()) +         f" KiB; ends at word {end} of {CAL_ROW} ({(CAL_ROW - end) * WORD / 1024:.0f} KiB free)"


if __name__ == "__main__":
    main()
