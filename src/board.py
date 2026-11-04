import math
from dataclasses import dataclass

import numpy as np
import torch

G = 32
Q4 = 7
F_U = 15
F_T = 8
F_ROPE = 14
F_E = 15
F_SIG = 15
SIG_STEP = 6
SIG_N = 512
RSQ_BITS = 10
RSQ_K = 26
EPS = 1e-5
LOG2E = 1.0 / math.log(2.0)
M_BITS = 15
MARGIN = 1.25
FMT_KEYS = ["x", "q", "k", "v", "h", "hb", "ug", "logit"]
G_BITS = 10
INV_T_BITS = 8


def fq4(w):
    R, C = w.shape
    wg = w.view(R, C // G, G)
    with torch.no_grad():
        amax_g = wg.abs().amax(-1).clamp_min(1e-12)
        amax_r = amax_g.amax(-1, keepdim=True)
        g8 = torch.ceil(255.0 * amax_g / amax_r).clamp(1, 255)
        step = (amax_r * g8 / (255.0 * Q4)).unsqueeze(-1)
    q = torch.clamp(torch.round(wg / step), -Q4, Q4) * step
    return (wg + (q - wg).detach()).view(R, C)


def fq8(w):
    with torch.no_grad():
        s = w.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    return w + (torch.clamp(torch.round(w / s), -127, 127) * s - w).detach()


def rsqrt_table():
    lo = 1 << (RSQ_BITS - 2)
    return np.array([round(2 ** RSQ_K / math.sqrt(m)) for m in range(lo, 1 << RSQ_BITS)], np.int64)


def exp2_table():
    return np.array([round(2 ** F_E * 2 ** (-f / 256)) for f in range(256)], np.int64)


def sigmoid_table(f_h):
    half = SIG_N // 2
    step = 2.0 ** (SIG_STEP - f_h)
    return np.array([min(round(2 ** F_SIG / (1 + math.exp(-(i - half + 0.5) * step))), 2 ** F_SIG - 1)
                     for i in range(SIG_N)], np.int64)


def rope_table(seq_len, head_size):
    pos = np.arange(seq_len)[:, None]
    freq = 10000.0 ** (-np.arange(0, head_size, 2) / head_size)[None, :]
    ang = pos * freq
    return (np.round(np.cos(ang) * 2 ** F_ROPE).astype(np.int64),
            np.round(np.sin(ang) * 2 ** F_ROPE).astype(np.int64))


def gumbel_table(f_logit):
    n = 1 << G_BITS
    return np.array([round(-math.log(-math.log((r + 0.5) / n)) * 2 ** f_logit) for r in range(n)], np.int64)


def frac_bits(amax):
    return int(min(15, max(0, 15 - math.ceil(math.log2(max(amax, 1e-6) * MARGIN)))))


@torch.no_grad()
def calibrate(net, x):
    from train import apply_rope, unit_rms
    c = net.c
    hs = c["dim"] // c["heads"]
    qc = LOG2E / math.sqrt(hs)
    amax = {k: 0.0 for k in FMT_KEYS}

    def rec(k, v):
        amax[k] = max(amax[k], float(v.abs().max()))

    net.eval()
    B, L = x.shape
    h = net.embed(x)
    rec("x", h)
    for b in net.blocks:
        u = unit_rms(h)
        q = apply_rope(b.lin(b.wq, u, b.n1.weight).view(B, L, b.nh, hs), net.cos, net.sin)
        k = apply_rope(b.lin(b.wk, u, b.n1.weight).view(B, L, b.nkv, hs), net.cos, net.sin)
        v = b.lin(b.wv, u, b.n1.weight).view(B, L, b.nkv, hs)
        rec("q", q * qc)
        rec("k", k)
        rec("v", v)
        o = torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                             is_causal=True, enable_gqa=True)
        h = h + b.lin(b.wo, o.transpose(1, 2).reshape(B, L, -1))
        rec("x", h)
        u = unit_rms(h)
        h1, h3 = b.lin(b.w1, u, b.n2.weight), b.lin(b.w3, u, b.n2.weight)
        rec("h", h1)
        rec("h", h3)
        hb = torch.nn.functional.silu(h1) * h3
        rec("hb", hb)
        h = h + b.lin(b.w2, hb)
        rec("x", h)
    ug = unit_rms(h) * net.norm.weight
    rec("ug", ug)
    rec("logit", ug @ net.table().T)
    return {k: frac_bits(v) for k, v in amax.items()}, amax


def _requant(s, f_in, f_out):
    f = s * 2.0 ** (f_out - f_in)
    e = np.floor(np.log2(f)).astype(np.int64)
    m = np.round(f / 2.0 ** e * 2 ** (M_BITS - 1)).astype(np.int64)
    sh = (M_BITS - 1) - e
    over = m >= (1 << M_BITS)
    m[over] >>= 1
    sh[over] -= 1
    assert np.all(sh >= 1), "requant shift underflow"
    return m, sh


def quant8_matrix(w, f_in, f_out):
    w = np.asarray(w, np.float64)
    s = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
    q = np.clip(np.round(w / s[:, None]), -127, 127).astype(np.int64)
    m, sh = _requant(s, f_in, f_out)
    return {"w": q, "m": m, "s": sh}


def quant4_matrix(w, f_in, f_out, extra=1.0):
    w = np.asarray(w, np.float64) * extra
    R, C = w.shape
    wg = w.reshape(R, C // G, G)
    amax_g = np.maximum(np.abs(wg).max(-1), 1e-12)
    amax_r = amax_g.max(-1, keepdims=True)
    g8 = np.clip(np.ceil(255.0 * amax_g / amax_r), 1, 255).astype(np.int64)
    step = (amax_r * g8 / (255.0 * Q4))[..., None]
    q4 = np.clip(np.round(wg / step), -Q4, Q4).astype(np.int64)
    m, sh = _requant(amax_r[:, 0] / (255.0 * Q4), f_in, f_out)
    return {"w": (q4 * g8[..., None]).reshape(R, C), "q4": q4.reshape(R, C), "g8": g8, "m": m, "s": sh}


@dataclass
class Cfg:
    dim: int
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

    @property
    def head_size(self):
        return self.dim // self.n_heads

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_size


def quantize(net, fmt):
    c = net.c
    cfg = Cfg(c["dim"], c["hidden"], c["layers"], c["heads"], c["kv_heads"], c["vocab"], c["seq"])
    d, hs = cfg.dim, cfg.head_size
    sd = math.sqrt(d)
    qc = LOG2E / math.sqrt(hs)
    f = fmt
    t = lambda p: p.detach().double().cpu().numpy()
    eps_q = max(round(EPS * d * 2 ** (2 * f["x"])), 1 << (RSQ_BITS - 2))
    p = {"cfg": cfg, "fmt": dict(fmt), "eps": eps_q, "layers": []}
    table = t(net.emb.weight)
    emb = quant8_matrix(table, 0, f["x"])
    cls = quant8_matrix(table, f["ug"], f["logit"])
    assert np.array_equal(emb["w"], cls["w"])
    p["table"] = emb["w"]
    p["emb_m"], p["emb_s"], p["cls_m"], p["cls_s"] = emb["m"], emb["s"], cls["m"], cls["s"]
    for b in net.blocks:
        ga, gf = t(b.n1.weight) * sd, t(b.n2.weight) * sd
        p["layers"].append({
            "wq": quant4_matrix(t(b.wq.weight) * ga, F_U, f["q"], qc),
            "wk": quant4_matrix(t(b.wk.weight) * ga, F_U, f["k"]),
            "wv": quant4_matrix(t(b.wv.weight) * ga, F_U, f["v"]),
            "wo": quant4_matrix(t(b.wo.weight), f["v"], f["x"]),
            "w1": quant4_matrix(t(b.w1.weight) * gf, F_U, f["h"]),
            "w3": quant4_matrix(t(b.w3.weight) * gf, F_U, f["h"]),
            "w2": quant4_matrix(t(b.w2.weight), f["hb"], f["x"]),
        })
    gfin = t(net.norm.weight) * sd
    p["f_fin"] = 15 - math.ceil(math.log2(np.abs(gfin).max() * 1.0001))
    p["gfin"] = np.round(gfin * 2 ** p["f_fin"]).astype(np.int64)
    p["rope_cos"], p["rope_sin"] = rope_table(cfg.seq_len, hs)
    p["rsqrt"] = rsqrt_table()
    p["exp2"] = exp2_table()
    p["sigmoid"] = sigmoid_table(f["h"])
    p["gumbel"] = gumbel_table(f["logit"])
    return p


def memory_map(p):
    c = p["cfg"]
    w4 = sum(L[k]["q4"].size for L in p["layers"] for k in L) // 2
    g8 = sum(L[k]["g8"].size for L in p["layers"] for k in L)
    rq = sum(L[k]["m"].size for L in p["layers"] for k in L) * 3
    table = p["table"].size + c.vocab_size * 6
    kv = 2 * c.n_layers * c.seq_len * c.kv_dim * (1 if p.get("kv8") else 2)
    return {"weights_4bit": w4, "group_scales": g8, "row_requant": rq, "table_8bit": table, "kv_cache": kv,
            "streamed_per_token": w4 + g8 + rq + table, "total": w4 + g8 + rq + table + kv}


def sat16(v):
    return np.clip(v, -32768, 32767)


def rshift_round(v, s):
    v = np.asarray(v, np.int64)
    if s <= 0:
        return v << -s
    return (v + (np.int64(1) << (s - 1))) >> s


def matvec(mat, x):
    acc = mat["w"] @ np.asarray(x, np.int64)
    return sat16((acc * mat["m"] + (np.int64(1) << (mat["s"] - 1))) >> mat["s"])


def rmsnorm(x, eps, table):
    ss = int(np.sum(np.asarray(x, np.int64) ** 2)) + eps
    sh = max(0, ss.bit_length() - RSQ_BITS)
    sh += sh & 1
    t = int(table[(ss >> sh) - (1 << (RSQ_BITS - 2))])
    return sat16(rshift_round(np.asarray(x, np.int64) * t, RSQ_K - 15 + sh // 2))


def rope(v, cos, sin):
    a, b = v[0::2], v[1::2]
    n = len(cos)
    c, s = np.tile(cos, len(a) // n), np.tile(sin, len(a) // n)
    out = np.empty_like(v)
    out[0::2] = sat16(rshift_round(a * c - b * s, F_ROPE))
    out[1::2] = sat16(rshift_round(a * s + b * c, F_ROPE))
    return out


def attention_head(q, ks, vs, exp2, shift, v_shift=0):
    t = np.clip(rshift_round(ks @ q, shift), -(1 << 17), (1 << 17) - 1)
    d = t.max() - t
    ip, fp = d >> 8, d & 255
    e = np.where(ip >= 16, 0, exp2[fp] >> np.minimum(ip, 15))
    total = int(e.sum())
    out = e @ vs
    q_ = ((np.abs(out) << v_shift) + (total >> 1)) // total
    return np.where(out < 0, -q_, q_)


KV8_MAX_SHIFT = 8


def kv8_q(v, s):
    return np.clip(rshift_round(v, s), -128, 127)


def kv8_shift(amax):
    for s in range(KV8_MAX_SHIFT + 1):
        if int(rshift_round(int(amax), s)) <= 127:
            return s
    return KV8_MAX_SHIFT


def kv8_calibrate(p, seqs, max_tokens=4000):
    q = dict(p)
    q.pop("kv8", None)
    m = IntModel(q)
    m.amax = np.zeros((q["cfg"].n_layers, 2), np.int64)
    n = 0
    for ids in seqs:
        m.reset()
        for pos, t in enumerate(ids[:q["cfg"].seq_len]):
            m.forward(int(t), pos, logits=False)
            n += 1
        if n >= max_tokens:
            break
    p["kv8"] = [(kv8_shift(k), kv8_shift(v)) for k, v in m.amax]
    t_shift = p["fmt"]["q"] + p["fmt"]["k"] - F_T
    assert all(t_shift - k >= 1 for k, _ in p["kv8"]), "K shift leaves no score shift"
    return m.amax


def silu_mul(h1, h3, table, shift):
    idx = np.clip((h1 >> SIG_STEP) + SIG_N // 2, 0, SIG_N - 1)
    s = sat16(rshift_round(h1 * table[idx], F_SIG))
    return sat16(rshift_round(s * h3, shift))


class IntModel:
    def __init__(self, p):
        self.p, self.cfg = p, p["cfg"]
        self.amax = None
        self.reset()

    def reset(self):
        c = self.cfg
        self.kc = np.zeros((c.n_layers, c.seq_len, c.kv_dim), np.int64)
        self.vc = np.zeros((c.n_layers, c.seq_len, c.kv_dim), np.int64)

    def forward(self, tok, pos, logits=True, base=0):
        p, c, f = self.p, self.cfg, self.p["fmt"]
        hs, rep = c.head_size, c.n_heads // c.n_kv_heads
        t_shift = f["q"] + f["k"] - F_T
        h_shift = 2 * f["h"] - f["hb"]
        cos, sin = p["rope_cos"][pos], p["rope_sin"][pos]
        x = sat16(rshift_round(p["table"][tok] * p["emb_m"][tok], int(p["emb_s"][tok])))
        for l, L in enumerate(p["layers"]):
            u = rmsnorm(x, p["eps"], p["rsqrt"])
            q = rope(matvec(L["wq"], u), cos, sin)
            k, v = rope(matvec(L["wk"], u), cos, sin), matvec(L["wv"], u)
            if self.amax is not None:
                self.amax[l] = np.maximum(self.amax[l], [np.abs(k).max(), np.abs(v).max()])
            sk, sv = p["kv8"][l] if p.get("kv8") else (0, 0)
            if p.get("kv8"):
                k, v = kv8_q(k, sk), kv8_q(v, sv)
            self.kc[l, pos], self.vc[l, pos] = k, v
            att = np.zeros(c.dim, np.int64)
            for h in range(c.n_heads):
                sl = slice((h // rep) * hs, (h // rep + 1) * hs)
                att[h * hs:(h + 1) * hs] = attention_head(q[h * hs:(h + 1) * hs], self.kc[l, base:pos + 1, sl],
                                                          self.vc[l, base:pos + 1, sl], p["exp2"], t_shift - sk, sv)
            x = sat16(x + matvec(L["wo"], sat16(att)))
            u = rmsnorm(x, p["eps"], p["rsqrt"])
            hb = silu_mul(matvec(L["w1"], u), matvec(L["w3"], u), p["sigmoid"], h_shift)
            x = sat16(x + matvec(L["w2"], hb))
        if not logits:
            return None
        u = rmsnorm(x, p["eps"], p["rsqrt"])
        ug = sat16(rshift_round(u * p["gfin"], F_U + p["f_fin"] - f["ug"]))
        return matvec({"w": p["table"], "m": p["cls_m"], "s": p["cls_s"]}, ug)


class XorShift:
    def __init__(self, seed):
        self.x = (seed & 0xFFFFFFFF) or 0x9E3779B9

    def next(self):
        x = self.x
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self.x = x
        return x

    def draws(self, n):
        return np.array([self.next() >> (32 - G_BITS) for _ in range(n)], np.int64)


LOG2E_Q8 = 369
GAIN_BITS = 8
MASS_BITS = 16


def class_mass(logits_q, mask, f_logit, exp2):
    lg = np.asarray(logits_q, np.int64)
    d = ((lg.max() - lg) * LOG2E_Q8) >> f_logit
    ip, fp = d >> 8, d & 255
    e = np.where(ip > 15 + MASS_BITS, 0, (exp2[fp] << MASS_BITS) >> np.minimum(ip, 15 + MASS_BITS))
    return int(e[mask].sum()), int(e.sum())


def pick(logits_q, inv_t, cut_q, rng, table, mask=None, gain_q=1 << GAIN_BITS, f_logit=0, exp2=None,
         bias_q=None):
    lg = np.asarray(logits_q, np.int64)
    keep = np.ones(len(lg), bool)
    if mask is not None:
        em, total = class_mass(lg, mask, f_logit, exp2)
        u = rng.next() >> 16
        keep = mask if (u * total) << GAIN_BITS < (em * gain_q) << 16 else ~mask
        if bias_q is not None and keep is mask:
            lg = lg + bias_q
    z = ((lg * inv_t) >> INV_T_BITS) + table[rng.draws(len(lg))]
    z[~keep | (lg < lg[keep].max() - cut_q)] = np.iinfo(np.int64).min
    return int(np.argmax(z))


LOG2_LUT = np.array([round(256 * math.log2(1 + m / 256)) for m in range(256)], np.int64)


def log2_q8(total):
    p = 0
    while total >= 512:
        total >>= 1
        p += 1
    return ((p + 8 - 31) << 8) + int(LOG2_LUT[total & 255])


def token_score(logits_q, tok, f_logit, total, w_q):
    lg = np.asarray(logits_q, np.int64)
    d = (int(lg.max() - lg[tok]) * LOG2E_Q8) >> f_logit
    return -d - log2_q8(total) + int(w_q[tok])


def score_table(mdir, key, vocab, lam=1.0, alpha=1.0, extra=None):
    import json
    from pathlib import Path
    mdir = Path(mdir)
    w = np.zeros(vocab)
    u, v = mdir / "generic_unigram.npy", mdir / f"{key}_voice.json"
    if u.exists():
        w -= alpha * np.log(np.load(u))
    if v.exists():
        for i, b in json.loads(v.read_text())["bonus"]:
            w[i] += lam * b
    for i, b in (extra or {}).items():
        w[i] += b
    w[:4] = 0.0
    return np.clip(np.round(w * math.log2(math.e) * 256), -32768, 32767).astype(np.int64)


def better(s_a, n_a, s_b, n_b):
    return s_a * n_b > s_b * n_a

