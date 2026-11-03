import pickle
import sys
from pathlib import Path

import numpy as np

import board as B
import pack as P

CLK_HZ = 54e6
WORDS_PER_CYCLE = 0.5
LANES = 8
ROW_OVERHEAD = 6


def token_cycles(p, pos):
    c = p["cfg"]
    mats = [(c.dim, c.dim), (c.kv_dim, c.dim), (c.kv_dim, c.dim), (c.dim, c.dim),
            (c.hidden, c.dim), (c.hidden, c.dim), (c.dim, c.hidden)]
    cyc = c.n_layers * sum(r * (P.row_words(k) / WORDS_PER_CYCLE + ROW_OVERHEAD) for r, k in mats)
    cyc += c.n_layers * (2 * (c.dim + 40) + (c.dim + c.kv_dim) // 2 + c.kv_dim + 2 * c.dim + c.hidden)
    cyc += c.n_layers * (pos + 1) * (c.kv_dim // 2 / WORDS_PER_CYCLE + c.n_heads * 2 * c.head_size / LANES)
    cyc += c.dim + 40 + c.dim + c.vocab_size * (32 / WORDS_PER_CYCLE + 2) + 3 * c.vocab_size
    return cyc


class Engine:
    def __init__(self, image_bytes):
        self.w = np.frombuffer(image_bytes, np.uint64).copy()
        cfg = [int(x) for x in self.w[:8]]
        self.dim, self.hidden = cfg[0] & 0xFFFF, cfg[0] >> 16 & 0xFFFF
        self.layers, self.heads, self.kv_heads = cfg[0] >> 32 & 0xFF, cfg[0] >> 40 & 0xFF, cfg[0] >> 48 & 0xFF
        self.vocab, self.seq = cfg[1] & 0xFFFF, cfg[1] >> 16 & 0xFFFF
        self.eps = cfg[2]
        self.t_shift, self.h_shift = cfg[3] & 0xFF, cfg[3] >> 8 & 0xFF
        self.ug_shift, self.f_logit = cfg[3] >> 16 & 0xFF, cfg[3] >> 24 & 0xFF
        lo, hi = (lambda v: v & 0xFFFFFFFF), (lambda v: v >> 32)
        ucode, n_ops = lo(cfg[4]), hi(cfg[4])
        self.rope_base, self.kv_base = lo(cfg[5]), hi(cfg[5])
        self.hs = self.dim // self.heads
        self.kv_dim = self.kv_heads * self.hs
        self.kvw = self.kv_dim // 8
        self.prog = self.w[ucode:ucode + n_ops].copy()
        self.w = np.concatenate([self.w, np.zeros(self.layers * self.seq * 2 * self.kvw, np.uint64)])
        n_rs, n_sig, n_g = (1 << B.RSQ_BITS) - (1 << (B.RSQ_BITS - 2)), B.SIG_N, 1 << B.G_BITS
        self.rsqrt = self._signed(self.w[lo(cfg[6]):lo(cfg[6]) + n_rs], 64)
        self.exp2 = self._signed(self.w[hi(cfg[6]):hi(cfg[6]) + 256] & 0xFFFF, 64)
        self.sigmoid = self._signed(self.w[lo(cfg[7]):lo(cfg[7]) + n_sig], 64)
        self.gumbel = self._signed(self.w[hi(cfg[7]):hi(cfg[7]) + n_g], 64)
        self.reg = {}
        self.words_read = 0
        self.cycles = 0

    def kv_addr(self, layer, v):
        return self.kv_base + (2 * layer + v) * self.seq * self.kvw

    @staticmethod
    def _signed(a, bits):
        a = np.asarray(a, np.uint64).astype(object)
        return np.array([int(v) - (1 << bits) if int(v) >> (bits - 1) else int(v) for v in a], np.int64)

    def read(self, addr, n):
        self.words_read += n
        return self.w[addr:addr + n]

    @staticmethod
    def unpack(ws, bits, signed=True):
        per = 64 // bits
        ws = np.asarray(ws, np.uint64)
        sh = (np.arange(per, dtype=np.uint64) * np.uint64(bits))
        v = ((ws[:, None] >> sh[None, :]) & np.uint64((1 << bits) - 1)).astype(np.int64).reshape(-1)
        if signed:
            v = np.where(v >= 1 << (bits - 1), v - (1 << bits), v)
        return v

    def matvec(self, addr, rows, cols, x):
        ng = cols // B.G
        gw = -(-ng // 8)
        rw = P.row_words(cols)
        out = np.empty(rows, np.int64)
        x = np.asarray(x, np.int64)
        for r in range(rows):
            ws = self.read(addr + r * rw, rw)
            m, s = int(ws[0]) & 0xFFFF, int(ws[0]) >> 16 & 0xFF
            g8 = self.unpack(ws[1:1 + gw], 8, signed=False)[:ng]
            q4 = self.unpack(ws[1 + gw:], 4)
            part = (q4.reshape(ng, B.G) * x.reshape(ng, B.G)).sum(1)
            acc = int((part * g8).sum())
            out[r] = (acc * m + (1 << (s - 1))) >> s
        self.cycles += rows * (rw / WORDS_PER_CYCLE + ROW_OVERHEAD)
        return B.sat16(out)

    def run(self, tok, pos, logits=True, trace=None):
        TR = P.TABLE_ROW
        for word in self.prog:
            word = int(word)
            op = P.OPS[word & 0xF]
            src, dst, aux = word >> 4 & 0xF, word >> 8 & 0xF, (word >> 12 & 0xF) | (word >> 60 & 1) << 4
            rows, cols, addr = word >> 16 & 0xFFF, word >> 28 & 0xFFF, word >> 40 & 0xFFFFF
            g = lambda r: self.reg[P.REGS[r]]
            if op == "END":
                break
            if op == "EMBED":
                blk = self.read(addr + tok * TR, TR)
                row = self.unpack(blk[1:], 8)
                em, es = int(blk[0]) & 0xFFFF, int(blk[0]) >> 16 & 0xFF
                self.reg["X"] = B.sat16(B.rshift_round(row * em, es))
                dst = P.R["X"]
            elif op == "RMS":
                self.reg[P.REGS[dst]] = B.rmsnorm(g(src), self.eps, self.rsqrt)
                self.cycles += cols + 40
            elif op == "MV":
                self.reg[P.REGS[dst]] = self.matvec(addr, rows, cols, g(src))
            elif op == "ROPE":
                rw = self.read(self.rope_base + pos * 8, (self.hs // 2) // 2)
                cs = self.unpack(rw, 16).reshape(-1, 2)
                self.reg[P.REGS[dst]] = B.rope(g(src), cs[:, 0], cs[:, 1])
                self.cycles += cols // 2
            elif op == "KVW":
                sk, sv = rows & 15, rows >> 4 & 15
                self.w[self.kv_addr(aux, 0) + pos * self.kvw:][:self.kvw] = P.pack_bytes(B.kv8_q(g(src), sk))
                self.w[self.kv_addr(aux, 1) + np.arange(self.kvw) * self.seq + pos] = P.pack_bytes(B.kv8_q(g(dst), sv))
                self.cycles += self.kvw * 2 * 2
                dst = None
            elif op == "ATTN":
                sk, sv = rows & 15, rows >> 4 & 15
                K = self.unpack(self.read(self.kv_addr(aux, 0), (pos + 1) * self.kvw), 8).reshape(pos + 1, -1)
                vw = np.stack([self.read(self.kv_addr(aux, 1) + w * self.seq, pos + 1) for w in range(self.kvw)], 1)
                V = self.unpack(vw.reshape(-1), 8).reshape(pos + 1, -1)
                q = g(src)
                rep = self.heads // self.kv_heads
                att = np.zeros(self.dim, np.int64)
                for h in range(self.heads):
                    sl = slice((h // rep) * self.hs, (h // rep + 1) * self.hs)
                    att[h * self.hs:(h + 1) * self.hs] = B.attention_head(
                        q[h * self.hs:(h + 1) * self.hs], K[:, sl], V[:, sl], self.exp2, self.t_shift - sk, sv)
                self.reg[P.REGS[dst]] = B.sat16(att)
                self.cycles += 2 * (pos + 1) * self.kv_dim // 4 * 4 + (pos + 1) * 10 + self.dim * 18
            elif op == "ADD":
                self.reg[P.REGS[dst]] = B.sat16(g(dst) + g(src))
                self.cycles += cols
            elif op == "SILU":
                self.reg[P.REGS[dst]] = B.silu_mul(self.reg["H1"], self.reg["H3"], self.sigmoid, self.h_shift)
                self.cycles += cols
            elif op == "GAIN":
                gf = self.unpack(self.read(addr, cols // 4), 16)
                self.reg[P.REGS[dst]] = B.sat16(B.rshift_round(g(src) * gf, self.ug_shift))
                self.cycles += cols
            elif op == "LOGITS":
                if not logits:
                    break
                ug = g(src)
                out = np.empty(rows, np.int64)
                for r in range(rows):
                    blk = self.read(addr + r * TR, TR)
                    row = self.unpack(blk[1:], 8)
                    pr = int(blk[0])
                    cm, cs = pr >> 24 & 0xFFFF, pr >> 40 & 0xFF
                    out[r] = (int(row @ ug) * cm + (1 << (cs - 1))) >> cs
                self.reg[P.REGS[dst]] = B.sat16(out)
                self.cycles += rows * (TR / WORDS_PER_CYCLE + 2)
            elif op == "SAMPLE":
                self.cycles += 2 * rows + self.vocab
                dst = None
            if trace is not None and dst is not None:
                trace(op, P.REGS[dst], self.reg[P.REGS[dst]])
        return self.reg.get("LOGIT") if logits else None


def main():
    pkl = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    p = pickle.loads(pkl.read_bytes())
    rng = np.random.default_rng(0)
    toks = rng.integers(4, p["cfg"].vocab_size, n)
    if not p.get("kv8"):
        B.kv8_calibrate(p, [toks])
    img, prog = P.build(p)
    eng = Engine(img.bytes())
    ref = B.IntModel(p)
    same = 0
    per_tok = []
    for pos, t in enumerate(toks):
        c0, w0 = eng.cycles, eng.words_read
        a = eng.run(int(t), pos)
        b = ref.forward(int(t), pos)
        same += int(np.array_equal(a, b))
        per_tok.append((eng.cycles - c0, eng.words_read - w0))
    cyc = np.array([c for c, _ in per_tok])
    words = np.array([w for _, w in per_tok])
    print(f"bit-exact logits on {same}/{n} tokens")
    print(f"per token at pos 0: {words[0] * 8 / 1e6:.2f} MB read, {cyc[0] / 1e3:.0f}k cycles "
          f"= {cyc[0] / CLK_HZ * 1e3:.1f} ms -> {CLK_HZ / cyc[0]:.1f} tok/s at {CLK_HZ / 1e6:.0f} MHz")
    kv_extra = (p["cfg"].seq_len - 1) * (p["cfg"].n_layers * (p["cfg"].kv_dim // 2 / WORDS_PER_CYCLE
                                                                + p["cfg"].n_heads * 2 * p["cfg"].head_size / LANES))
    full = cyc[0] + kv_extra
    print(f"at pos {p['cfg'].seq_len - 1}: {full / CLK_HZ * 1e3:.1f} ms -> {CLK_HZ / full:.1f} tok/s")


if __name__ == "__main__":
    main()
