from dataclasses import dataclass, field

import numpy as np

from . import isa
from .isa import N, Instr, Op
from .quant import requantize

INSTR_OVERHEAD = 5
SEQ_PER_INSTR = 2 + isa.INSTR_W // 32
TILE_SETUP = N
GATHER_TAIL_WORD = 2 + N

DRAIN_DELAY = 1
PERIOD_MIN = DRAIN_DELAY + N * N // 2 + 1
DRAIN_DONE = DRAIN_DELAY + N * N // 2 + 4 + 2
PERIOD_DW, DRAIN_DONE_DW = PERIOD_MIN, DRAIN_DONE
DW_K = (2, 3)
POOL_INIT = 1 + 7
CONST_WAIT = 9
GROUP_PARAM_WORDS = N * isa.PARAM_WORDS

EXT_LATENCY = 6
EXT_CYCLES_PER_WORD = 2

def ext_load_cycles(words: int) -> int:
    return EXT_LATENCY + 5 - EXT_CYCLES_PER_WORD + words * EXT_CYCLES_PER_WORD

def param_visible(start: int, g: int, param_words: int, uniform: bool = False) -> int:
    if uniform:
        return start + ext_load_cycles(param_words)
    return start + ext_load_cycles(min(GROUP_PARAM_WORDS * (g + 1), param_words)) - 1

def tile_engine_cycles(*, K: int, G: int, tiles: int, gather_issue: int, gather_tail: int,
                       cached: bool, param_words: int, weight_words: int) -> int:
    t = 1 + CONST_WAIT
    if cached:
        t += ext_load_cycles(weight_words)
    p_start = t + 1
    if cached:
        units = [(ti, 0, True, False) for ti in range(tiles)]
    else:
        units = [(ti, g, True, ti == 0) for g in range(G) for ti in range(tiles)]
    jobs_per_unit = G if cached else 1

    ready = [0] * len(units)
    release = [0] * len(units)
    p_t = t + 1
    c_t = t + 1
    last_inj = -10**9
    wdbl = K <= isa.WBUF_DEPTH // 2
    load_end = {-1: t}
    group_done = {}

    def load_visible(g):
        if g not in load_end:
            load_visible(g - 1)
            free_g = g - (2 if wdbl else 1)
            permit = group_done[free_g] if free_g >= 0 else 0
            start = max(load_end[g - 1] + 1, permit) + 1
            w_end = start + ext_load_cycles(K) - 1
            load_end[g] = w_end + ext_load_cycles(GROUP_PARAM_WORDS)
        return load_end[g] + 1

    for u, (ti, g, new_tile, wload) in enumerate(units):
        w = max(p_t, release[u - 2] if u >= 2 else 0)
        last = w + (TILE_SETUP if new_tile else 0) + gather_issue
        ready[u] = last + gather_tail
        p_t = last + 1
        s = max(c_t, ready[u], load_visible(g) if not cached else 0) + 1
        for j in range(jobs_per_unit):
            p_ok = param_visible(p_start, j, param_words) if cached else 0
            inj = max(s, last_inj + PERIOD_MIN, p_ok) + 1
            last_inj = inj
            s = inj + K + 1
        release[u] = s
        if not cached and ti == tiles - 1:
            group_done[g] = s
        c_t = s
    hand = last_inj + K - 1
    t = hand + DRAIN_DONE + 1
    return t + INSTR_OVERHEAD - 1

def dw_engine_cycles(*, ksize: int, stride: int, units: int, blocks: int, param_words: int,
                     weight_words: int, uniform: bool) -> int:
    seg = 7 * stride + 3
    t = 1 + CONST_WAIT + ext_load_cycles(weight_words)
    period = max(ksize * seg, PERIOD_DW)
    per_block = units // blocks
    t0_last = t + 1 - period
    for b in range(blocks):
        t0_last = max(t0_last + period, param_visible(t + 1, b, param_words, uniform))
        t0_last += (per_block - 1) * period
    last_word = ksize - 1 if stride == 1 else 9
    hand = t0_last + (ksize - 1) * seg + last_word
    t = hand + DRAIN_DONE_DW + 1
    return t + INSTR_OVERHEAD - 1

def total_cycles(stats: "Stats", n_instrs: int) -> int:
    return sum(stats.cycles) + SEQ_PER_INSTR * n_instrs

@dataclass
class Stats:
    cycles: list[int] = field(default_factory=list)
    macs: list[int] = field(default_factory=list)

def _out_grid(ins: Instr):
    oy, ox = np.meshgrid(np.arange(ins.out_h), np.arange(ins.out_w), indexing="ij")
    return oy.reshape(-1, 1, 1), ox.reshape(-1, 1, 1)

class NpuSim:
    def __init__(self, instrs: list[Instr], ext: np.ndarray, input_base: int = 0):
        self.imem = list(instrs)
        self.ext = np.asarray(ext, np.int8).reshape(-1, N)
        self.ext_u64 = self.ext.view("<u8").reshape(-1)
        self.input_base = input_base
        self.act = np.zeros((isa.ACT_DEPTH, N), np.int8)
        self.result = (0, 0)
        self.stats = Stats()

    @classmethod
    def from_program(cls, prog):
        return cls(prog.instrs, prog.ext, prog.meta["input"]["base"])

    def read_tensor(self, base, h, w, cb, c):
        words = self.act[base:base + h * w * cb]
        return words.reshape(h, w, cb * N)[:, :, :c]

    def load_input(self, image_u8: np.ndarray):
        flat = np.asarray(image_u8, np.uint8).reshape(-1)
        words = -(-flat.size // N)
        buf = np.zeros(words * N, np.uint8)
        buf[:flat.size] = flat
        self.act[self.input_base:self.input_base + words] = buf.view(np.int8).reshape(words, N)

    @staticmethod
    def _check(addr, depth, what):
        if addr.size and np.any((addr < 0) | (addr >= depth)):
            raise IndexError(f"{what} address out of range: {addr.min()}..{addr.max()} (depth {depth})")

    def _window(self, ins: Instr):
        oy, ox = _out_grid(ins)
        k = ins.ksize
        iy = oy * ins.stride - ins.pad + np.arange(k).reshape(1, k, 1)
        ix = ox * ins.stride - ins.pad + np.arange(k).reshape(1, 1, k)
        valid = (iy >= 0) & (iy < ins.in_h) & (ix >= 0) & (ix < ins.in_w)
        return iy, ix, valid

    def _gather_words(self, ins: Instr, fill):
        iy, ix, valid = self._window(ins)
        cb = ins.in_cb
        addr = ins.in_base + ((iy * ins.in_w + ix) * cb)[..., None] + np.arange(cb)
        v = np.broadcast_to(valid[..., None], addr.shape)
        self._check(addr[v], isa.ACT_DEPTH, f"{ins.op.name} src")
        return np.where(v[..., None], self.act[np.where(v, addr, 0)], np.int8(fill)), v

    def _write_out(self, ins: Instr, y: np.ndarray):
        P, G = y.shape[0], y.shape[1]
        dst = ins.out_base + np.arange(P * G)
        self._check(dst, isa.ACT_DEPTH, f"{ins.op.name} dst")
        self.act[dst] = y.reshape(P * G, N)

    def _params(self, ins: Instr, n):
        addr = ins.p_base + isa.PARAM_WORDS * np.arange(n)
        self._check(addr + 1, isa.EXT_DEPTH, "param")
        w0, w1 = self.ext_u64[addr], self.ext_u64[addr + 1]
        bias = (w0 & 0xFFFFFFFF).astype(np.uint32).view(np.int32).astype(np.int64)
        mult = (w0 >> 32).astype(np.uint32).view(np.int32).astype(np.int64)
        shift = w1.view(np.int64)
        return bias, mult, shift

    def _requant(self, ins: Instr, acc, G):
        bias, mult, shift = (p.reshape(G, N) for p in self._params(ins, G * N))
        return requantize(acc, bias, mult, shift, ins.out_zp, ins.act_min, ins.act_max, ins.rmode)

    def _weights(self, ins: Instr, n):
        addr = ins.w_base + np.arange(n)
        self._check(addr, isa.EXT_DEPTH, "weight")
        return self.ext[addr].astype(np.int64)

    def run(self, image_u8: np.ndarray, max_instrs: int = 1000, trace=None):
        self.load_input(image_u8)
        self.stats = Stats()
        for pc in range(max_instrs):
            ins = self.imem[pc]
            if ins.op == Op.END:
                return self.result
            getattr(self, "_" + ins.op.name.lower())(ins)
            if trace:
                trace(pc, ins)
        raise RuntimeError("program did not reach END")

    def _nop(self, ins):
        self.stats.cycles.append(1)
        self.stats.macs.append(0)

    def _conv(self, ins: Instr):
        K, G = ins.K, ins.out_g
        if ins.in_src:
            iy, ix, valid = self._window(ins)
            C, ks, KR = ins.in_cb, ins.ksize, ins.packed_row_k
            addr = ins.in_base * N + ((iy * ins.in_w + ix) * C)[..., None] + np.arange(C)
            v = np.broadcast_to(valid[..., None], addr.shape)
            self._check(addr[v] // N, isa.ACT_DEPTH, "CONV packed input")
            act_bytes = self.act.view(np.uint8).reshape(-1)
            u8 = np.where(v, act_bytes[np.where(v, addr, 0)], 0).astype(np.int64)
            P0 = u8.shape[0]
            A = np.zeros((P0, ks, KR), np.int64)
            A[:, :, :ks * C] = (((u8 + ins.in_zp + 128) & 0xFF) - 128).reshape(P0, ks, ks * C)
            A = A.reshape(P0, K)
            real_k = ks * ks * C
        else:
            words, _ = self._gather_words(ins, ins.in_zp)
            A = words.reshape(words.shape[0], -1).astype(np.int64)
            real_k = K
        P = A.shape[0]
        Wm = self._weights(ins, G * K).reshape(G, K, N)
        acc = np.einsum("pk,gkc->pgc", A, Wm)
        self._write_out(ins, self._requant(ins, acc, G))

        tiles = -(-P // N)
        if ins.in_src:
            issue, tail = K + N * ins.ksize, GATHER_TAIL_WORD
        else:
            issue, tail = N * (K // N), GATHER_TAIL_WORD
        self.stats.cycles.append(tile_engine_cycles(
            K=K, G=G, tiles=tiles, gather_issue=issue, gather_tail=tail,
            cached=G * K <= isa.WBUF_DEPTH, param_words=G * N * isa.PARAM_WORDS, weight_words=G * K))
        self.stats.macs.append(P * real_k * G * N)

    def _dwconv(self, ins: Instr):
        cb, T = ins.in_cb, ins.K
        words, _ = self._gather_words(ins, ins.in_zp)
        P = words.shape[0]
        A = words.reshape(P, T, cb, N).astype(np.int64)
        nb = 1 if ins.uniform else cb
        Wm = np.broadcast_to(self._weights(ins, nb * T).reshape(nb, T, N), (cb, T, N))
        acc = np.einsum("ptgc,gtc->pgc", A, Wm)
        if ins.uniform:
            bias, mult, shift = (np.full((cb, N), p[0]) for p in self._params(ins, 1))
            y = requantize(acc, bias, mult, shift, ins.out_zp, ins.act_min, ins.act_max, ins.rmode)
        else:
            y = self._requant(ins, acc, cb)
        self._write_out(ins, y)
        assert ins.ksize in DW_K and ins.stride in (1, 2) and cb * T <= isa.WBUF_DEPTH
        self.stats.cycles.append(dw_engine_cycles(
            ksize=ins.ksize, stride=ins.stride, units=cb * ins.out_h * -(-ins.out_w // N),
            blocks=cb, param_words=(1 if ins.uniform else cb * N) * isa.PARAM_WORDS,
            weight_words=nb * T, uniform=bool(ins.uniform)))
        self.stats.macs.append(P * T * cb * N)

    def _maxpool(self, ins: Instr):
        words, v = self._gather_words(ins, -128)
        m = np.where(v[..., None], words, np.int8(-128)).max(axis=(1, 2))
        m = np.clip(m, ins.act_min, ins.act_max).astype(np.int8)
        self._write_out(ins, m)
        self.stats.cycles.append(m.shape[0] * ins.in_cb * (ins.ksize ** 2 + 2)
                                 + POOL_INIT - 1 + INSTR_OVERHEAD)
        self.stats.macs.append(0)

    def _avgpool(self, ins: Instr):
        raise NotImplementedError("AVGPOOL is lowered to DWCONV by the compiler")

    def _argmax(self, ins: Instr):
        count = ins.in_w
        idx = np.arange(count)
        v = self.act[ins.in_base + idx // N, idx % N].astype(np.int64)
        cls = int(np.argmax(v))
        self.result = (cls, int(v[cls]))
        self.stats.cycles.append(count + 3)
        self.stats.macs.append(0)
