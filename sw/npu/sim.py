"""Instruction-level NPU simulator with a cycle-exact model of the RTL.

Executes the compiled memory images with the same addressing, padding and integer
math as the RTL. It is the golden model for RTL tests, and is itself checked
bit-exactly against TFLite's reference kernels.

Stats.cycles reproduces the v1 RTL's per-instruction cycle counts exactly (the
state-machine durations of npu_tile_engine / npu_pool / npu_core); tb/test_npu_core.py
asserts the match, so micro-architecture changes can be evaluated here first.
"""

from dataclasses import dataclass, field

import numpy as np

from . import isa
from .isa import N, Instr, Op
from .quant import requantize

# ---- cycle model of the RTL (state durations in npu_tile_engine / npu_pool / npu_core)
INSTR_OVERHEAD = 5                 # unit start/init/done handshake as seen by instr_cycles
SEQ_PER_INSTR = 2 + isa.INSTR_W // 32   # fetch beats + next, counted in the run total only
TILE_SETUP = N                     # per-row pixel coordinates
GATHER_TAIL_WORD = 2 + N           # last word lands, transposer writes its N columns
GATHER_TAIL_PACKED = 2             # last byte lands
DRAIN_DELAY = 3                    # npu_tile_engine: capture -> first shadow read
PERIOD_MIN = DRAIN_DELAY + N * N + 1   # min cycles between injections
DRAIN_DONE = DRAIN_DELAY + N * N + 4 + 2   # capture -> drain + requant pipeline empty

# External store (rtl/sim/ext_mem_model.sv defaults, as instantiated by npu_sim_top)
EXT_LATENCY = 6
EXT_CYCLES_PER_WORD = 2


def ext_load_cycles(words: int) -> int:
    """Request handshake + latency + burst, until the engine sees the last word."""
    return EXT_LATENCY + 5 - EXT_CYCLES_PER_WORD + words * EXT_CYCLES_PER_WORD


def tile_engine_cycles(*, K: int, G: int, tiles: int, gather_issue: int, gather_tail: int,
                       dw: bool, cached: bool, param_words: int, weight_words: int) -> int:
    """Event-level replica of npu_tile_engine (overlapped drain O3 + double-buffered A_T).

    Units: cached CONV -> one per tile (all G jobs); DWCONV -> one per (tile, channel
    block); group mode -> one per (group, tile). A producer gathers units into two A_T
    halves; a consumer streams each unit's jobs, then releases the half. Injections must be
    PERIOD_MIN cycles apart; every wait state lasts at least one cycle; a flag set in a
    state's last cycle is visible the cycle after.
    """
    t = 1 + ext_load_cycles(param_words)                      # S_INIT, params
    if cached:
        t += ext_load_cycles(weight_words)                    # whole layer's weights
    # units in order: (tile index, group, new_tile, needs_weight_load)
    units = []
    if dw:
        units = [(ti, g, g == 0, False) for ti in range(tiles) for g in range(G)]
    elif cached:
        units = [(ti, 0, True, False) for ti in range(tiles)]
    else:
        units = [(ti, g, True, ti == 0) for g in range(G) for ti in range(tiles)]
    jobs_per_unit = G if (cached and not dw) else 1

    ready = [0] * len(units)            # cycle the unit's full flag becomes visible
    release = [0] * len(units)          # cycle the unit's release becomes visible
    p_t = t + 1                         # producer: first P_WAIT_BUF cycle
    c_t = t + 1                         # consumer: first C_WAIT_FULL cycle
    last_inj = -10**9
    # group-mode weight loader: group g's load ends (last L_LOAD cycle) at load_end[g]; it may
    # start once the previous load is done and its buffer half is free (group g-2 finished
    # with the half-buffer prefetch, g-1 without it)
    wdbl = K <= isa.WBUF_DEPTH // 2
    load_end = {-1: t}                  # t = first S_RUN cycle - 1 ... (run_start seen at t)
    group_done = {}                     # cycle the consumer's "group finished" is visible

    def load_visible(g):
        if g not in load_end:
            load_visible(g - 1)
            free_g = g - (2 if wdbl else 1)
            permit = group_done[free_g] if free_g >= 0 else 0
            start = max(load_end[g - 1] + 1, permit) + 1                # L_IDLE -> L_REQ
            load_end[g] = start + ext_load_cycles(K) - 1
        return load_end[g] + 1

    for u, (ti, g, new_tile, wload) in enumerate(units):
        # ---- producer gathers unit u into half u % 2
        w = max(p_t, release[u - 2] if u >= 2 else 0)        # P_WAIT_BUF passes
        e = w + (TILE_SETUP if new_tile else 0) + gather_issue + gather_tail   # last P_TAIL cycle
        ready[u] = e + 1
        p_t = e + 1
        # ---- consumer streams unit u (group mode: once its group's weights have landed)
        s = max(c_t, ready[u], load_visible(g) if not cached else 0) + 1   # C_WAIT_FULL passes
        for _ in range(jobs_per_unit):
            inj = max(s, last_inj + PERIOD_MIN) + 1           # C_INJ_WAIT passes; C_STREAM
            last_inj = inj
            s = inj + K + 1                                   # stream K, C_NEXT
        release[u] = s                                        # released in the last C_NEXT
        if not cached and ti == tiles - 1:
            group_done[g] = s
        c_t = s
    cap = max(c_t, last_inj + PERIOD_MIN) + 1                 # C_CAP_WAIT; C_CAPTURE
    t = cap + 2                                               # C_DONE seen; S_FINISH
    t = max(t, cap + DRAIN_DONE) + 1
    return t + INSTR_OVERHEAD - 1


def total_cycles(stats: "Stats", n_instrs: int) -> int:
    """Cycles from start to done for a whole program run (matches npu_core.cycles)."""
    return sum(stats.cycles) + SEQ_PER_INSTR * n_instrs


@dataclass
class Stats:
    cycles: list[int] = field(default_factory=list)   # per executed instruction
    macs: list[int] = field(default_factory=list)     # MACs the model needs (no padding)


def _out_grid(ins: Instr):
    oy, ox = np.meshgrid(np.arange(ins.out_h), np.arange(ins.out_w), indexing="ij")
    return oy.reshape(-1, 1, 1), ox.reshape(-1, 1, 1)


class NpuSim:
    def __init__(self, instrs: list[Instr], ext: np.ndarray, input_base: int = 0):
        self.imem = list(instrs)
        self.ext = np.asarray(ext, np.int8).reshape(-1, N)        # external store image
        self.ext_u64 = self.ext.view("<u8").reshape(-1)
        self.input_base = input_base
        self.act = np.zeros((isa.ACT_DEPTH, N), np.int8)
        self.result = (0, 0)
        self.stats = Stats()

    @classmethod
    def from_program(cls, prog):
        return cls(prog.instrs, prog.ext, prog.meta["input"]["base"])

    # ------------------------------------------------------------ helpers
    def read_tensor(self, base, h, w, cb, c):
        words = self.act[base:base + h * w * cb]
        return words.reshape(h, w, cb * N)[:, :, :c]

    def load_input(self, image_u8: np.ndarray):
        """Packed uint8 HWC bytes into activation memory at input_base (what the host does)."""
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
        """Input coordinates of each output pixel's k x k window: iy, ix [P, k, k], valid mask."""
        oy, ox = _out_grid(ins)
        k = ins.ksize
        iy = oy * ins.stride - ins.pad + np.arange(k).reshape(1, k, 1)
        ix = ox * ins.stride - ins.pad + np.arange(k).reshape(1, 1, k)
        valid = (iy >= 0) & (iy < ins.in_h) & (ix >= 0) & (ix < ins.in_w)
        return iy, ix, valid

    def _gather_words(self, ins: Instr, fill):
        """Channel-blocked window gather: [P, k, k, cb, N] int8, out-of-bounds = fill."""
        iy, ix, valid = self._window(ins)
        cb = ins.in_cb
        addr = ins.in_base + ((iy * ins.in_w + ix) * cb)[..., None] + np.arange(cb)
        v = np.broadcast_to(valid[..., None], addr.shape)
        self._check(addr[v], isa.ACT_DEPTH, f"{ins.op.name} src")
        return np.where(v[..., None], self.act[np.where(v, addr, 0)], np.int8(fill)), v

    def _write_out(self, ins: Instr, y: np.ndarray):
        """y: [P, out_g, N] int8 -> raster order, out_g words per pixel."""
        P, G = y.shape[0], y.shape[1]
        dst = ins.out_base + np.arange(P * G)
        self._check(dst, isa.ACT_DEPTH, f"{ins.op.name} dst")
        self.act[dst] = y.reshape(P * G, N)

    def _params(self, ins: Instr, n):
        """(bias, mult, shift) for n channels from the external store (2 words per entry)."""
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

    # ------------------------------------------------------------ execution
    def run(self, image_u8: np.ndarray, max_instrs: int = 1000, trace=None):
        """Run the program on one image. trace(pc, ins) is called after each instruction."""
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
        self.stats.cycles.append(1)   # not measured on RTL yet
        self.stats.macs.append(0)

    def _conv(self, ins: Instr):
        K, G = ins.K, ins.out_g
        if ins.in_src:
            # Packed input at byte in_base*8 of activation memory: k = (ky*ksize + kx)*C + c;
            # uint8 -> int8 by adding in_zp (padding reads as u8 0, i.e. int8 in_zp).
            iy, ix, valid = self._window(ins)
            C = ins.in_cb
            addr = ins.in_base * N + ((iy * ins.in_w + ix) * C)[..., None] + np.arange(C)   # bytes
            v = np.broadcast_to(valid[..., None], addr.shape)
            self._check(addr[v] // N, isa.ACT_DEPTH, "CONV packed input")
            act_bytes = self.act.view(np.uint8).reshape(-1)
            u8 = np.where(v, act_bytes[np.where(v, addr, 0)], 0).astype(np.int64)
            A = np.zeros((u8.shape[0], K), np.int64)
            A[:, :u8[0].size] = (((u8 + ins.in_zp + 128) & 0xFF) - 128).reshape(u8.shape[0], -1)
            real_k = ins.ksize * ins.ksize * C
        else:
            words, _ = self._gather_words(ins, ins.in_zp)
            A = words.reshape(words.shape[0], -1).astype(np.int64)                  # [P, K]
            real_k = K
        P = A.shape[0]
        Wm = self._weights(ins, G * K).reshape(G, K, N)
        acc = np.einsum("pk,gkc->pgc", A, Wm)                                         # [P, G, N]
        self._write_out(ins, self._requant(ins, acc, G))

        tiles = -(-P // N)
        if ins.in_src:   # packed: one byte per cycle
            issue, tail = N * K, GATHER_TAIL_PACKED
        else:            # one word per cycle through the transposer
            issue, tail = N * (K // N), GATHER_TAIL_WORD
        self.stats.cycles.append(tile_engine_cycles(
            K=K, G=G, tiles=tiles, gather_issue=issue, gather_tail=tail, dw=False,
            cached=G * K <= isa.WBUF_DEPTH, param_words=G * N * isa.PARAM_WORDS, weight_words=G * K))
        self.stats.macs.append(P * real_k * G * N)

    def _dwconv(self, ins: Instr):
        cb, T = ins.in_cb, ins.K
        words, _ = self._gather_words(ins, ins.in_zp)                                 # [P, k, k, cb, N]
        P = words.shape[0]
        A = words.reshape(P, T, cb, N).astype(np.int64)
        Wm = self._weights(ins, cb * T).reshape(cb, T, N)
        acc = np.einsum("ptgc,gtc->pgc", A, Wm)                                       # [P, cb, N]
        self._write_out(ins, self._requant(ins, acc, cb))
        # CONV datapath with block-diagonal weights (K = T*N), A re-gathered per channel block;
        # depthwise weights always fit the weight buffer (cached).
        tiles = -(-P // N)
        assert cb * T <= isa.WBUF_DEPTH
        self.stats.cycles.append(tile_engine_cycles(
            K=T * N, G=cb, tiles=tiles, gather_issue=N * T, gather_tail=GATHER_TAIL_WORD,
            dw=True, cached=True,
            param_words=cb * N * isa.PARAM_WORDS, weight_words=cb * T))
        self.stats.macs.append(P * T * cb * N)

    def _maxpool(self, ins: Instr):
        words, v = self._gather_words(ins, -128)                                      # [P, k, k, cb, N]
        m = np.where(v[..., None], words, np.int8(-128)).max(axis=(1, 2))
        m = np.clip(m, ins.act_min, ins.act_max).astype(np.int8)
        self._write_out(ins, m)
        self.stats.cycles.append(self._pool_cycles(ins, m.shape[0], divide=False))
        self.stats.macs.append(0)

    @staticmethod
    def _pool_cycles(ins: Instr, P: int, divide: bool) -> int:
        # taps + tail + (divider: init + 16 steps | 1) + write + next, per pixel per block
        return P * ins.in_cb * (ins.ksize ** 2 + 1 + (17 if divide else 1) + 1 + 1) + INSTR_OVERHEAD

    def _avgpool(self, ins: Instr):
        # TFLite reference int8 AveragePool: sum over in-bounds taps, divide by their count
        # rounding half away from zero (C truncating division), then clamp.
        words, v = self._gather_words(ins, 0)
        s = np.where(v[..., None], words.astype(np.int64), 0).sum(axis=(1, 2))        # [P, cb, N]
        cnt = v[..., 0].sum(axis=(1, 2)).reshape(-1, 1, 1).astype(np.int64)
        half = cnt // 2
        q = np.where(s > 0, (s + half) // cnt, -((-(s - half)) // cnt))
        q = np.clip(q, ins.act_min, ins.act_max).astype(np.int8)
        self._write_out(ins, q)
        self.stats.cycles.append(self._pool_cycles(ins, q.shape[0], divide=True))
        self.stats.macs.append(0)

    def _argmax(self, ins: Instr):
        count = ins.in_w
        idx = np.arange(count)
        v = self.act[ins.in_base + idx // N, idx % N].astype(np.int64)
        cls = int(np.argmax(v))          # first maximum, like TFLite ARG_MAX
        self.result = (cls, int(v[cls]))
        self.stats.cycles.append(count + 3)
        self.stats.macs.append(0)
