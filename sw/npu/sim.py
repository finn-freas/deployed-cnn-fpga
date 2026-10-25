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

DRAIN_DELAY = 1                    # npu_tile_engine: handover -> first shadow read (row skew only)
PERIOD_MIN = DRAIN_DELAY + N * N // 2 + 1  # min cycles between injections (2 requants)
DRAIN_DONE = DRAIN_DELAY + N * N // 2 + 4 + 2   # handover -> drain + requant pipeline empty
PERIOD_DW, DRAIN_DONE_DW = PERIOD_MIN, DRAIN_DONE
DW_K = (2, 3)                      # depthwise unit kernel sizes (SEG = 7*stride + 3 words per row)
POOL_INIT = 1 + 7                  # npu_pool: S_INIT + shift-and-add of in_w * in_cb
CONST_WAIT = 9                     # npu_tile_engine S_CONST: shift-and-add constants
GROUP_PARAM_WORDS = N * isa.PARAM_WORDS

# External store (rtl/sim/ext_mem_model.sv defaults, as instantiated by npu_sim_top)
EXT_LATENCY = 6
EXT_CYCLES_PER_WORD = 2


def ext_load_cycles(words: int) -> int:
    """Request handshake + latency + burst, until the engine sees the last word."""
    return EXT_LATENCY + 5 - EXT_CYCLES_PER_WORD + words * EXT_CYCLES_PER_WORD


def param_visible(start: int, g: int, param_words: int, uniform: bool = False) -> int:
    """Param loader (cached / depthwise), request issued at `start`: the cycle a job or unit
    of output group g first sees its params (PL_LOAD counting words; uniform: PL_DONE)."""
    if uniform:
        return start + ext_load_cycles(param_words)
    return start + ext_load_cycles(min(GROUP_PARAM_WORDS * (g + 1), param_words)) - 1


def tile_engine_cycles(*, K: int, G: int, tiles: int, gather_issue: int, gather_tail: int,
                       cached: bool, param_words: int, weight_words: int) -> int:
    """Event-level replica of npu_tile_engine for CONV (overlapped drain O3 + double-buffered
    A_T).

    Units: cached -> one per tile (all G jobs); group mode -> one per (group, tile). A
    producer gathers units into two A_T halves; a consumer streams each unit's jobs, then
    releases the half. Injections must be PERIOD_MIN cycles apart; every wait state lasts at
    least one cycle; a flag set in a state's last cycle is visible the cycle after.
    """
    t = 1 + CONST_WAIT                                        # S_INIT, S_CONST
    if cached:
        t += ext_load_cycles(weight_words)                    # whole layer's weights
    p_start = t + 1                                           # cached: param loader requests
    # units in order: (tile index, group, new_tile, needs_weight_load)
    if cached:
        units = [(ti, 0, True, False) for ti in range(tiles)]
    else:
        units = [(ti, g, True, ti == 0) for g in range(G) for ti in range(tiles)]
    jobs_per_unit = G if cached else 1

    ready = [0] * len(units)            # cycle the unit's full flag becomes visible
    release = [0] * len(units)          # cycle the unit's release becomes visible
    p_t = t + 1                         # producer: first P_WAIT_BUF cycle
    c_t = t + 1                         # consumer: first C_WAIT_FULL cycle
    last_inj = -10**9
    # group-mode loader: group g's weights then its params; the params' last L_PLOAD cycle is
    # load_end[g]. A group's load may start once the previous one is done and its buffer half
    # is free (group g-2 finished with the half-buffer prefetch, g-1 without it)
    wdbl = K <= isa.WBUF_DEPTH // 2
    load_end = {-1: t}                  # t = first S_RUN cycle - 1 ... (run_start seen at t)
    group_done = {}                     # cycle the consumer's "group finished" is visible

    def load_visible(g):
        if g not in load_end:
            load_visible(g - 1)
            free_g = g - (2 if wdbl else 1)
            permit = group_done[free_g] if free_g >= 0 else 0
            start = max(load_end[g - 1] + 1, permit) + 1                # L_IDLE -> L_REQ
            w_end = start + ext_load_cycles(K) - 1
            load_end[g] = w_end + ext_load_cycles(GROUP_PARAM_WORDS)       # L_PREQ, L_PLOAD
        return load_end[g] + 1

    for u, (ti, g, new_tile, wload) in enumerate(units):
        # ---- producer gathers unit u into half u % 2
        w = max(p_t, release[u - 2] if u >= 2 else 0)        # P_WAIT_BUF passes
        last = w + (TILE_SETUP if new_tile else 0) + gather_issue   # last read issued
        ready[u] = last + gather_tail                         # published as the transposer finishes
        p_t = last + 1                                        # next unit starts right away
        # ---- consumer streams unit u (group mode: once its group's weights have landed)
        s = max(c_t, ready[u], load_visible(g) if not cached else 0) + 1   # C_WAIT_FULL passes
        for j in range(jobs_per_unit):
            p_ok = param_visible(p_start, j, param_words) if cached else 0
            inj = max(s, last_inj + PERIOD_MIN, p_ok) + 1     # C_INJ_WAIT passes; C_STREAM
            last_inj = inj
            s = inj + K + 1                                   # stream K, C_NEXT
        release[u] = s                                        # released in the last C_NEXT
        if not cached and ti == tiles - 1:
            group_done[g] = s
        c_t = s
    hand = last_inj + K - 1                                   # last job's last value: drain takes it
    t = hand + DRAIN_DONE + 1                                 # (C_DONE, S_FINISH come sooner)
    return t + INSTR_OVERHEAD - 1


def dw_engine_cycles(*, ksize: int, stride: int, units: int, blocks: int, param_words: int,
                     weight_words: int, uniform: bool) -> int:
    """npu_tile_engine in depthwise mode: a unit (N pixels of an output row x one channel
    block) reads ksize kernel rows of SEG words, one per cycle, each tap issued with the read
    of its word; its first read is its first tap (the injection), and injections are at least
    PERIOD_DW apart. The drain takes the last unit with its last tap: word ksize-1 (stride 1)
    or word 9 (stride 2, where the odd columns follow the even ones) of its last kernel row.
    (The previous instruction's last injection is always long past, so the first unit starts
    at once.) Units run channel block by channel block; a block's first unit also waits for
    its params, which load after the weights."""
    seg = 7 * stride + 3
    t = 1 + CONST_WAIT + ext_load_cycles(weight_words)   # run_start seen
    period = max(ksize * seg, PERIOD_DW)
    per_block = units // blocks
    t0_last = t + 1 - period
    for b in range(blocks):
        t0_last = max(t0_last + period, param_visible(t + 1, b, param_words, uniform))
        t0_last += (per_block - 1) * period           # the block's last unit's first read
    last_word = ksize - 1 if stride == 1 else 9
    hand = t0_last + (ksize - 1) * seg + last_word    # its last tap
    t = hand + DRAIN_DONE_DW + 1
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
            # Packed input at byte in_base*8 of activation memory: k = ky*KR + kx*C + c (each
            # kernel row padded to KR with zeros); uint8 -> int8 by adding in_zp (padding taps
            # read as u8 0, i.e. int8 in_zp).
            iy, ix, valid = self._window(ins)
            C, ks, KR = ins.in_cb, ins.ksize, ins.packed_row_k
            addr = ins.in_base * N + ((iy * ins.in_w + ix) * C)[..., None] + np.arange(C)   # bytes
            v = np.broadcast_to(valid[..., None], addr.shape)
            self._check(addr[v] // N, isa.ACT_DEPTH, "CONV packed input")
            act_bytes = self.act.view(np.uint8).reshape(-1)
            u8 = np.where(v, act_bytes[np.where(v, addr, 0)], 0).astype(np.int64)   # [P, ks, ks, C]
            P0 = u8.shape[0]
            A = np.zeros((P0, ks, KR), np.int64)
            A[:, :, :ks * C] = (((u8 + ins.in_zp + 128) & 0xFF) - 128).reshape(P0, ks, ks * C)
            A = A.reshape(P0, K)
            real_k = ks * ks * C
        else:
            words, _ = self._gather_words(ins, ins.in_zp)
            A = words.reshape(words.shape[0], -1).astype(np.int64)                  # [P, K]
            real_k = K
        P = A.shape[0]
        Wm = self._weights(ins, G * K).reshape(G, K, N)
        acc = np.einsum("pk,gkc->pgc", A, Wm)                                         # [P, G, N]
        self._write_out(ins, self._requant(ins, acc, G))

        tiles = -(-P // N)
        if ins.in_src:   # packed (O2): per (kernel row, row) its first window reads two aligned words, the rest one
            issue, tail = K + N * ins.ksize, GATHER_TAIL_WORD
        else:            # one word per cycle through the transposer
            issue, tail = N * (K // N), GATHER_TAIL_WORD
        self.stats.cycles.append(tile_engine_cycles(
            K=K, G=G, tiles=tiles, gather_issue=issue, gather_tail=tail,
            cached=G * K <= isa.WBUF_DEPTH, param_words=G * N * isa.PARAM_WORDS, weight_words=G * K))
        self.stats.macs.append(P * real_k * G * N)

    def _dwconv(self, ins: Instr):
        cb, T = ins.in_cb, ins.K
        words, _ = self._gather_words(ins, ins.in_zp)                                 # [P, k, k, cb, N]
        P = words.shape[0]
        A = words.reshape(P, T, cb, N).astype(np.int64)
        nb = 1 if ins.uniform else cb          # channel blocks of weights / params stored
        Wm = np.broadcast_to(self._weights(ins, nb * T).reshape(nb, T, N), (cb, T, N))
        acc = np.einsum("ptgc,gtc->pgc", A, Wm)                                       # [P, cb, N]
        if ins.uniform:
            bias, mult, shift = (np.full((cb, N), p[0]) for p in self._params(ins, 1))
            y = requantize(acc, bias, mult, shift, ins.out_zp, ins.act_min, ins.act_max, ins.rmode)
        else:
            y = self._requant(ins, acc, cb)
        self._write_out(ins, y)
        # depthwise datapath; the weights always fit the weight buffer (cached)
        assert ins.ksize in DW_K and ins.stride in (1, 2) and cb * T <= isa.WBUF_DEPTH
        self.stats.cycles.append(dw_engine_cycles(
            ksize=ins.ksize, stride=ins.stride, units=cb * ins.out_h * -(-ins.out_w // N),
            blocks=cb, param_words=(1 if ins.uniform else cb * N) * isa.PARAM_WORDS,
            weight_words=nb * T, uniform=bool(ins.uniform)))
        self.stats.macs.append(P * T * cb * N)

    def _maxpool(self, ins: Instr):
        words, v = self._gather_words(ins, -128)                                      # [P, k, k, cb, N]
        m = np.where(v[..., None], words, np.int8(-128)).max(axis=(1, 2))
        m = np.clip(m, ins.act_min, ins.act_max).astype(np.int8)
        self._write_out(ins, m)
        # taps + tail + write (and advance), per pixel per block
        self.stats.cycles.append(m.shape[0] * ins.in_cb * (ins.ksize ** 2 + 2)
                                 + POOL_INIT - 1 + INSTR_OVERHEAD)
        self.stats.macs.append(0)

    def _avgpool(self, ins: Instr):
        raise NotImplementedError("AVGPOOL is lowered to DWCONV by the compiler")

    def _argmax(self, ins: Instr):
        count = ins.in_w
        idx = np.arange(count)
        v = self.act[ins.in_base + idx // N, idx % N].astype(np.int64)
        cls = int(np.argmax(v))          # first maximum, like TFLite ARG_MAX
        self.result = (cls, int(v[cls]))
        self.stats.cycles.append(count + 3)
        self.stats.macs.append(0)
