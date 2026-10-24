"""Instruction-level NPU simulator.

Executes the compiled memory images with the same addressing, padding and integer
math as the RTL. It is the golden model for RTL tests, and is itself checked
bit-exactly against TFLite's reference kernels.
"""

from dataclasses import dataclass, field

import numpy as np

from . import isa
from .isa import N, Instr, Op
from .quant import requantize

# Cycle model for the v1 (fully sequential) micro-architecture; see docs/isa.md.
FLUSH_CYCLES = 1 + (N - 1) + 2 * (N - 1) + 1   # bank read + skew + array diagonal + PE register
DRAIN_CYCLES = N * N + 4                       # one result per cycle + requant pipeline
TRANSPOSE_LAT = N


@dataclass
class Stats:
    cycles: list[int] = field(default_factory=list)   # per executed instruction
    macs: list[int] = field(default_factory=list)     # MACs the model needs (no padding)


def _out_grid(ins: Instr):
    oy, ox = np.meshgrid(np.arange(ins.out_h), np.arange(ins.out_w), indexing="ij")
    return oy.reshape(-1, 1, 1), ox.reshape(-1, 1, 1)


class NpuSim:
    def __init__(self, instrs: list[Instr], wmem: np.ndarray, pmem: list[tuple[int, int, int]]):
        self.imem = list(instrs)
        self.wmem = np.asarray(wmem, np.int8).reshape(-1, N)
        p = np.zeros((isa.P_DEPTH, 3), np.int64)
        p[:len(pmem)] = pmem
        self.p_bias, self.p_mult, self.p_shift = p[:, 0], p[:, 1], p[:, 2]
        self.act = np.zeros((isa.ACT_DEPTH, N), np.int8)
        self.input_buffer = np.zeros(isa.IN_DEPTH, np.uint8)
        self.result = (0, 0)
        self.stats = Stats()

    @classmethod
    def from_program(cls, prog):
        return cls(prog.instrs, prog.wmem, prog.pmem)

    # ------------------------------------------------------------ helpers
    def read_tensor(self, base, h, w, cb, c):
        words = self.act[base:base + h * w * cb]
        return words.reshape(h, w, cb * N)[:, :, :c]

    def load_input(self, image_u8: np.ndarray):
        flat = np.asarray(image_u8, np.uint8).reshape(-1)
        self.input_buffer[:] = 0
        self.input_buffer[:flat.size] = flat

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

    def _requant(self, ins: Instr, acc, G):
        p_addr = ins.p_base + np.arange(G * N)
        self._check(p_addr, isa.P_DEPTH, "param")
        sh = (G, N)
        return requantize(acc, self.p_bias[p_addr].reshape(sh), self.p_mult[p_addr].reshape(sh),
                          self.p_shift[p_addr].reshape(sh), ins.out_zp, ins.act_min, ins.act_max,
                          ins.rmode)

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
        self.stats.cycles.append(1)
        self.stats.macs.append(0)

    def _conv(self, ins: Instr):
        K, G = ins.K, ins.out_g
        if ins.in_src:
            # Packed input buffer: k = (ky*ksize + kx)*C + c; uint8 -> int8 by adding in_zp
            # (padding reads as u8 0, i.e. int8 in_zp).
            iy, ix, valid = self._window(ins)
            C = ins.in_cb
            addr = ((iy * ins.in_w + ix) * C)[..., None] + np.arange(C)               # [P, k, k, C]
            v = np.broadcast_to(valid[..., None], addr.shape)
            self._check(addr[v], isa.IN_DEPTH, "CONV input buffer")
            u8 = np.where(v, self.input_buffer[np.where(v, addr, 0)], 0).astype(np.int64)
            A = np.zeros((u8.shape[0], K), np.int64)
            A[:, :u8[0].size] = (((u8 + ins.in_zp + 128) & 0xFF) - 128).reshape(u8.shape[0], -1)
            real_k = ins.ksize * ins.ksize * C
        else:
            words, _ = self._gather_words(ins, ins.in_zp)
            A = words.reshape(words.shape[0], -1).astype(np.int64)                  # [P, K]
            real_k = K
        P = A.shape[0]
        w_addr = ins.w_base + np.arange(G * K)
        self._check(w_addr, isa.W_DEPTH, "weight")
        Wm = self.wmem[w_addr].astype(np.int64).reshape(G, K, N)
        acc = np.einsum("pk,gkc->pgc", A, Wm)                                         # [P, G, N]
        self._write_out(ins, self._requant(ins, acc, G))

        tiles = -(-P // N)
        gather = K if not ins.in_src else ins.ksize * ins.ksize * N   # packed: byte reads, ~1/cycle/pixel tap
        self.stats.cycles.append(tiles * ((gather + TRANSPOSE_LAT) + G * (1 + K + FLUSH_CYCLES + DRAIN_CYCLES)))
        self.stats.macs.append(P * real_k * G * N)

    def _dwconv(self, ins: Instr):
        cb, T = ins.in_cb, ins.K
        words, _ = self._gather_words(ins, ins.in_zp)                                 # [P, k, k, cb, N]
        P = words.shape[0]
        A = words.reshape(P, T, cb, N).astype(np.int64)
        w_addr = ins.w_base + np.arange(cb * T)
        self._check(w_addr, isa.W_DEPTH, "weight")
        Wm = self.wmem[w_addr].astype(np.int64).reshape(cb, T, N)
        acc = np.einsum("ptgc,gtc->pgc", A, Wm)                                       # [P, cb, N]
        self._write_out(ins, self._requant(ins, acc, cb))
        # Baseline mapping: one 8-pixel x 8-channel tile per (tile, cb), only T taps deep.
        tiles = -(-P // N)
        self.stats.cycles.append(tiles * cb * (T + TRANSPOSE_LAT + 1 + T + FLUSH_CYCLES + DRAIN_CYCLES))
        self.stats.macs.append(P * T * cb * N)

    def _maxpool(self, ins: Instr):
        words, v = self._gather_words(ins, -128)                                      # [P, k, k, cb, N]
        m = np.where(v[..., None], words, np.int8(-128)).max(axis=(1, 2))
        m = np.clip(m, ins.act_min, ins.act_max).astype(np.int8)
        self._write_out(ins, m)
        self.stats.cycles.append(m.shape[0] * ins.in_cb * (ins.ksize ** 2 + 1))
        self.stats.macs.append(0)

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
        self.stats.cycles.append(q.shape[0] * ins.in_cb * (ins.ksize ** 2 + 2))
        self.stats.macs.append(0)

    def _argmax(self, ins: Instr):
        count = ins.in_w
        idx = np.arange(count)
        v = self.act[ins.in_base + idx // N, idx % N].astype(np.int64)
        cls = int(np.argmax(v))          # first maximum, like TFLite ARG_MAX
        self.result = (cls, int(v[cls]))
        self.stats.cycles.append(count + 2)
        self.stats.macs.append(0)
