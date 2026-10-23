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
    macs: list[int] = field(default_factory=list)


class NpuSim:
    def __init__(self, instrs: list[Instr], wmem: np.ndarray, pmem: list[tuple[int, int, int]]):
        self.imem = list(instrs)
        self.wmem = np.asarray(wmem, np.int8).reshape(-1, N)
        p = np.zeros((isa.P_DEPTH, 3), np.int64)
        p[:len(pmem)] = pmem
        self.p_bias, self.p_mult, self.p_shift = p[:, 0], p[:, 1], p[:, 2]
        self.act = np.zeros((isa.ACT_DEPTH, N), np.int8)
        self.result = (0, 0)
        self.stats = Stats()

    @classmethod
    def from_program(cls, prog):
        return cls(prog.instrs, prog.wmem, prog.pmem)

    # ------------------------------------------------------------ helpers
    def read_tensor(self, base, h, w, cb, c):
        words = self.act[base:base + h * w * cb]
        return words.reshape(h, w, cb * N)[:, :, :c]

    @staticmethod
    def _check(addr, depth, what):
        if np.any((addr < 0) | (addr >= depth)):
            raise IndexError(f"{what} address out of range: {addr.min()}..{addr.max()} (depth {depth})")

    # ------------------------------------------------------------ execution
    def run(self, image_u8: np.ndarray, max_instrs: int = 1000):
        self.input_buffer = np.asarray(image_u8, np.uint8)
        self.stats = Stats()
        for pc in range(max_instrs):
            ins = self.imem[pc]
            if ins.op == Op.END:
                return self.result
            getattr(self, "_" + ins.op.name.lower())(ins)
        raise RuntimeError("program did not reach END")

    def _nop(self, ins):
        self.stats.cycles.append(1)
        self.stats.macs.append(0)

    def _load_input(self, ins: Instr):
        h, w = ins.in_h, ins.in_w
        px = self.input_buffer.reshape(-1)[:h * w].astype(np.int64)
        addr = ins.out_base + np.arange(h * w)
        self._check(addr, isa.ACT_DEPTH, "LOAD_INPUT dst")
        self.act[addr] = 0
        self.act[addr, 0] = ((px + ins.in_zp) & 0xFF).astype(np.uint8).view(np.int8)
        self.stats.cycles.append(h * w)
        self.stats.macs.append(0)

    def _gather(self, ins: Instr):
        """im2col: [P, K] int8 matrix, padding filled with in_zp, as the gather unit produces."""
        k, s, pad, cb = ins.ksize, ins.stride, ins.pad, ins.in_cb
        oy, ox = np.meshgrid(np.arange(ins.out_h), np.arange(ins.out_w), indexing="ij")
        oy, ox = oy.reshape(-1, 1, 1, 1), ox.reshape(-1, 1, 1, 1)
        ky = np.arange(k).reshape(1, k, 1, 1)
        kx = np.arange(k).reshape(1, 1, k, 1)
        c = np.arange(cb).reshape(1, 1, 1, cb)
        iy, ix = oy * s - pad + ky, ox * s - pad + kx
        valid = (iy >= 0) & (iy < ins.in_h) & (ix >= 0) & (ix < ins.in_w)
        addr = ins.in_base + (iy * ins.in_w + ix) * cb + c                       # [P, k, k, cb]
        valid = np.broadcast_to(valid, addr.shape)
        self._check(addr[valid], isa.ACT_DEPTH, "CONV src")
        words = np.where(valid[..., None], self.act[np.where(valid, addr, 0)], np.int8(ins.in_zp))
        return words.reshape(addr.shape[0], -1)                                   # [P, K]

    def _conv(self, ins: Instr):
        K, G = ins.K, ins.out_g
        A = self._gather(ins).astype(np.int64)
        P = A.shape[0]
        w_addr = ins.w_base + np.arange(G * K)
        self._check(w_addr, isa.W_DEPTH, "weight")
        Wm = self.wmem[w_addr].astype(np.int64).reshape(G, K, N)
        acc = np.einsum("pk,gkc->pgc", A, Wm)                                     # [P, G, N]
        p_addr = ins.p_base + np.arange(G * N)
        self._check(p_addr, isa.P_DEPTH, "param")
        y = requantize(acc, self.p_bias[p_addr].reshape(G, N), self.p_mult[p_addr].reshape(G, N),
                       self.p_shift[p_addr].reshape(G, N), ins.out_zp, ins.act_min, ins.act_max,
                       ins.rmode)
        dst = ins.out_base + np.arange(P * G)
        self._check(dst, isa.ACT_DEPTH, "CONV dst")
        self.act[dst] = y.reshape(P * G, N)

        tiles = -(-P // N)
        self.stats.cycles.append(tiles * ((K + TRANSPOSE_LAT) + G * (1 + K + FLUSH_CYCLES + DRAIN_CYCLES)))
        self.stats.macs.append(P * K * G * N)

    def _maxpool(self, ins: Instr):
        k, s, cb = ins.ksize, ins.stride, ins.in_cb
        oy, ox = np.meshgrid(np.arange(ins.out_h), np.arange(ins.out_w), indexing="ij")
        oy, ox = oy.reshape(-1, 1, 1, 1), ox.reshape(-1, 1, 1, 1)
        iy = oy * s + np.arange(k).reshape(1, k, 1, 1)
        ix = ox * s + np.arange(k).reshape(1, 1, k, 1)
        addr = ins.in_base + (iy * ins.in_w + ix) * cb + np.arange(cb).reshape(1, 1, 1, cb)
        self._check(addr, isa.ACT_DEPTH, "MAXPOOL src")
        m = self.act[addr].max(axis=(1, 2))                                       # [P, cb, N]
        m = np.clip(m, ins.act_min, ins.act_max).astype(np.int8)
        P = m.shape[0]
        dst = ins.out_base + np.arange(P * cb)
        self._check(dst, isa.ACT_DEPTH, "MAXPOOL dst")
        self.act[dst] = m.reshape(P * cb, N)
        self.stats.cycles.append(P * cb * (k * k + 1))
        self.stats.macs.append(0)

    def _argmax(self, ins: Instr):
        count = ins.in_w
        idx = np.arange(count)
        v = self.act[ins.in_base + idx // N, idx % N].astype(np.int64)
        cls = int(np.argmax(v))          # first maximum, like TFLite ARG_MAX
        self.result = (cls, int(v[cls]))
        self.stats.cycles.append(count + 2)
        self.stats.macs.append(0)
