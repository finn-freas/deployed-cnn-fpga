"""NPU instruction set and machine parameters: the single source of truth.

rtl/npu/npu_pkg.sv is generated from this file (sw/npu/gen_pkg.py); a test checks
the two stay in sync. See docs/isa.md for the semantics of each instruction.
"""

from dataclasses import dataclass, field, fields
from enum import IntEnum

# ---------------------------------------------------------------- machine parameters
N = 8                 # systolic array is N x N; also channels per activation/weight word
DATA_W = 8            # int8 operands
ACC_W = 32            # int32 accumulators
ACT_DEPTH = 2048      # activation memory, 64-bit words
W_DEPTH = 2048        # weight memory, 64-bit words
P_DEPTH = 256         # param memory, one entry per output channel
IMEM_DEPTH = 32       # instruction memory, 128-bit words
K_MAX = 1024          # max reduction length (kh*kw*Cb*8) per CONV
INSTR_W = 128
SHIFT_W = 6           # signed requant shift: >0 left, <=0 right


class Op(IntEnum):
    NOP = 0x0
    CONV = 0x1
    MAXPOOL = 0x2
    LOAD_INPUT = 0x3
    ARGMAX = 0x4
    END = 0xF


# (name, width, signed). Packed LSB-first in this order.
FIELDS = [
    ("op",       4, False),
    ("in_base", 11, False),
    ("out_base", 11, False),
    ("w_base",  11, False),
    ("p_base",   8, False),
    ("in_h",     7, False),
    ("in_w",     7, False),
    ("in_cb",    6, False),
    ("out_h",    7, False),
    ("out_w",    7, False),
    ("out_g",    6, False),
    ("ksize",    3, False),
    ("stride",   2, False),
    ("pad",      3, False),
    ("in_zp",    8, True),
    ("out_zp",   8, True),
    ("act_min",  8, True),
    ("act_max",  8, True),
    ("rmode",    1, False),   # requant rounding: 0 = double (conv), 1 = single (FC); see quant.py
]
assert sum(w for _, w, _ in FIELDS) <= INSTR_W


@dataclass
class Instr:
    op: Op = Op.NOP
    in_base: int = 0
    out_base: int = 0
    w_base: int = 0
    p_base: int = 0
    in_h: int = 0
    in_w: int = 0
    in_cb: int = 0
    out_h: int = 0
    out_w: int = 0
    out_g: int = 0
    ksize: int = 0
    stride: int = 0
    pad: int = 0
    in_zp: int = 0
    out_zp: int = 0
    act_min: int = -128
    act_max: int = 127
    rmode: int = 0
    comment: str = field(default="", compare=False)

    @property
    def K(self) -> int:
        """Reduction length of a CONV."""
        return self.ksize * self.ksize * self.in_cb * N

    def encode(self) -> int:
        word, pos = 0, 0
        for name, width, signed in FIELDS:
            v = int(getattr(self, name))
            lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
            if not lo <= v <= hi:
                raise ValueError(f"{name}={v} does not fit in {width} {'signed' if signed else 'unsigned'} bits")
            word |= (v & ((1 << width) - 1)) << pos
            pos += width
        return word

    @classmethod
    def decode(cls, word: int) -> "Instr":
        kw, pos = {}, 0
        for name, width, signed in FIELDS:
            v = (word >> pos) & ((1 << width) - 1)
            if signed and v >> (width - 1):
                v -= 1 << width
            kw[name] = Op(v) if name == "op" else v
            pos += width
        return cls(**kw)

    def __str__(self) -> str:
        skip = {"comment"}
        parts = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                 if f.name not in skip and f.name != "op" and getattr(self, f.name) != f.default]
        return f"{self.op.name:10s} " + " ".join(parts) + (f"   # {self.comment}" if self.comment else "")


# ---------------------------------------------------------------- param memory word
# {shift[5:0] (signed), mult[31:0] (signed), bias[31:0] (signed)} = 70 bits
PARAM_W = SHIFT_W + 32 + 32


def encode_param(bias: int, mult: int, shift: int) -> int:
    return ((shift & 0x3F) << 64) | ((mult & 0xFFFFFFFF) << 32) | (bias & 0xFFFFFFFF)


def decode_param(word: int) -> tuple[int, int, int]:
    def s(v, w):
        return v - (1 << w) if v >> (w - 1) else v
    return s(word & 0xFFFFFFFF, 32), s((word >> 32) & 0xFFFFFFFF, 32), s((word >> 64) & 0x3F, 6)
