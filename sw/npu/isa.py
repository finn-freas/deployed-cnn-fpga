"""NPU instruction set and machine parameters: the single source of truth.

rtl/npu/npu_pkg.sv is generated from this file (sw/npu/gen_pkg.py); a test checks
the two stay in sync. See docs/isa.md for the semantics of each instruction.

ISA v2: adds DWCONV and AVGPOOL, lets the first CONV read the packed (byte-per-
channel) input buffer directly (in_src=1), and widens fields for MobileNet-sized
models. LOAD_INPUT from v1 is gone.
"""

from dataclasses import dataclass, field, fields
from enum import IntEnum

# ---------------------------------------------------------------- machine parameters
N = 8                 # systolic array is N x N; also channels per activation/weight word
DATA_W = 8            # int8 operands
ACC_W = 32            # int32 accumulators
ACT_DEPTH = 8192      # activation memory address space, 64-bit words
W_DEPTH = 65536       # weight store address space (streamed from flash/SDRAM), 64-bit words
P_DEPTH = 4096        # param store address space, one entry per output channel (streamed with
                      # the weights; on chip only the current layer's params are buffered)
IMEM_DEPTH = 64       # instruction memory, INSTR_W-bit words
IN_DEPTH = 32768      # input buffer, bytes (packed HWC, uint8): 96*96*3 = 27648
K_MAX = 1024          # max reduction length per CONV output (see Instr.K)
INSTR_W = 160
SHIFT_W = 6           # signed requant shift: >0 left, <=0 right


class Op(IntEnum):
    NOP = 0x0
    CONV = 0x1
    MAXPOOL = 0x2
    DWCONV = 0x3
    ARGMAX = 0x4
    AVGPOOL = 0x5
    END = 0xF


# (name, width, signed). Packed LSB-first in this order.
FIELDS = [
    ("op",       4, False),
    ("in_base", 16, False),
    ("out_base", 16, False),
    ("w_base",  16, False),
    ("p_base",  12, False),
    ("in_h",     7, False),
    ("in_w",     7, False),
    ("in_cb",    6, False),   # channel blocks; for in_src=1 the channel count C
    ("out_h",    7, False),
    ("out_w",    7, False),
    ("out_g",    6, False),
    ("ksize",    3, False),
    ("stride",   2, False),
    ("pad",      3, False),   # top/left padding; bottom/right follows from the bounds
    ("in_zp",    8, True),
    ("out_zp",   8, True),
    ("act_min",  8, True),
    ("act_max",  8, True),
    ("rmode",    1, False),   # requant rounding: 0 = double, 1 = single; see quant.py
    ("in_src",   1, False),   # CONV input: 0 = activation memory, 1 = packed input buffer
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
    in_src: int = 0
    comment: str = field(default="", compare=False)

    @property
    def K(self) -> int:
        """Reduction length per output value (CONV), or taps per channel (DWCONV)."""
        if self.op == Op.DWCONV:
            return self.ksize * self.ksize
        if self.in_src:   # packed input: k = (ky*ksize + kx)*C + c, padded to a multiple of N
            return -(-self.ksize * self.ksize * self.in_cb // N) * N
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
        skip = {"comment", "op"}
        always = {"in_base", "out_base"} if self.op not in (Op.NOP, Op.END) else set()
        parts = [f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                 if f.name not in skip and (f.name in always or getattr(self, f.name) != f.default)]
        return f"{self.op.name:8s} " + " ".join(parts) + (f"   # {self.comment}" if self.comment else "")


# ---------------------------------------------------------------- param memory word
# {shift[5:0] (signed), mult[31:0] (signed), bias[31:0] (signed)} = 70 bits
PARAM_W = SHIFT_W + 32 + 32


def encode_param(bias: int, mult: int, shift: int) -> int:
    return ((shift & 0x3F) << 64) | ((mult & 0xFFFFFFFF) << 32) | (bias & 0xFFFFFFFF)


def decode_param(word: int) -> tuple[int, int, int]:
    def s(v, w):
        return v - (1 << w) if v >> (w - 1) else v
    return s(word & 0xFFFFFFFF, 32), s((word >> 32) & 0xFFFFFFFF, 32), s((word >> 64) & 0x3F, 6)
