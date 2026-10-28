from dataclasses import dataclass, field, fields
from enum import IntEnum

N = 8
DATA_W = 8
ACC_W = 26
ACT_DEPTH = 7168
EXT_DEPTH = 65536
WBUF_DEPTH = 1024
PBUF_DEPTH = 256
PARAM_WORDS = 2
IMEM_DEPTH = 64
K_MAX = 1024
INSTR_W = 160
SHIFT_W = 6

class Op(IntEnum):
    NOP = 0x0
    CONV = 0x1
    MAXPOOL = 0x2
    DWCONV = 0x3
    ARGMAX = 0x4
    AVGPOOL = 0x5
    END = 0xF

FIELDS = [
    ("op",       4, False),
    ("in_base", 16, False),
    ("out_base", 16, False),
    ("w_base",  16, False),
    ("p_base",  16, False),
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
    ("rmode",    1, False),
    ("in_src",   1, False),
    ("uniform",  1, False),
]
assert sum(w for _, w, _ in FIELDS) <= INSTR_W
assert K_MAX * 128 * 128 < 1 << (ACC_W - 1), "accumulators must hold K_MAX int8 products"

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
    uniform: int = 0
    comment: str = field(default="", compare=False)

    @property
    def K(self) -> int:
        if self.op == Op.DWCONV:
            return self.ksize * self.ksize
        if self.in_src:
            return self.ksize * self.packed_row_k
        return self.ksize * self.ksize * self.in_cb * N

    @property
    def packed_row_k(self) -> int:
        return -(-self.ksize * self.in_cb // N) * N

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

PARAM_W = SHIFT_W + 32 + 32

def encode_param(bias: int, mult: int, shift: int) -> int:
    return ((shift & 0x3F) << 64) | ((mult & 0xFFFFFFFF) << 32) | (bias & 0xFFFFFFFF)

def decode_param(word: int) -> tuple[int, int, int]:
    def s(v, w):
        return v - (1 << w) if v >> (w - 1) else v
    return s(word & 0xFFFFFFFF, 32), s((word >> 32) & 0xFFFFFFFF, 32), s((word >> 64) & 0x3F, 6)

def param_ext_words(bias: int, mult: int, shift: int) -> tuple[int, int]:
    return (((mult & 0xFFFFFFFF) << 32) | (bias & 0xFFFFFFFF), shift & 0xFFFFFFFFFFFFFFFF)
