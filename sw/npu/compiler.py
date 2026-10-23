"""Compile the exported IR into NPU memory images.

    python -m npu.compiler build/model --out build/npu     (run from sw/)

Outputs program.hex / weights.hex / params.hex for $readmemh, meta.json (memory map
and per-layer tensor locations), and program.txt (disassembly).
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import isa
from .isa import N, Instr, Op
from .quant import (INT32_MAX, INT32_MIN, RMODE_DOUBLE, RMODE_SINGLE, effective_scale,
                    quantize_multiplier)


def ceil_div(a, b):
    return -(-a // b)


@dataclass
class Program:
    instrs: list[Instr] = field(default_factory=list)
    wmem: np.ndarray = field(default_factory=lambda: np.zeros((isa.W_DEPTH, N), np.int8))
    pmem: list[tuple[int, int, int]] = field(default_factory=list)   # (bias, mult, shift)
    w_used: int = 0
    meta: dict = field(default_factory=dict)

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        (out / "program.hex").write_text("".join(f"{i.encode():032x}\n" for i in self.instrs))
        words = self.wmem[:self.w_used].view(np.uint8)
        (out / "weights.hex").write_text(
            "".join("".join(f"{b:02x}" for b in row[::-1]) + "\n" for row in words))  # byte 0 = LSB
        (out / "params.hex").write_text(
            "".join(f"{isa.encode_param(*p):018x}\n" for p in self.pmem))
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2))
        (out / "program.txt").write_text("".join(f"{n:2d}: {i}\n" for n, i in enumerate(self.instrs)))


class ActAllocator:
    """Ping-pong placement: each output goes at the opposite end of memory from its input."""

    def __init__(self, depth):
        self.depth = depth

    def place(self, size, avoid):
        lo, hi = avoid
        base = self.depth - size if lo < self.depth // 2 else 0
        if base < 0 or not (base + size <= lo or base >= hi):
            raise MemoryError(f"activation memory: cannot place {size} words clear of [{lo},{hi})")
        return base


def act_range(act: str, out_zp: int, out_scale: float):
    if act == "none":
        return -128, 127
    if act == "relu":
        return max(-128, out_zp), 127
    if act == "relu6":
        return max(-128, out_zp), min(127, out_zp + round(6.0 / out_scale))
    raise ValueError(act)


def compile_ir(ir: dict, arrays: dict) -> Program:
    prog = Program()
    alloc = ActAllocator(isa.ACT_DEPTH)
    H, W, C = ir["input"]["shape"]
    assert C == 1, "LOAD_INPUT supports single-channel (grayscale) input"
    cur = {"base": 0, "h": H, "w": W, "cb": 1, "c": 1}
    prog.instrs.append(Instr(Op.LOAD_INPUT, out_base=0, in_h=H, in_w=W, in_zp=ir["input"]["zp"],
                             comment=f"input {H}x{W} u8 -> i8"))
    prog.meta = {"input": dict(cur), "layers": [], "isa": {
        "N": N, "ACT_DEPTH": isa.ACT_DEPTH, "W_DEPTH": isa.W_DEPTH, "P_DEPTH": isa.P_DEPTH}}

    for L in ir["layers"]:
        ih, iw, ic = L["in_shape"]
        oh, ow, oc = L["out_shape"]
        assert (ih, iw, ceil_div(ic, N)) == (cur["h"], cur["w"], cur["cb"]), f"{L['name']}: shape chain"
        in_cb = cur["cb"]
        out_g = ceil_div(oc, N)
        size = oh * ow * out_g
        base = alloc.place(size, (cur["base"], cur["base"] + cur["h"] * cur["w"] * cur["cb"]))

        if L["op"] == "conv":
            k = L["kh"]
            assert L["kh"] == L["kw"], "square kernels only"
            w = arrays[L["name"] + ".w"].astype(np.int64)          # [O, k, k, I]
            b = arrays[L["name"] + ".b"].astype(np.int64)
            w_scale = arrays[L["name"] + ".w_scale"]
            K = k * k * in_cb * N
            assert K <= isa.K_MAX, f"{L['name']}: K={K} > K_MAX"

            # Weights: pad I->Cb*8, O->G*8; rows ordered [g][ky][kx][cb*8+cin], 8 output channels per row.
            wp = np.zeros((out_g * N, k, k, in_cb * N), np.int64)
            wp[:w.shape[0], :, :, :w.shape[3]] = w
            rows = wp.reshape(out_g, N, k, k, in_cb * N).transpose(0, 2, 3, 4, 1).reshape(out_g * K, N)
            w_base = prog.w_used
            if w_base + rows.shape[0] > isa.W_DEPTH:
                raise MemoryError(f"weight memory overflow at {L['name']}")
            prog.wmem[w_base:w_base + rows.shape[0]] = rows.astype(np.int8)
            prog.w_used += rows.shape[0]

            # Params: fold the input zero point into the bias (padding is filled with in_zp).
            p_base = len(prog.pmem)
            for o in range(out_g * N):
                if o < w.shape[0]:
                    bias = int(b[o] - L["in_zp"] * w[o].sum())
                    assert INT32_MIN <= bias <= INT32_MAX
                    mult, shift = quantize_multiplier(
                        effective_scale(L["in_scale"], w_scale[o], L["out_scale"]))
                    prog.pmem.append((bias, mult, shift))
                else:
                    prog.pmem.append((0, 0, 0))
            if len(prog.pmem) > isa.P_DEPTH:
                raise MemoryError("param memory overflow")

            amin, amax = act_range(L["act"], L["out_zp"], L["out_scale"])
            prog.instrs.append(Instr(
                Op.CONV, in_base=cur["base"], out_base=base, w_base=w_base, p_base=p_base,
                in_h=ih, in_w=iw, in_cb=in_cb, out_h=oh, out_w=ow, out_g=out_g,
                ksize=k, stride=L["stride"], pad=L["pad"], in_zp=L["in_zp"], out_zp=L["out_zp"],
                act_min=amin, act_max=amax, comment=L["name"],
                rmode=RMODE_SINGLE if L.get("from_fc") else RMODE_DOUBLE))
        elif L["op"] == "maxpool":
            prog.instrs.append(Instr(
                Op.MAXPOOL, in_base=cur["base"], out_base=base, in_h=ih, in_w=iw, in_cb=in_cb,
                out_h=oh, out_w=ow, out_g=in_cb, ksize=L["k"], stride=L["stride"], comment=L["name"]))
        else:
            raise NotImplementedError(L["op"])

        cur = {"base": base, "h": oh, "w": ow, "cb": out_g, "c": oc}
        prog.meta["layers"].append({"name": L["name"], "op": L["op"], "instr": len(prog.instrs) - 1, **cur})

    assert cur["h"] == cur["w"] == 1
    prog.instrs.append(Instr(Op.ARGMAX, in_base=cur["base"], in_w=cur["c"], comment="class = argmax"))
    prog.instrs.append(Instr(Op.END))
    assert len(prog.instrs) <= isa.IMEM_DEPTH
    prog.meta.update(w_used=prog.w_used, p_used=len(prog.pmem), n_instr=len(prog.instrs),
                     classes=cur["c"])
    return prog


def load_ir(model_dir: Path):
    ir = json.loads((model_dir / "ir.json").read_text())
    arrays = dict(np.load(model_dir / "ir.npz"))
    return ir, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("build/npu"))
    args = ap.parse_args()
    prog = compile_ir(*load_ir(args.model_dir))
    prog.write(args.out)
    print((args.out / "program.txt").read_text(), end="")
    print(f"weights {prog.w_used}/{isa.W_DEPTH} words, params {len(prog.pmem)}/{isa.P_DEPTH}, "
          f"instrs {len(prog.instrs)}/{isa.IMEM_DEPTH} -> {args.out}")


if __name__ == "__main__":
    main()
