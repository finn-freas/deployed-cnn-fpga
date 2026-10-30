import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import isa
from .isa import N, Instr, Op
from .quant import (INT32_MAX, INT32_MIN, RMODE_DOUBLE, RMODE_SINGLE, effective_scale,
                    quantize_multiplier)

RMODE = {"conv": RMODE_DOUBLE, "fc": RMODE_SINGLE, "dwconv": RMODE_DOUBLE}
RMODE_NAMES = {"double": RMODE_DOUBLE, "single": RMODE_SINGLE}

def layer_rmode(L: dict, kind: str) -> int:
    return RMODE_NAMES[L["rmode"]] if "rmode" in L else RMODE[kind]

def ceil_div(a, b):
    return -(-a // b)

@dataclass
class Program:
    instrs: list[Instr] = field(default_factory=list)
    ext: np.ndarray = field(default_factory=lambda: np.zeros((isa.EXT_DEPTH, N), np.int8))
    ext_used: int = 0
    params: list[tuple[int, int, int]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def ext_words(self) -> list[int]:
        return np.frombuffer(self.ext[:self.ext_used].view(np.uint8).tobytes(), dtype="<u8").tolist()

    def write(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        hexw = isa.INSTR_W // 4
        (out / "program.hex").write_text("".join(f"{i.encode():0{hexw}x}\n" for i in self.instrs))
        (out / "ext.hex").write_text("".join(f"{w:016x}\n" for w in self.ext_words()))
        (out / "meta.json").write_text(json.dumps(self.meta, indent=2))
        (out / "program.txt").write_text("".join(f"{n:2d}: {i}\n" for n, i in enumerate(self.instrs)))

    def _append(self, rows: np.ndarray, what: str) -> int:
        base = self.ext_used
        if base + rows.shape[0] > isa.EXT_DEPTH:
            raise MemoryError(f"external store overflow at {what}")
        self.ext[base:base + rows.shape[0]] = rows
        self.ext_used += rows.shape[0]
        return base

    def add_weights(self, rows: np.ndarray, what: str) -> int:
        if rows.shape[0] > isa.WBUF_DEPTH * 64:
            raise MemoryError(f"{what}: weights too large")
        return self._append(rows.astype(np.int8), what)

    def add_params(self, bias, w_sum, in_zp, scales, n_padded) -> int:
        params = []
        for o in range(n_padded):
            if o < len(bias):
                b = int(bias[o] - in_zp * w_sum[o])
                assert INT32_MIN <= b <= INT32_MAX
                params.append((b, *quantize_multiplier(scales[o])))
            else:
                params.append((0, 0, 0))
        return self.add_raw_params(params)

    def add_raw_params(self, params) -> int:
        if len(params) > isa.PBUF_DEPTH:
            raise MemoryError(f"{len(params)} output channels exceed the param buffer")
        words = []
        for p in params:
            self.params.append(p)
            words.extend(isa.param_ext_words(*p))
        rows = np.frombuffer(np.array(words, dtype="<u8").tobytes(), dtype=np.int8).reshape(-1, N)
        return self._append(rows, "params")

def avgpool_multiplier(n: int) -> tuple[int, int] | None:
    if n < 2:
        return None
    mult, shift = quantize_multiplier(1.0 / n)
    if shift > 0:
        return None
    s = np.arange(-128 * n, 127 * n + 1, dtype=np.int64)
    ref = np.where(s > 0, (s + n // 2) // n, -((-(s - n // 2)) // n))
    total = 31 - shift
    got = (s * mult + (1 << (total - 1))) >> total
    return (mult, shift) if np.array_equal(got, ref) else None

def check_single(prog: Program, rmode: int, n: int, name: str):
    if rmode == RMODE_SINGLE and any(s > 0 for _, _, s in prog.params[-n:]):
        raise NotImplementedError(f"{name}: single rounding needs scale < 1 (shift <= 0)")

class ActAllocator:

    def __init__(self, depth):
        self.depth = depth

    def place(self, size, avoid):
        lo, hi = avoid
        room_below, room_above = lo, self.depth - hi
        for base, fits in sorted([(0, size <= room_below), (self.depth - size, size <= room_above)],
                                 key=lambda c: -(room_below if c[0] == 0 else room_above)):
            if fits:
                return base
        raise MemoryError(f"activation memory: cannot place {size} words clear of [{lo},{hi})")

def act_range(act: str, out_zp: int, out_scale: float):
    if act in ("none", ""):
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
    in_words = ceil_div(H * W * C, N)
    if in_words > isa.ACT_DEPTH:
        raise MemoryError("input does not fit activation memory")
    cur = {"src": "input", "base": 0, "h": H, "w": W, "c": C, "cb": None, "words": in_words}
    prog.meta = {"input": {"base": 0, "words": in_words, "h": H, "w": W, "c": C}, "layers": [],
                 "isa": {"N": N, "ACT_DEPTH": isa.ACT_DEPTH, "EXT_DEPTH": isa.EXT_DEPTH,
                         "INSTR_W": isa.INSTR_W},
                 "softmax_dropped": bool(ir["output"].get("softmax_dropped"))}

    arrays = dict(arrays)
    for L in ir["layers"]:
        if L["op"] == "dwconv" and L["kh"] == 1:
            w = arrays[L["name"] + ".w"]
            wd = np.zeros((w.shape[0], 1, 1, w.shape[0]), w.dtype)
            wd[np.arange(w.shape[0]), 0, 0, np.arange(w.shape[0])] = w.reshape(-1)
            arrays[L["name"] + ".w"] = wd
            L = dict(L, op="conv")
        ih, iw, ic = L["in_shape"]
        oh, ow, oc = L["out_shape"]
        assert (ih, iw, ic) == (cur["h"], cur["w"], cur["c"]), f"{L['name']}: shape chain"
        from_input = cur["src"] == "input"
        if from_input and L["op"] != "conv":
            raise NotImplementedError("the first layer must be a CONV (it reads the packed input)")
        out_g = ceil_div(oc, N)
        base = alloc.place(oh * ow * out_g, (cur["base"], cur["base"] + cur["words"]))
        if "act_min" in L:
            amin, amax = L["act_min"], L["act_max"]
        else:
            amin, amax = act_range(L.get("act", "none"), L["out_zp"], L["out_scale"])
        common = dict(in_base=cur["base"], out_base=base, in_h=ih, in_w=iw, out_h=oh, out_w=ow,
                      out_g=out_g, stride=L["stride"], pad=L.get("pad", 0), in_zp=L["in_zp"],
                      out_zp=L["out_zp"], act_min=amin, act_max=amax, comment=L["name"])

        if L["op"] == "conv":
            k = L["kh"]
            assert L["kh"] == L["kw"], "square kernels only"
            w = arrays[L["name"] + ".w"].astype(np.int64)
            b = arrays[L["name"] + ".b"].astype(np.int64)
            w_scale = arrays[L["name"] + ".w_scale"]
            if from_input:
                ins = Instr(Op.CONV, in_cb=C, in_src=1, ksize=k)
                K, KR = ins.K, ins.packed_row_k
                flat = np.zeros((out_g * N, k, KR), np.int64)
                flat[:w.shape[0], :, :k * ic] = w.reshape(w.shape[0], k, k * ic)
                flat = flat.reshape(out_g * N, K)
            else:
                cb = cur["cb"]
                ins = Instr(Op.CONV, in_cb=cb, ksize=k)
                K = ins.K
                wp = np.zeros((out_g * N, k, k, cb * N), np.int64)
                wp[:w.shape[0], :, :, :w.shape[3]] = w
                flat = wp.reshape(out_g * N, K)
            assert K <= isa.K_MAX, f"{L['name']}: K={K} > K_MAX"
            rows = flat.reshape(out_g, N, K).transpose(0, 2, 1).reshape(out_g * K, N)
            w_base = prog.add_weights(rows, L["name"])
            scales = [effective_scale(L["in_scale"], s, L["out_scale"]) for s in w_scale]
            p_base = prog.add_params(b, w.reshape(w.shape[0], -1).sum(1), L["in_zp"], scales,
                                     out_g * N)
            rmode = layer_rmode(L, "fc" if L.get("from_fc") else "conv")
            check_single(prog, rmode, out_g * N, L["name"])
            prog.instrs.append(Instr(Op.CONV, in_cb=ins.in_cb, in_src=ins.in_src, ksize=k,
                                     w_base=w_base, p_base=p_base, rmode=rmode, **common))
        elif L["op"] == "dwconv":
            k = L["kh"]
            cb = cur["cb"]
            assert out_g == cb
            if not 2 <= k <= 3 or L["stride"] not in (1, 2):
                raise NotImplementedError(f"{L['name']}: the depthwise unit supports k 2..3, stride 1/2")
            w = arrays[L["name"] + ".w"].astype(np.int64)
            b = arrays[L["name"] + ".b"].astype(np.int64)
            w_scale = arrays[L["name"] + ".w_scale"]
            wp = np.zeros((cb * N, k * k), np.int64)
            wp[:w.shape[0]] = w.reshape(w.shape[0], -1)
            rows = wp.reshape(cb, N, k * k).transpose(0, 2, 1).reshape(cb * k * k, N)
            w_base = prog.add_weights(rows, L["name"])
            scales = [effective_scale(L["in_scale"], s, L["out_scale"]) for s in w_scale]
            p_base = prog.add_params(b, w.reshape(w.shape[0], -1).sum(1), L["in_zp"], scales, cb * N)
            rmode = layer_rmode(L, "dwconv")
            check_single(prog, rmode, cb * N, L["name"])
            prog.instrs.append(Instr(Op.DWCONV, in_cb=cb, ksize=k, w_base=w_base, p_base=p_base,
                                     rmode=rmode, **common))
        elif L["op"] == "maxpool":
            assert out_g == cur["cb"]
            prog.instrs.append(Instr(Op.MAXPOOL, in_cb=cur["cb"], ksize=L["k"], **common))
        elif L["op"] == "avgpool":
            cb, k = cur["cb"], L["k"]
            n = k * k
            stride = 1 if oh == ow == 1 else L["stride"]
            if L.get("pad", 0) or (oh - 1) * stride + k > ih or (ow - 1) * stride + k > iw:
                raise NotImplementedError(f"{L['name']}: AVGPOOL windows must lie inside the input")
            same_q = L["in_scale"] == L["out_scale"] and L["in_zp"] == L["out_zp"]
            mq = avgpool_multiplier(n) if same_q else None
            if mq is not None:
                param, zin, zout = (0, *mq), 0, 0
            elif L.get("approx"):
                m, sh = quantize_multiplier(L["in_scale"] / (n * L["out_scale"]))
                param, zin, zout = (-n * L["in_zp"], m, sh), 0, L["out_zp"]
            else:
                raise NotImplementedError(f"{L['name']}: no multiplier reproduces TFLite's "
                                          f"AVGPOOL division for a {k}x{k} window (needs odd k*k)")
            rmode = RMODE_SINGLE if param[2] <= 0 else RMODE_DOUBLE
            common.update(stride=stride, pad=0, in_zp=zin, out_zp=zout)
            if 2 <= k <= 3 and stride in (1, 2):
                w_base = prog.add_weights(np.ones((n, N), np.int64), L["name"])
                p_base = prog.add_raw_params([param])
                prog.instrs.append(Instr(Op.DWCONV, in_cb=cb, ksize=k, w_base=w_base, p_base=p_base,
                                         rmode=rmode, uniform=1, **common))
            else:
                ins = Instr(Op.CONV, in_cb=cb, ksize=k)
                K = ins.K
                if K > isa.K_MAX:
                    raise NotImplementedError(f"{L['name']}: {k}x{k} AVGPOOL over {ic} channels "
                                              f"needs K={K} > K_MAX={isa.K_MAX}")
                wp = np.zeros((cb * N, k, k, cb * N), np.int64)
                wp[np.arange(ic), :, :, np.arange(ic)] = 1
                rows = wp.reshape(out_g, N, K).transpose(0, 2, 1).reshape(out_g * K, N)
                w_base = prog.add_weights(rows, L["name"])
                p_base = prog.add_raw_params([param] * ic + [(0, 0, 0)] * (cb * N - ic))
                prog.instrs.append(Instr(Op.CONV, in_cb=cb, ksize=k, w_base=w_base, p_base=p_base,
                                         rmode=rmode, **common))
        else:
            raise NotImplementedError(L["op"])

        cur = {"src": "act", "base": base, "h": oh, "w": ow, "c": oc, "cb": out_g,
               "words": oh * ow * out_g}
        prog.meta["layers"].append({"name": L["name"], "op": L["op"], "instr": len(prog.instrs) - 1,
                                    **{k: cur[k] for k in ("base", "h", "w", "cb", "c")}})

    assert cur["h"] == cur["w"] == 1
    prog.instrs.append(Instr(Op.ARGMAX, in_base=cur["base"], in_w=cur["c"], comment="class = argmax"))
    prog.instrs.append(Instr(Op.END))
    assert len(prog.instrs) <= isa.IMEM_DEPTH, "program too long"
    for ins in prog.instrs:
        ins.encode()
    prog.meta.update(ext_used=prog.ext_used, n_params=len(prog.params), n_instr=len(prog.instrs),
                     classes=cur["c"])
    return prog

def load_ir(model_dir: Path):
    missing = [f for f in ("ir.json", "ir.npz") if not (model_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{model_dir}: missing {' and '.join(missing)} "
                                f"(a model directory holds the exported ir.json and ir.npz)")
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
    print(f"external store {prog.ext_used}/{isa.EXT_DEPTH} words ({prog.ext_used * 8 / 1024:.1f} KiB: "
          f"weights + {len(prog.params)} params), instrs {len(prog.instrs)}/{isa.IMEM_DEPTH} -> {args.out}")

if __name__ == "__main__":
    main()
