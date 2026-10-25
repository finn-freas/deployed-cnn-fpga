"""npuc: compile a quantized classifier for the NPU.

    python -m npuc model.onnx -o build/npuc/model          (run from sw/)
    python -m npuc model.onnx --verify 50 --images val.npz --flash-image cat.png

Writes the NPU memory images (program.hex, ext.hex, meta.json, program.txt), the IR
(ir.json, ir.npz: `python -m npu.board <out>` builds board images from it), report.md, and
with --flash-image a board flash image (flash.bin).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from npu import isa
from npu.compiler import compile_ir
from npu.isa import Op

from .onnx_frontend import UnsupportedModel, import_onnx
from .verify import run_sim, verify

NPU_MHZ = 54.0          # npu_board_top's NPU clock
MACS_PER_CYCLE = isa.N * isa.N


def load_model(path: Path, rounding: str):
    if path.suffix == ".onnx":
        return import_onnx(path, rounding)
    if path.suffix == ".tflite":
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
            from export_ir import export
        except ImportError as e:
            raise SystemExit(f"npuc: .tflite input needs TensorFlow (.venv-tf): {e}")
        _, ir, arrays = export(path)
        return ir, arrays
    raise SystemExit(f"npuc: unknown model format {path.suffix!r} (.onnx or .tflite)")


def load_images(spec, h, w, c, n, seed=0):
    if spec is None:                                  # uniform noise exercises every code
        return np.random.default_rng(seed).integers(0, 256, (n, h, w, c)).astype(np.uint8)
    p = Path(spec)
    if p.suffix == ".npz":
        imgs = np.load(p)["images"][:n].astype(np.uint8)
        return imgs.reshape(len(imgs), h, w, c)
    from npu.board import load_image
    files = sorted(f for f in (p.iterdir() if p.is_dir() else [p])
                   if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".npy"))[:n]
    return np.stack([load_image(f, h, w, c) for f in files])


def layer_rows(ir, prog, stats):
    by_instr = {L["instr"]: L for L in prog.meta["layers"]}
    rows = []
    for pc, ins in enumerate(prog.instrs[:-1]):
        L = by_instr.get(pc)
        src = next((x for x in ir["layers"] if L and x["name"] == L["name"]), None)
        rows.append({
            "layer": L["name"] if L else "argmax",
            "source": (src.get("onnx_node") or src["name"]) if src else "",
            "npu": ins.op.name + (" (uniform)" if ins.uniform else ""),
            "shape": f"{'x'.join(map(str, src['in_shape']))} -> {'x'.join(map(str, src['out_shape']))}" if src else "",
            "macs": stats.macs[pc], "cycles": stats.cycles[pc]})
    return rows


def render_report(model, ir, prog, rows, total, v, args):
    lines = [f"# npuc report: {model.name}", ""]
    macs = sum(r["macs"] for r in rows)
    ms = total / (NPU_MHZ * 1e3)
    lines += ["| | |", "|-|-|",
              f"| Latency | {total:,} cycles = {ms:.2f} ms at {NPU_MHZ:g} MHz (ideal memory) |",
              f"| Work | {macs / 1e6:.2f} M MACs, {macs / total:.1f} MAC/cycle "
              f"({100 * macs / total / MACS_PER_CYCLE:.0f}% of the 8x8 array) |",
              f"| Weights + params | {prog.ext_used * 8 / 1024:.1f} KiB of {isa.EXT_DEPTH * 8 // 1024} KiB |",
              f"| Program | {len(prog.instrs)} of {isa.IMEM_DEPTH} instructions |",
              f"| Input | {'x'.join(map(str, ir['input']['shape']))} uint8 pixels (scale {ir['input']['scale']:.6g}) |",
              f"| Rounding | {ir.get('rounding', 'tflite')} |", ""]
    if v is not None:
        exact = sum(d.exact for d in v.layers)
        vals = sum(d.values for d in v.layers)
        lines += [f"**Check against ONNX Runtime** ({v.images} images): class agrees on "
                  f"{v.argmax_agree}/{v.images}; {exact}/{vals} layer outputs identical "
                  f"({100 * exact / vals:.3f}%), max difference "
                  f"{max(d.max_diff for d in v.layers)} LSB.", ""]
    lines += ["| # | Layer | Source node | NPU | Shape (HxWxC) | MACs | Cycles | % |",
              "|-|-|-|-|-|-:|-:|-:|"]
    for i, r in enumerate(rows):
        lines.append(f"| {i} | {r['layer']} | {r['source']} | {r['npu']} | {r['shape']} | "
                     f"{r['macs']:,} | {r['cycles']:,} | {100 * r['cycles'] / total:.1f} |")
    if v is not None:
        lines += ["", "| Layer | Identical outputs | Max diff (LSB) |", "|-|-:|-:|"]
        lines += [f"| {d.name} | {100 * d.exact / d.values:.3f}% | {d.max_diff} |" for d in v.layers]
    notes = list(ir.get("notes", []))
    if any(i.op == Op.MAXPOOL for i in prog.instrs):
        notes.append("uses MAXPOOL: needs a bitstream with the pool unit (npu_board_top is "
                     "built with HAS_POOL=0)")
    if ir["output"].get("softmax_dropped"):
        notes.append("the final Softmax is dropped (argmax is unchanged); the NPU returns logits")
    if notes:
        lines += ["", "Notes:", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="npuc", description="Compile a quantized classifier for the NPU.")
    ap.add_argument("model", type=Path, help=".onnx (QDQ or QOperator int8) or .tflite (int8)")
    ap.add_argument("-o", "--out", type=Path, help="output directory (default build/npuc/<model>)")
    ap.add_argument("--rounding", choices=["onnx", "tflite"], default=None,
                    help="requant rounding: closest to ONNX's float rounding (default for .onnx) "
                         "or TFLite's per-op integer rounding")
    ap.add_argument("--verify", type=int, default=10, metavar="N",
                    help="compare N images against ONNX Runtime, layer by layer (0: skip)")
    ap.add_argument("--images", help=".npz with 'images' (NHWC uint8), an image file or a "
                                     "directory of images; default: random pixels")
    ap.add_argument("--flash-image", type=Path, help="also write flash.bin for the board with this demo image")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or Path("build/npuc") / args.model.stem
    rounding = args.rounding or "onnx"
    try:
        ir, arrays = load_model(args.model, rounding)
        prog = compile_ir(ir, arrays)
    except UnsupportedModel as e:
        raise SystemExit(f"npuc: {args.model.name}: unsupported model: {e}")
    except (NotImplementedError, MemoryError) as e:
        raise SystemExit(f"npuc: {args.model.name}: does not fit the NPU: {e}")

    h, w, c = ir["input"]["shape"]
    images = load_images(args.images, h, w, c, max(args.verify, 1))
    _, _, _, stats, total = run_sim(prog, images[0])
    v = None
    if args.verify and args.model.suffix == ".onnx":
        v = verify(prog, ir, args.model, images[:args.verify])

    out.mkdir(parents=True, exist_ok=True)
    prog.write(out)
    (out / "ir.json").write_text(json.dumps(ir, indent=2))
    np.savez(out / "ir.npz", **arrays)
    rows = layer_rows(ir, prog, stats)
    report = render_report(args.model, ir, prog, rows, total, v, args)
    (out / "report.md").write_text(report)
    if args.flash_image:
        from npu.board import flash_words, load_image
        words = flash_words(prog, load_image(args.flash_image, h, w, c))
        (out / "flash.bin").write_bytes(np.array(words, dtype="<u8").tobytes())

    if not args.quiet:
        print(report, end="")
        print(f"-> {out}")
    if v is not None and v.argmax_agree != v.images:
        sys.exit(3)


if __name__ == "__main__":
    main()
