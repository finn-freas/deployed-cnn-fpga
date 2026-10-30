import argparse
import subprocess
from pathlib import Path

import numpy as np

from . import isa
from .compiler import compile_ir, load_ir
from .isa import Op

FLASH_BASE = 0x40_0000
BOOT_W = 0x1_0000
MAGIC = 0b101101

def boot_words(prog, image_u8: np.ndarray) -> list[int]:
    meta = prog.meta["input"]
    h, w, c = meta["h"], meta["w"], meta["c"]
    img = np.asarray(image_u8, np.uint8).reshape(h, w, c)
    beats = isa.INSTR_W // 32
    beat_list = [(ins.encode() >> (32 * b)) & 0xFFFFFFFF for ins in prog.instrs for b in range(beats)]
    if len(beat_list) % 2:
        beat_list.append(0)
    flat = img.reshape(-1)
    in_words = -(-flat.size // 8)
    buf = np.zeros(in_words * 8, np.uint8)
    buf[:flat.size] = flat
    assert w < 128 and c < 8
    header = (len(beat_list) | meta["base"] << 16 | in_words << 32 | w << 48 | c << 55 | MAGIC << 58)
    prog_words = [beat_list[i] | beat_list[i + 1] << 32 for i in range(0, len(beat_list), 2)]
    return [header, len(prog.ext_words())] + prog_words + np.frombuffer(buf.tobytes(), "<u8").tolist()

def flash_words(prog, image_u8: np.ndarray) -> list[int]:
    ext = prog.ext_words()
    assert len(ext) <= BOOT_W, "external store image overlaps the boot block"
    return ext + [0] * (BOOT_W - len(ext)) + boot_words(prog, image_u8)

def image_sum(words) -> int:
    return sum((int(w) & 0xFFFFFFFF) + (int(w) >> 32) for w in words) & 0xFFFFFFFF

def load_image(path: Path, h: int, w: int, c: int) -> np.ndarray:
    if not Path(path).is_file():
        raise SystemExit(f"{path}: no such image file")
    if path.suffix == ".npy":
        arr = np.load(path)
        if arr.size != h * w * c:
            raise SystemExit(f"{path}: {arr.size} values, the model input {h}x{w}x{c} needs {h * w * c}")
        return arr.astype(np.uint8).reshape(h, w, c)
    from PIL import Image
    im = Image.open(path).convert("RGB" if c == 3 else "L").resize((w, h))
    return np.asarray(im, np.uint8).reshape(h, w, c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--image", type=Path,
                    help="demo image (png/jpg/npy); default: golden image 0 if the model directory "
                         "has golden.npz, else mid-grey")
    ap.add_argument("--out", type=Path, default=Path("build/board/flash.bin"))
    ap.add_argument("--program", action="store_true", help="write it to the board's flash")
    ap.add_argument("--pool", action="store_true",
                    help="the bitstream has the MAXPOOL unit (npu_board_top is built without it)")
    args = ap.parse_args()
    prog = compile_ir(*load_ir(args.model_dir))
    if not args.pool and any(i.op == Op.MAXPOOL for i in prog.instrs):
        raise SystemExit("the model uses MAXPOOL, which this bitstream does not have (HAS_POOL=0)")
    m = prog.meta["input"]
    if args.image:
        img = load_image(args.image, m["h"], m["w"], m["c"])
    elif (args.model_dir / "golden.npz").exists():
        img = np.load(args.model_dir / "golden.npz")["images"][0]
    else:
        print("no --image and no golden.npz: the demo image is mid-grey")
        img = np.full((m["h"], m["w"], m["c"]), 128, np.uint8)
    words = flash_words(prog, img)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(np.array(words, dtype="<u8").tobytes())
    print(f"{args.out}: {len(words) * 8 / 1024:.0f} KiB ({len(prog.ext_words())} external-store words, "
          f"{len(prog.instrs)} instructions) for flash offset {FLASH_BASE:#x}; "
          f"bring-up flash check: header {words[BOOT_W]:016x}, sum {image_sum(prog.ext_words()):08x}")
    if args.program:
        try:
            subprocess.run(["openFPGALoader", "-b", "tangnano20k", "--external-flash", "--verify", "-o",
                            str(FLASH_BASE), str(args.out)], check=True)
        except FileNotFoundError:
            raise SystemExit("openFPGALoader is not installed or not on PATH; flash.bin was written, "
                             "program it with openFPGALoader -b tangnano20k --external-flash "
                             f"-o {FLASH_BASE:#x} {args.out}")
        except subprocess.CalledProcessError as e:
            raise SystemExit(f"openFPGALoader failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()
