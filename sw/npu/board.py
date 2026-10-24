"""Flash image for the board (rtl/board/npu_system.sv) and programming helper.

    python -m npu.board build/vww --image demo.png --out build/board/flash.bin [--program]
                                                                     (run from sw/)

Layout in 64-bit words from FLASH_BASE (0x400000): the external-store image (weights and
params) at word 0, then at BOOT_W a header, the external-store image's length, the
program (two 32-bit instruction beats per word) and a demo image (packed uint8 HWC bytes). --program writes it with openFPGALoader
next to the bitstream (which sits at the start of the same flash).
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np

from . import isa
from .compiler import compile_ir, load_ir

FLASH_BASE = 0x40_0000
BOOT_W = 0x1_0000
MAGIC = 0b101101


def boot_words(prog, image_u8: np.ndarray) -> list[int]:
    """Header (2 words) + program + demo image, as uint64 words."""
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
    """The whole user-data region from FLASH_BASE, as uint64 words."""
    ext = prog.ext_words()
    assert len(ext) <= BOOT_W, "external store image overlaps the boot block"
    return ext + [0] * (BOOT_W - len(ext)) + boot_words(prog, image_u8)


def load_image(path: Path, h: int, w: int, c: int) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.uint8).reshape(h, w, c)
    from PIL import Image
    im = Image.open(path).convert("RGB" if c == 3 else "L").resize((w, h))
    return np.asarray(im, np.uint8).reshape(h, w, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--image", type=Path, help="demo image (png/jpg/npy); default: golden image 0")
    ap.add_argument("--out", type=Path, default=Path("build/board/flash.bin"))
    ap.add_argument("--program", action="store_true", help="write it to the board's flash")
    args = ap.parse_args()
    prog = compile_ir(*load_ir(args.model_dir))
    m = prog.meta["input"]
    if args.image:
        img = load_image(args.image, m["h"], m["w"], m["c"])
    else:
        img = np.load(args.model_dir / "golden.npz")["images"][0]
    words = flash_words(prog, img)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(np.array(words, dtype="<u8").tobytes())
    print(f"{args.out}: {len(words) * 8 / 1024:.0f} KiB ({len(prog.ext_words())} external-store words, "
          f"{len(prog.instrs)} instructions) for flash offset {FLASH_BASE:#x}")
    if args.program:
        subprocess.run(["openFPGALoader", "-b", "tangnano20k", "--external-flash", "-o",
                        str(FLASH_BASE), str(args.out)], check=True)


if __name__ == "__main__":
    main()
