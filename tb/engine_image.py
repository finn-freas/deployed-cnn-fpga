import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import board as B
import pack as P

HEX = ROOT / "build" / "sim" / "engine_mem.hex"
MODEL = Path(os.environ.get("FB_MODEL", ROOT / "models" / "pr_board.pkl"))
KEY = MODEL.name.split("_")[0]


def beat_index(a):
    b = a >> 2
    return (b & 3) << 19 | (b >> 7) << 8 | ((b >> 2) & 31) << 3 | (a & 3) << 1


def model(path=MODEL):
    p = pickle.loads(Path(path).read_bytes())
    p["score_w"] = B.score_table(Path(path).parent, Path(path).name.split("_")[0], p["cfg"].vocab_size)
    if not p.get("kv8"):
        B.kv8_calibrate(p, [np.random.default_rng(0).integers(4, p["cfg"].vocab_size, 64)])
    return p


def write(path=MODEL, out=HEX):
    img, _ = P.build(model(path))
    w = np.array(img.words, np.uint64)
    mem = np.zeros(1 << 21, np.uint32)
    idx = beat_index(np.arange(len(w), dtype=np.int64))
    mem[idx] = (w & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    mem[idx + 1] = (w >> np.uint64(32)).astype(np.uint32)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(f"{v:08x}" for v in mem.tolist()) + "\n")


if __name__ == "__main__":
    write()
