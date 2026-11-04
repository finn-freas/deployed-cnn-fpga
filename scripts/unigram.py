import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tok as T


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models"
    out.mkdir(parents=True, exist_ok=True)
    c = np.bincount(np.fromfile(T.TOK / "val.bin", np.uint16), minlength=T.VOCAB)[:T.VOCAB].astype(float)
    np.save(out / "generic_unigram.npy", (c / c.sum() + 1e-7) / (1 + 1e-7 * T.VOCAB))


if __name__ == "__main__":
    main()
