"""Build the MLPerf Tiny `vww01` accuracy set (1000 images: 500 person / 500 not).

Two sources:
  --official DIR   the extracted official `vw_coco2014_96` dataset (exact MLPerf images,
                   already 96x96). Recommended for reported accuracy numbers.
  (default)        rebuild from COCO val2017 originals downloaded image by image, resized
                   to 96x96 with PIL bilinear. Silabs' resize method is undocumented, so
                   pixels may differ slightly from the official set. Fine for bit-exact
                   NPU-vs-TFLite checks; accuracy is then only approximately official.

    python sw/model/make_vww01.py --out build/vww/vww01.npz [--official ~/datasets/vww/extracted]
Writes an .npz with images (uint8 [1000, 96, 96, 3]), labels (1 = person), names, source.
"""

import argparse
import csv
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = Path(__file__).resolve().parents[2] / "models" / "vww" / "y_labels.csv"
COCO_URL = "http://images.cocodataset.org/val2017/{}.jpg"


def read_labels():
    with open(LABELS) as f:
        return [(r[0].removesuffix(".bin"), int(r[2])) for r in csv.reader(f)]


def fetch(image_id: str, cache: Path) -> bytes:
    path = cache / f"{image_id}.jpg"
    if path.exists():
        return path.read_bytes()
    for attempt in range(6):
        try:
            with urllib.request.urlopen(COCO_URL.format(image_id), timeout=60) as r:
                data = r.read()
            path.write_bytes(data)
            return data
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"failed to download {image_id}")


def from_coco(entries, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        blobs = list(pool.map(lambda e: fetch(e[0], cache), entries))
    imgs = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((96, 96), Image.BILINEAR))
            for b in blobs]
    return np.stack(imgs).astype(np.uint8)


def from_official(entries, root: Path):
    index = {p.stem.split("_")[-1]: p for p in root.rglob("*.jpg")}
    imgs = []
    for image_id, _ in entries:
        img = Image.open(index[image_id]).convert("RGB")
        if img.size != (96, 96):
            raise ValueError(f"{image_id}: expected 96x96, got {img.size}")
        imgs.append(np.asarray(img))
    return np.stack(imgs).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--official", type=Path, help="extracted vw_coco2014_96 directory")
    ap.add_argument("--cache", type=Path, default=Path.home() / "datasets" / "vww" / "coco_val2017")
    args = ap.parse_args()

    entries = read_labels()
    if args.official:
        images, source = from_official(entries, args.official), "official vw_coco2014_96"
    else:
        images, source = from_coco(entries, args.cache), "COCO val2017 + PIL bilinear (approximate)"
    labels = np.array([lab for _, lab in entries], np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, images=images, labels=labels,
                        names=np.array([n for n, _ in entries]), source=source)
    print(f"{len(labels)} images ({labels.sum()} person) from {source} -> {args.out}")


if __name__ == "__main__":
    main()
