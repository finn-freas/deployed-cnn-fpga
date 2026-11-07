import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USAGE = "usage: python scripts/copy_check.py source[,source...] [drop]"
norm = lambda s: " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("’", "'")).split())


def parse_args(argv):
    if not 1 <= len(argv) <= 2 or (len(argv) == 2 and argv[1] != "drop"):
        sys.exit(USAGE)
    sources = [s for s in argv[0].split(",") if s]
    if not sources:
        sys.exit(USAGE)
    style_file = ROOT / "data" / "style" / "tiktok_roasts.json"
    if not style_file.is_file():
        sys.exit(f"{style_file} not found")
    for src in sources:
        for split in ("train", "val"):
            f = ROOT / "data" / "people" / src / f"{split}.jsonl"
            if not f.is_file():
                sys.exit(f"{f} not found")
    return sources, len(argv) == 2, style_file


def main():
    sources, drop, style_file = parse_args(sys.argv[1:])
    style = {norm(x) for x in json.loads(style_file.read_text())}
    style = {s for s in style if len(s.split()) >= 4}

    def copied(t):
        n = norm(t)
        return any(s in n for s in style) if len(n.split()) >= 4 else False

    for src in sources:
        for split in ("train", "val"):
            f = ROOT / "data" / "people" / src / f"{split}.jsonl"
            convs = [json.loads(line)["turns"] for line in open(f)]
            hits = [(i, j) for i, c in enumerate(convs) for j, (k, t) in enumerate(c) if k == 1 and copied(t)]
            b = sum(k == 1 for c in convs for k, _ in c)
            print(f"{src} {split}: {len(hits)} of {b} bot lines copy a style example, in "
                  f"{len({i for i, _ in hits})} of {len(convs)} conversations")
            for i, j in hits[:6] if split == "train" else []:
                print("   ", convs[i][j - 1][1][:70], "->", convs[i][j][1][:90])
            if drop:
                first = {}
                for i, j in hits:
                    first.setdefault(i, j)
                out = [c[:first[i] - 1] if i in first else c for i, c in enumerate(convs)]
                out = [c for c in out if sum(k == 1 for k, _ in c) >= 2]
                f.write_text("".join(json.dumps({"turns": c}, ensure_ascii=False) + "\n" for c in out))
                print(f"   -> {len(out)} conversations kept")


if __name__ == "__main__":
    main()
