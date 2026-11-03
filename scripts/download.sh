#!/bin/bash
set -u
cd "$(dirname "$0")/.."
mkdir -p data/raw/soda data/raw/reddit

get() {
  local url="$1" out="$2"
  for try in 1 2 3 4 5; do
    curl -s -4 -L -C - --retry 3 -m 3600 -o "$out.part" "$url" && mv "$out.part" "$out" && { echo "ok $out $(du -h "$out" | cut -f1)"; return 0; }
    echo "retry $try $out"
  done
  echo "FAILED $out"
}

HF=https://huggingface.co/datasets
[ -f data/raw/soda/train.parquet ] || get "$HF/allenai/soda/resolve/main/train.parquet" data/raw/soda/train.parquet
[ -f data/raw/soda/valid.parquet ] || get "$HF/allenai/soda/resolve/main/valid.parquet" data/raw/soda/valid.parquet

python3 - <<'EOF' > data/raw/reddit/files.txt
import json, urllib.request
t = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/HuggingFaceGECLM/REDDIT_comments/tree/main/data", timeout=60))
want = {"socialskills": 9, "sports": 9, "Showerthoughts": 2, "tifu": 2, "gaming": 2,
        "relationship_advice": 1, "mildlyinteresting": 1, "Games": 1, "Fitness": 1}
for f in sorted(t, key=lambda f: f["path"]):
    name = f["path"].split("/")[-1]
    sub, shard = name.split("-0")[0], int(name.split("-")[1].split("-of")[0]) if "-of-" in name else 0
    if sub in want and shard < want[sub]:
        print(f["path"])
EOF
while read -r p; do
  out="data/raw/reddit/$(basename "$p")"
  [ -f "$out" ] || get "$HF/HuggingFaceGECLM/REDDIT_comments/resolve/main/$p" "$out"
done < data/raw/reddit/files.txt
echo DONE
du -sh data/raw/*
