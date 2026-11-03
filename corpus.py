import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "corpus"
MAX_CHARS = 280
URL = re.compile(r"https?://|www\.|\]\(|&gt;|&amp;|/u/|/r/|\bu/|\br/", re.I)


def clean(t):
    t = re.sub(r"\s+", " ", t.replace("​", "")).strip()
    return t if 0 < len(t) <= MAX_CHARS else None


def textify(t, rng):
    t = t.lower()
    if rng.random() < 0.8:
        t = re.sub(r"\.$", "", t)
    if rng.random() < 0.5:
        t = t.replace("'", "")
    if rng.random() < 0.3:
        t = t.replace(", ", " ")
    return t


def soda(path, rng, styled=0.5):
    for b in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["dialogue", "speakers"]):
        for d in b.to_pylist():
            names = list(dict.fromkeys(d["speakers"]))
            if len(names) != 2:
                continue
            sty = rng.random() < styled
            turns = []
            for s, t in zip(d["speakers"], d["dialogue"]):
                t = clean(t)
                if t is None:
                    turns = None
                    break
                turns.append([names.index(s), textify(t, rng) if sty else t])
            if turns and len(turns) >= 2:
                yield {"src": "soda", "turns": turns}


def reddit_chains(path, min_len=2, max_len=8):
    cols = ["body", "id", "parent_id", "author"]
    t = pq.read_table(path, columns=cols).to_pydict()
    node, parent = {}, {}
    kids = defaultdict(list)
    for body, cid, par, au in zip(t["body"], t["id"], t["parent_id"], t["author"]):
        if au in ("[deleted]", "AutoModerator") or not body or body in ("[deleted]", "[removed]") or URL.search(body):
            continue
        b = clean(body)
        if b is None:
            continue
        node[cid] = (b, au)
        if par.startswith("t1_"):
            kids[par[3:]].append(cid)
            parent[cid] = par[3:]
    for c in node:
        if parent.get(c) in node:
            continue
        chain, cur = [c], c
        while len(chain) < max_len:
            nxt = [k for k in kids.get(cur, []) if k in node]
            if not nxt:
                break
            cur = nxt[0]
            chain.append(cur)
        authors = [node[x][1] for x in chain]
        a, b = authors[0], authors[1] if len(authors) > 1 else None
        n = 1
        while n < len(chain) and authors[n] == (b if n % 2 else a) and b != a:
            n += 1
        if n >= min_len:
            yield {"src": "reddit", "turns": [[i % 2, node[x][0]] for i, x in enumerate(chain[:n])]}


def main():
    rng = random.Random(0)
    OUT.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with open(OUT / "train.jsonl", "w") as tr, open(OUT / "val.jsonl", "w") as va:
        def put(d):
            f = va if rng.random() < 0.01 else tr
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            stats[d["src"] + "_dialogues"] += 1
            stats[d["src"] + "_chars"] += sum(len(t) for _, t in d["turns"])

        for p in [RAW / "soda" / "train.parquet", RAW / "soda" / "valid.parquet"]:
            for d in soda(p, rng):
                put(d)
        print(dict(stats), flush=True)
        for p in sorted(glob.glob(str(RAW / "reddit" / "*.parquet"))):
            for d in reddit_chains(p):
                put(d)
            print(Path(p).name, dict(stats), flush=True)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
