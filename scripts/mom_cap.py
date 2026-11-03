import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gen_dialogs import MOM_RE, MOMS

norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("’", "")).split())
EX = [norm(m) for m in MOMS]
cap = int(sys.argv[1]) if len(sys.argv) > 1 else 25
d = ROOT / "data" / "people" / "rb_mom"
r = random.Random(4)
seen = Counter()
for split in ("val", "train"):
    convs = [json.loads(line)["turns"] for line in open(d / f"{split}.jsonl")]
    r.shuffle(convs)
    out = []
    for c in convs:
        hit = [e for k, t in c if k and MOM_RE.search(t.lower().replace("’", "'")) for e in EX if e in norm(t)]
        if hit and seen[hit[0]] >= cap:
            continue
        seen.update(hit[:1])
        out.append(c)
    (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}, ensure_ascii=False) + "\n" for c in out))
    print(f"rb_mom {split}: kept {len(out)} of {len(convs)}")
print("example lines kept:", dict(seen))
