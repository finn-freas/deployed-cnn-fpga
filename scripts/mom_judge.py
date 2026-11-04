import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from chat import AGGRO
from gen_dialogs import MOM_RE
from writer import WRITER, chat, q4

ASK = ("A teenager (F) is texting a savage roast bot (B).\n{ctx}\nB's last text has a 'your mom' joke. Does it make "
       "sense right there, as a cocky comeback to what F just said (F insulted, dissed or challenged B, and the joke "
       "answers that, like answering 'fuck you' with 'i had your mom in bed last night')? Answer N if it comes out of "
       "nowhere, ignores what F said, or doesn't make sense.\nAnswer with only Y or N.")
opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
drop = float(next((a for a in sys.argv[1:] if "=" not in a), 0.25))
d = ROOT / "data" / "people" / opt.get("dst", "rb5")
tok, net = q4(WRITER)
yn = [tok.convert_tokens_to_ids(c) for c in "YN"]
for split in ("train", "val"):
    convs = [json.loads(line)["turns"] for line in open(d / f"{split}.jsonl")]
    spots = [(i, j) for i, c in enumerate(convs) for j, (k, t) in enumerate(c)
             if k == 1 and j and MOM_RE.search(t.lower().replace("’", "'"))]
    py = []
    for b in range(0, len(spots), 16):
        ps = [chat(tok, None, ASK.format(ctx="\n".join(f"{'FB'[k]}: {t}" for k, t in convs[i][max(0, j - 2):j + 1])))
              for i, j in spots[b:b + 16]]
        x = tok(ps, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            py += net(**x, logits_to_keep=1).logits[:, -1, yn].float().softmax(-1)[:, 0].tolist()
    aggro = [bool(AGGRO.search(convs[i][j - 1][1].replace("’", "'"))) for i, j in spots]
    rated = sorted(p for p, a in zip(py, aggro) if a)
    cut = rated[int(len(rated) * drop)] if rated else 1.0
    bad = {}
    for (i, j), p, a in zip(spots, py, aggro):
        if not a or p < cut:
            bad[i] = min(bad.get(i, j), j)
    if split == "train":
        ranked = sorted((p, s) for p, s, a in zip(py, spots, aggro) if a)
        for tag, part in (("doesn't fit", ranked[:8]), ("fits", ranked[-8:])):
            for p, (i, j) in part:
                print(f"{tag} {p:.2f}  F: {convs[i][j - 1][1][:70]}  ->  B: {convs[i][j][1][:90]}")
    out = [c[:bad[i] - 1] if i in bad else c for i, c in enumerate(convs)]
    out = [c for c in out if sum(k == 1 for k, _ in c) >= 2]
    (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}, ensure_ascii=False) + "\n" for c in out))
    print(f"{d.name} {split}: {len(spots)} your-mom lines, {len(spots) - sum(aggro)} not after an insult, "
          f"{sum(a and p < cut for p, a in zip(py, aggro))} rated lowest (< {cut:.2f}); {len(bad)} conversations cut; "
          f"{len(out)} of {len(convs)} conversations kept", flush=True)
