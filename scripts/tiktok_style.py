import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from writer import GROUPS, NEVER, WRITER, chat, q4

STY = re.compile(r"💀|😭|\bbro\b|ain.?t no way|who let|\bgiving\b|\bnot (the|you|him|her)\b|\bnah\b|be fr|\bratio\b|"
                 r"main character|the audacity|\bhe thinks\b|\bshe thinks\b|\bgng\b", re.I)
RATE = ("Is this TikTok comment a funny roast or teasing joke, the kind that gets the most likes? Score 0 to 9. "
        "9 = hilarious roast; 5 = mildly funny tease; 0 = not a joke (a question, praise, a product comment).\n"
        "Comment: {c}\nAnswer with only one digit.")
TOP = 500


def main():
    from datasets import load_dataset
    ds = load_dataset("ahnnqoc/tiktok-product-video-comments", split="train").select_columns(["text"])
    pool = set()
    for batch in ds.iter(batch_size=20000):
        pool |= {c.strip() for c in batch["text"] if isinstance(c, str)
                 and re.fullmatch(r"[\x00-\x7f💀😭🙏😂🤣😩😹🫠🥀]+", c.strip()) and 4 <= len(c.split()) <= 20
                 and STY.search(c) and not NEVER.search(c) and not GROUPS.search(c) and "@" not in c}
    pool = sorted(pool)
    del ds
    print(f"{len(pool)} candidates", flush=True)
    tok, net = q4(WRITER)
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    score = []
    for b in range(0, len(pool), 16):
        part = pool[b:b + 16]
        x = tok([chat(tok, None, RATE.format(c=c)) for c in part], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            p = net(**x, logits_to_keep=1).logits[:, -1, digits].float().softmax(-1)
        score += (p * torch.arange(10, device=p.device)).sum(-1).tolist()
        if b // 16 % 100 == 0:
            print(f"  scored {b + len(part)}/{len(pool)}", flush=True)
    best = [c for _, c in sorted(zip(score, pool), reverse=True)[:TOP]]
    out = ROOT / "data" / "style" / "tiktok_roasts.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(best, ensure_ascii=False, indent=0))
    print(f"kept {len(best)} -> {out}; top:")
    for c in best[:25]:
        print("  ", c)


if __name__ == "__main__":
    main()
