import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gen_dialogs import stock_lines
from writer import WRITER, q4

HEAD = "Text chat between a teenager (F) and a savage roast bot (B).\n"
BATCH = 4


def line_scores(tok, net, convs):
    out = []
    for b in range(0, len(convs), BATCH):
        part = convs[b:b + BATCH]
        seqs, spans = [], []
        for c in part:
            ids, sp = tok(HEAD, add_special_tokens=False).input_ids, []
            for k, t in c:
                ids += tok(("B:" if k else "F:"), add_special_tokens=False).input_ids
                r = tok(" " + t + "\n", add_special_tokens=False).input_ids
                if k:
                    sp.append((len(ids), len(ids) + len(r)))
                ids += r
            seqs.append(ids[:1024])
            spans.append([(a, z) for a, z in sp if z <= 1024])
        L = max(map(len, seqs))
        x = torch.tensor([s + [tok.pad_token_id or 0] * (L - len(s)) for s in seqs], device="cuda:0")
        att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda:0")
        with torch.no_grad():
            lg = net(input_ids=x, attention_mask=att).logits
            for k, sp in enumerate(spans):
                lp = lg[k, :-1].float().log_softmax(-1)
                tgt = lp.gather(-1, x[k, 1:, None])[:, 0]
                out.append([tgt[a - 1:z - 1].mean().item() for a, z in sp])
            del lg, lp
        if b % (BATCH * 200) == 0:
            print(f"  {b}/{len(convs)}", flush=True)
    return out


def conv_score(s):
    return (sum(s) + min(s)) / (len(s) + 1) if s else -99.0


def main():
    drop = float(next((a for a in sys.argv[1:] if "=" not in a), 0.15))
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tok, net = q4(WRITER)
    tok.padding_side = "right"
    r = random.Random(5)
    for name in opt.get("src", "rb,rb_arg").split(","):
        d = ROOT / "data" / "people" / name
        convs, seen = [], set()
        for line in open(d / "partial.jsonl"):
            c = json.loads(line)["turns"]
            key = json.dumps(c)
            if stock_lines(c) <= 1 and key not in seen and any(k for k, _ in c):
                seen.add(key)
                convs.append(c)
        print(f"{name}: {len(convs)} conversations, scoring", flush=True)
        scores = line_scores(tok, net, convs)
        test = r.sample(range(len(convs)), min(300, len(convs)))
        pool = [t for i in test for k, t in convs[i] if k]
        r.shuffle(pool)
        it = iter(pool)
        shuf = line_scores(tok, net, [[[k, next(it) if k else t] for k, t in convs[i]] for i in test])
        real = [v for i in test for v in scores[i]]
        fake = [v for s in shuf for v in s]
        print(f"{name}: bot lines in their own conversation {sum(real) / len(real):.3f}, shuffled onto others "
              f"{sum(fake) / len(fake):.3f} nats/token; real line beats its shuffled copy in "
              f"{sum(a > b for a, b in zip(real, fake)) / len(real):.0%}", flush=True)
        cs = [conv_score(s) for s in scores]
        cut = sorted(cs)[int(len(cs) * drop)]
        order = sorted(range(len(convs)), key=lambda i: cs[i])
        for tag, idx in (("worst", order[:4]), ("cutoff", order[int(len(cs) * drop):][:2]), ("best", order[-2:])):
            for i in idx:
                print(f"--- {tag} {cs[i]:.2f}\n" + "\n".join(f"{'FB'[k]}: {t}" for k, t in convs[i]))
        kept = [c for c, s in zip(convs, cs) if s >= cut]
        r.shuffle(kept)
        k = max(1, len(kept) // 12)
        for split, part in (("val", kept[:k]), ("train", kept[k:])):
            (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": t}, ensure_ascii=False) + "\n"
                                                      for t in part))
        (d / "sense_scores.json").write_text(json.dumps(cs))
        print(f"{name}: kept {len(kept)} of {len(convs)} (score >= {cut:.2f}) -> {d}/train,val", flush=True)


if __name__ == "__main__":
    main()
