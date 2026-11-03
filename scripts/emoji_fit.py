import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from writer import WRITER, chat, q4

EMOJI = list("💀😭🙏🤣😂😩😤👀😈🤓😏🤨🙄😬👍😎😔🤦😳")
LETTERS = "ABCDEFGHIJKLMNOPQRST"
MAJORITY, CRY, DOUBLE = 0.55, 0.4, 0.35
RECENT, AVOID = 2, 3.0
ASK = ("Texts between two friends; B roasts F.\n{ctx}\nWhich emoji fits best at the end of B's last text, given "
       "what it says and how it lands?\n{opts}\nAnswer with only the letter.")
FACE = re.compile(r"facial|expression|\b(your|ur|his) face\b", re.I)
EMO = r"[\s.,☀-➿♀♂‍️\U0001f000-\U0001faff]+"
TAIL, HEAD = re.compile(EMO + "$"), re.compile("^" + EMO)
LOOK = re.compile(r"\b(you|u) look like\b", re.I)
BOARD = re.compile(r"tiny (computer )?board|piece of plastic|microcontroller|computer board|toaster|tin can|"
                   r"processing power|circuit|motherboard|calculator", re.I)
BOTTY = re.compile(r"\b(bot|robot|ai|machine|computer|unplug|plug|motherboard|toaster|program|code|chip|board|"
                   r"real|made you|calculator|phone)\b", re.I)
BALLSACK, BALL = {"rb": 20, "rb_arg": 10}, ["you look like my left ballsack", "you look like my left ballsack fatass",
                      "you look like my left ballsack 💀", "you look like my left ballsack 😭",
                      "you look like my left ballsack lil bitch", "bro you look like my left ballsack"]


def unprompted_board(c):
    for j, (k, t) in enumerate(c):
        if k == 1 and BOARD.search(t) and not (j and c[j - 1][0] == 0 and BOTTY.search(c[j - 1][1])):
            return j
    return None


def load(src, r):
    data = {s: [json.loads(line)["turns"] for line in open(src / f"{s}.jsonl")] for s in ("train", "val")}
    for s in data:
        n = len(data[s])
        data[s] = [c for c in data[s] if not any(k == 1 and FACE.search(t) for k, t in c)]
        cut = 0
        for i, c in enumerate(data[s]):
            j = unprompted_board(c)
            if j is not None:
                data[s][i] = c[:j - 1]
                cut += 1
        data[s] = [c for c in data[s] if sum(k == 1 for k, _ in c) >= 2]
        print(f"{src.name} {s}: {n - len(data[s])} conversations dropped (F's face, too short), {cut} cut before "
              f"unprompted board talk")
    looks = [(s, i, j) for s in ("train",) for i, c in enumerate(data[s]) for j, (k, t) in enumerate(c)
             if k == 1 and LOOK.search(t) and "ballsack" not in t]
    for s, i, j in r.sample(looks, min(BALLSACK.get(src.name, 20), len(looks))):
        data[s][i][j][1] = r.choice(BALL)
    return data


def score(data, cache, model):
    spots = [(s, i, j) for s in data for i, c in enumerate(data[s]) for j, (k, _) in enumerate(c) if k == 1]
    if cache.exists():
        scores = json.loads(cache.read_text())
        if len(scores) == len(spots):
            return spots, scores
    tok, net = model()
    ids = [tok.convert_tokens_to_ids(c) for c in LETTERS]
    opts = "\n".join(f"{L}. {e}" for L, e in zip(LETTERS, ["(no emoji)"] + EMOJI))
    scores = []
    for b in range(0, len(spots), 24):
        ps = []
        for s, i, j in spots[b:b + 24]:
            c = data[s][i]
            ctx = "\n".join(f"{'F' if k == 0 else 'B'}: {TAIL.sub('', t) if q == j else t}"
                            for q, (k, t) in enumerate(c[max(0, j - 2):j + 1], start=max(0, j - 2)))
            ps.append(chat(tok, None, ASK.format(ctx=ctx, opts=opts)))
        x = tok(ps, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            lp = net(**x, logits_to_keep=1).logits[:, -1, ids].float().log_softmax(-1)
        scores += lp.tolist()
        if b % 2400 == 0:
            print(f"  scored {b + len(ps)}/{len(spots)}", flush=True)
    cache.write_text(json.dumps(scores))
    return spots, scores


def main():
    nums = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    share = float(nums[0]) if nums else 0.4
    spread = float(nums[1]) if len(nums) > 1 else 1.0
    srcs = opt.get("src", "rb,rb_arg").split(",")
    dst = ROOT / "data" / "people" / opt.get("dst", "rb3")
    dst.mkdir(exist_ok=True)
    r = random.Random(11)
    loaded = []
    model = lambda: loaded[0] if loaded else loaded.append(q4(WRITER)) or loaded[0]
    parts = []
    for name in srcs:
        data = load(ROOT / "data" / "people" / name, r)
        spots, scores = score(data, dst / f"emoji_scores_{name}.json", model)
        parts.append((data, spots, scores))
    scores = [sc for _, _, s in parts for sc in s]

    none_lp = sorted(sc[0] for sc in scores)
    cut = none_lp[int(share * len(none_lp))]
    mean = [math.log(sum(math.exp(sc[k]) for sc in scores) / len(scores)) for k in range(1, len(LETTERS))]
    fit = lambda sc, bonus, cry, e: sc[e + 1] - spread * mean[e] + bonus * (e < 2) + cry * (e == 1)
    pick_of = lambda sc, bonus, cry: max(range(len(EMOJI)), key=lambda e: fit(sc, bonus, cry, e))
    picks = [sc for sc in scores if sc[0] < cut]
    bonus = cry = 0.0
    for _ in range(4):
        lo, hi = 0.0, 20.0
        for _ in range(25):
            mid = (lo + hi) / 2
            got = [pick_of(sc, mid, cry) < 2 for sc in picks]
            lo, hi = (mid, hi) if sum(got) / len(got) < MAJORITY else (lo, mid)
        bonus, lo, hi = hi, -10.0, 10.0
        for _ in range(25):
            mid = (lo + hi) / 2
            got = [pick_of(sc, bonus, mid) for sc in picks]
            lo, hi = (mid, hi) if got.count(1) / max(1, got.count(0) + got.count(1)) < CRY else (lo, mid)
        cry = hi
    print(f"bonus for 💀 😭: {bonus:.2f}, extra for 😭: {cry:.2f}")
    used, ends, shown = Counter(), Counter(), 0
    out = {"train": [], "val": []}
    for data, spots, sc_ in parts:
        seen = {}
        for (s, i, j), sc in zip(spots, sc_):
            t = data[s][i][j][1]
            if "ballsack" in t and TAIL.search(t):
                continue
            base = TAIL.sub("", t)
            inner = [q for q, x in enumerate(EMOJI) if x in base]
            recent = set(inner) | {q for prev in seen.get((s, i), [])[-RECENT:] for q in prev}
            if (sc[0] < cut or not base) and not (inner and HEAD.match(t) and r.random() < 0.5):
                fit_r = lambda q: fit(sc, bonus, cry, q) - AVOID * (q in recent)
                k = max(range(len(EMOJI)), key=fit_r)
                e = EMOJI[k]
                if r.random() < DOUBLE:
                    e += e if r.random() < 0.5 else EMOJI[max((q for q in range(len(EMOJI)) if q != k), key=fit_r)]
                seen.setdefault((s, i), []).append({EMOJI.index(x) for x in e} | set(inner))
                data[s][i][j][1] = f"{base} {e}" if base else e
                used.update(e)
                ends[len(e)] += 1
                if shown < 30 and r.random() < 0.03:
                    print(f"   {base[:80]!r} -> {e}")
                    shown += 1
            else:
                data[s][i][j][1] = base
                seen.setdefault((s, i), []).append(set(inner))
        for s in data:
            out[s] += data[s]
    tot, n_spots = sum(ends.values()), sum(len(p[1]) for p in parts)
    print(f"{tot}/{n_spots} B texts end in emoji ({ends[2]} with two); 💀+😭 = "
          f"{(used['💀'] + used['😭']) / sum(used.values()):.0%}")
    print("  ", " ".join(f"{e}{n}" for e, n in used.most_common()))
    for s in out:
        r.shuffle(out[s])
        (dst / f"{s}.jsonl").write_text("".join(json.dumps({"turns": c}, ensure_ascii=False) + "\n" for c in out[s]))
    print(f"-> {dst}: {len(out['train'])} train, {len(out['val'])} val conversations")


if __name__ == "__main__":
    main()
