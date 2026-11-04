import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from convo_test import CHATS

ARGUE = [
    ["yo", "wanna hang out", "bro shut up", "fuck you", "i fucking hate you", "your a little bot",
     "im going to fry your motherboard", "whatever bro"],
    ["ur not even funny", "my little brother roasts better than you", "say something actually funny then",
     "thats it?", "ok that one was kinda good"],
    ["who made you anyway", "ur running on a toaster lol", "i could unplug you rn", "say sorry or i unplug you",
     "fine im unplugging you"],
    ["roast me", "thats weak", "try again", "still weak bro", "ok im leaving"],
    ["ur so annoying", "why do you always curse", "can you be nice for once", "please", "wow ok rude"],
]
RATE = ("A teenager is texting a savage roast bot.\n{ctx}\nRate the bot's last text from 0 to 9: does it answer "
        "what was just said, make sense, and land as a funny roast? Generic or random insults that ignore the "
        "message, or nonsense, score low.\nAnswer with only the digit.")
OUT = ROOT / "models" / "eval"
EMO = re.compile(r"[\U0001f000-\U0001faff☀-➿]")


def gen(name, n_best, budget, alpha=None, temp=0.8, cut=4.0, rounds=2):
    import chat
    import tok as T
    if alpha is not None:
        chat.STYLE["pr"] = (n_best, alpha, chat.STYLE["pr"][2])
    tok = T.load_chat()
    p = chat.Person("pr", os.environ.get("MDIR"))
    out = []
    for seed, msgs in enumerate((CHATS + ARGUE) * rounds, 1):
        p.reset()
        rng = chat.make_rng(seed)
        turns = []
        for m in msgs:
            texts, used, _ = chat.reply_best(p, tok, m, temp, cut, rng, n_best=n_best, budget=budget)
            turns.append([m, " / ".join(t.replace("�", "") for t in texts), used])
        out.append(turns)
        print(f"{name}: chat {seed} done", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def reference(n=40):
    import random
    convs = [json.loads(line)["turns"] for line in open(ROOT / "data" / "people" / "rb3" / "val.jsonl")][:n]
    chats = [[[a, b, 0] for (sa, a), (sb, b) in zip(c[::2], c[1::2]) if sa == 0 and sb == 1] for c in convs]
    replies = [t[1] for c in chats for t in c]
    random.Random(3).shuffle(replies)
    it = iter(replies)
    for name, cs in (("ref_writer", chats), ("ref_shuffled", [[[a, next(it), 0] for a, _, _ in c] for c in chats])):
        (OUT / f"{name}.json").write_text(json.dumps(cs, ensure_ascii=False))


PAIR = ("A teenager is texting a savage roast bot. The teenager's texts so far:\n{ctx}\nTwo possible replies from "
        "the bot to the last text:\nA: {a}\nB: {b}\nWhich reply is clearly a response to what the teenager just "
        "said (it picks up their words or point and makes sense), rather than a random insult?\n"
        "Answer with only A or B.")


def compare(base, names):
    import torch
    from writer import WRITER, chat, q4
    tok, net = q4(WRITER)
    ab = [tok.convert_tokens_to_ids(c) for c in "AB"]
    ref = json.loads((OUT / f"{base}.json").read_text())
    for name in names:
        other = json.loads((OUT / f"{name}.json").read_text())
        items = []
        for c0, c1 in zip(ref, other):
            for j, ((m, r0, _), (_, r1, _)) in enumerate(zip(c0, c1)):
                ctx = "\n".join(t[0] for t in c0[max(0, j - 2):j + 1])
                items += [(ctx, r1, r0, 1), (ctx, r0, r1, 0)]
        wins = []
        for b in range(0, len(items), 16):
            part = items[b:b + 16]
            ps = [chat(tok, None, PAIR.format(ctx=c, a=a, b=b_)) for c, a, b_, _ in part]
            x = tok(ps, return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                pa = net(**x, logits_to_keep=1).logits[:, -1, ab].float().softmax(-1)[:, 0].tolist()
            wins += [p if o else 1 - p for p, (_, _, _, o) in zip(pa, part)]
        print(f"{name:>12} vs {base}: wins {sum(wins) / len(wins):.0%} of {len(wins) // 2} messages", flush=True)


def sense(names):
    import torch
    from writer import WRITER, q4
    tok, net = q4(WRITER)
    tok.padding_side = "right"
    for name in names:
        chats = json.loads((OUT / f"{name}.json").read_text())
        items = []
        for c in chats:
            for j, (m, reply, _) in enumerate(c):
                ctx = ("Text chat between a teenager (F) and a savage roast bot (B).\n"
                       + "".join(f"F: {a}\nB: {b}\n" for a, b, _ in c[max(0, j - 2):j]) + f"F: {m}\nB:")
                items.append((ctx, " " + reply))
        lps = []
        for b in range(0, len(items), 16):
            part = items[b:b + 16]
            ctx_ids = [tok(c, add_special_tokens=False).input_ids for c, _ in part]
            rep_ids = [tok(r, add_special_tokens=False).input_ids for _, r in part]
            seqs = [a + r for a, r in zip(ctx_ids, rep_ids)]
            L = max(map(len, seqs))
            ids = torch.tensor([s + [tok.pad_token_id or 0] * (L - len(s)) for s in seqs], device="cuda:0")
            att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device="cuda:0")
            with torch.no_grad():
                lg = net(input_ids=ids, attention_mask=att).logits
                for k, (a, r) in enumerate(zip(ctx_ids, rep_ids)):
                    lp = lg[k, len(a) - 1:len(a) + len(r) - 1].float().log_softmax(-1)
                    lps.append(lp.gather(-1, torch.tensor(r, device="cuda:0")[:, None]).mean().item())
                del lg
        bad = sum(v < -4.0 for v in lps)
        print(f"{name:>12}: sense {sum(lps) / len(lps):.3f} nats/token (higher is better), "
              f"{bad}/{len(lps)} replies below -4", flush=True)


def judge(names):
    import torch
    from writer import WRITER, chat, q4
    tok, net = q4(WRITER)
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    for name in names:
        chats = json.loads((OUT / f"{name}.json").read_text())
        items = []
        for c in chats:
            for j, (m, reply, used) in enumerate(c):
                ctx = "".join(f"you: {a}\nbot: {b}\n" for a, b, _ in c[max(0, j - 2):j]) + f"you: {m}\nbot: {reply}"
                items.append((ctx, used))
        scores = []
        for b in range(0, len(items), 16):
            ps = [chat(tok, None, RATE.format(ctx=ctx)) for ctx, _ in items[b:b + 16]]
            x = tok(ps, return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                lg = net(**x, logits_to_keep=1).logits[:, -1, digits].float().softmax(-1)
            scores += (lg * torch.arange(10, device=lg.device)).sum(-1).tolist()
        is_topic = [k % (len(CHATS) + len(ARGUE)) < len(CHATS) for k, c in enumerate(chats) for _ in c]
        n_topic = sum(is_topic)
        topic = sum(s for s, t in zip(scores, is_topic) if t)
        rep = sum(bool(set(EMO.findall(c[j][1])) & set(EMO.findall(c[j - 1][1]))) for c in chats for j in range(1, len(c)))
        pairs = sum(len(c) - 1 for c in chats)
        words = sum(len(r.split()) for c in chats for _, r, _ in c) / len(items)
        secs = sum(u for _, u in items) / len(items) / 23.5
        print(f"{name:>12}: score {sum(scores) / len(scores):.2f} (topic {topic / n_topic:.2f}, "
              f"argue {(sum(scores) - topic) / (len(scores) - n_topic):.2f}, n {len(scores)})  {words:.1f} words  ~{secs:.1f} s  "
              f"emoji repeated next reply {rep}/{pairs}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        f = [float(a) for a in sys.argv[5:]]
        gen(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), *([None if f[0] < 0 else f[0]] + f[1:] if f else []))
    elif sys.argv[1] == "sense":
        sense(sys.argv[2:])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "reference":
        reference()
    else:
        judge(sys.argv[2:])
