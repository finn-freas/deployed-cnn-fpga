import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MDIR = ROOT / "models"
sys.path.insert(0, str(ROOT))
import chat
import tok as T
from finetune import encode_sessions
from train import Net

CORR_RANGE = (0.25, 4.0)
SMOOTH = 1.0
BIAS_RANGE = (-2.0, 2.5)


@torch.no_grad()
def measure(key):
    ck = torch.load(MDIR / f"{key}.pt", map_location="cuda")
    net = Net(dict(ck["config"], dropout=0.0)).cuda().eval()
    net.load_state_dict(ck["state"])
    net.set_qat(True)
    mask = torch.tensor(chat.EMOJI, device="cuda")
    seq = ck["config"]["seq"]
    pred_tok = torch.zeros(T.VOCAB, device="cuda")
    real_tok = torch.zeros(T.VOCAB, device="cuda")
    for split in ["train", "val"]:
        for ids, lab in encode_sessions(T.load_chat(), ROOT / "data" / "people" / key / f"{split}.jsonl"):
            for s in range(0, len(ids) - 1, seq):
                x = torch.tensor(ids[s:s + seq + 1], device="cuda")
                if len(x) < 2:
                    continue
                m = torch.tensor(lab[s + 1:s + seq + 1], device="cuda") & ~mask[x[:-1]]
                if not m.any():
                    continue
                p = torch.softmax(net(x[None, :-1])[0].float(), -1)
                pred_tok += p[m].sum(0)
                y = x[1:][m]
                real_tok.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    pred_tok, real_tok = pred_tok.cpu() * mask.cpu(), real_tok.cpu() * mask.cpu()
    return pred_tok, real_tok


def mix_bias(pred_tok, real_tok):
    scale = real_tok.sum() / max(pred_tok.sum().item(), 1e-9)
    b = torch.log((real_tok + SMOOTH) / (pred_tok * scale + SMOOTH)).clamp(*BIAS_RANGE)
    return [[int(i), round(b[i].item(), 3)] for i in range(T.VOCAB) if chat.EMOJI[i]]


def real_rate(key):
    tok = T.load_chat()
    msgs = [t for split in ["train", "val"] for line in open(ROOT / "data" / "people" / key / f"{split}.jsonl")
            for sp, t in json.loads(line)["turns"] if sp == 1]
    return sum(any(chat.EMOJI[i] for i in e.ids) for e in tok.encode_batch(msgs, add_special_tokens=False)) / len(msgs)


TEXTS = ["how are you", "what are you doing", "did you eat yet", "wanna hang out later", "how was school",
         "are you home", "can you pick me up at 5", "did you see the game last night", "happy birthday!!",
         "i got the job!", "im so bored", "goodnight", "what should we eat tonight", "are you coming tonight?",
         "where are you", "lol that was so funny", "call me when you can", "did you finish your homework",
         "how was work", "i miss you", "what time is practice tomorrow", "can i borrow your charger",
         "my car broke down", "are you awake", "whats up", "do you want pizza or tacos", "i failed my test",
         "thanks for today", "wyd this weekend", "did you get my package", "how was your trip",
         "the movie was so good", "im running late", "are you mad at me", "what are you wearing to the party",
         "my phone is about to die", "guess what happened", "can you send me the pics", "is it cold outside",
         "love you"]


def cold_mass(key, tok, seed=1):
    p = chat.Person(key, MDIR)
    rng = chat.make_rng(seed)
    mass = []
    for q in TEXTS:
        p.reset()
        chat.reply(p, tok, q, 0.8, 4.0, rng)
        mass += p.bot_mass
    return sum(mass) / max(len(mass), 1)


def main():
    global MDIR
    args = sys.argv[1:]
    if args and args[0] == "--dir":
        MDIR, args = Path(args[1]), args[2:]
    for key in args or ["pr"]:
        pred_tok, real_tok = measure(key)
        pred, actual = pred_tok.sum().item(), real_tok.sum().item()
        corr = min(max(actual / max(pred, 1e-9), CORR_RANGE[0]), CORR_RANGE[1])
        rate = real_rate(key)
        bias = mix_bias(pred_tok, real_tok)
        (MDIR / f"{key}_emoji.json").write_text(json.dumps(
            {"corr": round(corr, 3), "rate": round(rate, 4), "pred": pred, "actual": actual, "bias": bias}))
        tok = T.load_chat()
        cold = cold_mass(key, tok)
        d = json.loads((MDIR / f"{key}_emoji.json").read_text())
        d["cold_mass"] = round(cold, 5)
        (MDIR / f"{key}_emoji.json").write_text(json.dumps(d))
        top = sorted(bias, key=lambda b: -b[1])[:6]
        print(f"{key}: model expects {pred:.0f} emoji starts, they made {actual:.0f}: correction x{corr:.2f}; "
              f"{rate:.1%} of their messages have an emoji; most boosted "
              + " ".join(f"{tok.decode([i])}{b:+.1f}" for i, b in top) + f"; fresh-chat mass {cold:.4f}", flush=True)


if __name__ == "__main__":
    main()
