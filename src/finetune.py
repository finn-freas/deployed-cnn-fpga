import argparse
import json
import re
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import board as B
import tok as T
from train import Net, Stream, chat_samples

ROOT = Path(__file__).resolve().parent.parent


def encode_sessions(tok, path):
    out = []
    for line in open(path):
        turns = json.loads(line)["turns"]
        enc = tok.encode_batch([t for _, t in turns], add_special_tokens=False)
        ids, lab = [T.BOS], [False]
        for (s, _), e in zip(turns, enc):
            ids.append(T.HIM if s == 1 else T.YOU)
            lab.append(True)
            ids += e.ids
            lab += [s == 1] * len(e.ids)
        ids.append(T.YOU)
        lab.append(True)
        out.append((np.array(ids, np.int64), np.array(lab, bool)))
    return out


def load_kd(tok, path, sess):
    import pickle
    d = pickle.loads(Path(path).read_bytes())
    fold = {}
    for i in range(4, T.VOCAB):
        s = tok.decode([i])
        if s == "\n":
            fold[i] = T.YOU
        elif s != s.lower():
            if s.startswith(" "):
                e = tok.encode(s.lower().lstrip(" "), add_special_tokens=False).ids
            else:
                e = [tok.token_to_id(s.lower())]
            if len(e) == 1 and e[0] is not None:
                fold[i] = e[0]
    out = []
    for (ids, _), (pos, idx, prob) in zip(sess, d["rows"]):
        ki = np.full((len(ids), d["K"]), -1, np.int16)
        kp = np.zeros((len(ids), d["K"]), np.float32)
        for p_, i_, q_ in zip(pos, idx, prob):
            acc = {}
            for a, b in zip(i_.tolist(), q_.tolist()):
                if a >= 0:
                    a = fold.get(a, a)
                    acc[a] = acc.get(a, 0.0) + b
            for k, (a, b) in enumerate(sorted(acc.items(), key=lambda x: -x[1])):
                ki[p_, k], kp[p_, k] = a, b
        out.append((ki, kp))
    return out


class People:
    def __init__(self, sess, seq, seed, kd=None):
        self.s, self.seq, self.kd = sess, seq, kd
        self.rng = np.random.default_rng(seed)
        w = np.array([len(i) for i, _ in sess], float)
        self.p = w / w.sum()

    def batch(self, n):
        x = np.full((n, self.seq), T.PAD, np.int64)
        y = np.full((n, self.seq), -1, np.int64)
        K = self.kd[0][0].shape[1] if self.kd else 1
        ki = np.full((n, self.seq, K), -1, np.int64)
        kp = np.zeros((n, self.seq, K), np.float32)
        for r in range(n):
            j = self.rng.choice(len(self.s), p=self.p)
            ids, lab = self.s[j]
            st = int(self.rng.integers(0, max(1, len(ids) - self.seq)))
            seg, sl = ids[st:st + self.seq + 1], lab[st:st + self.seq + 1]
            L = len(seg) - 1
            x[r, :L] = seg[:-1]
            y[r, :L] = np.where(sl[1:], seg[1:], -1)
            if self.kd:
                ki[r, :L] = self.kd[j][0][st + 1:st + 1 + L]
                kp[r, :L] = self.kd[j][1][st + 1:st + 1 + L]
        if self.kd:
            return (torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(), torch.from_numpy(ki).cuda(),
                    torch.from_numpy(kp).cuda())
        return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


@torch.no_grad()
def eval_people(net, sess, seq):
    net.eval()
    tot, n = 0.0, 0
    for ids, lab in sess:
        for st in range(0, max(1, len(ids) - 1), seq):
            seg = torch.from_numpy(ids[st:st + seq + 1]).cuda()[None]
            m = torch.from_numpy(lab[st + 1:st + seq + 1]).cuda()
            if seg.shape[1] < 2 or not m.any():
                continue
            with torch.autocast("cuda", torch.bfloat16):
                lg = net(seg[:, :-1])[0].float()
            nll = F.cross_entropy(lg, seg[0, 1:], reduction="none")
            tot += float(nll[m].sum())
            n += int(m.sum())
    net.train()
    return tot / max(n, 1), n


def int_check(p, net, sess, max_tokens=1500):
    im = B.IntModel(p)
    f_logit = p["fmt"]["logit"]
    nll_i, nll_f, agree, n = 0.0, 0.0, 0, 0
    net.eval()
    for ids, lab in sess:
        if n >= max_tokens:
            break
        ids, lab = ids[:p["cfg"].seq_len], lab[:p["cfg"].seq_len]
        with torch.no_grad():
            fl = net(torch.from_numpy(ids[None]).cuda())[0].float().cpu().numpy()
        im.reset()
        for pos in range(len(ids) - 1):
            need = lab[pos + 1]
            li = im.forward(int(ids[pos]), pos, logits=need)
            if not need:
                continue
            li = li.astype(np.float64) / 2 ** f_logit
            t = ids[pos + 1]
            nll_i += float(np.logaddexp.reduce(li) - li[t])
            nll_f += float(np.logaddexp.reduce(fl[pos]) - fl[pos, t])
            agree += int(np.argmax(li) == np.argmax(fl[pos]))
            n += 1
    return {"tokens": n, "nll_int": nll_i / n, "nll_float": nll_f / n, "top1_agree": agree / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("person")
    ap.add_argument("--init", default=str(ROOT / "models" / "pre_b.pt"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mix", type=float, default=0.15, help="share of rows from general chat")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--until-overfit", action="store_true",
                    help="past --steps keep going at the lowest learning rate until --patience stops it")
    ap.add_argument("--start-step", type=int, default=0, help="continue a run: --init its pr.pt, same --steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--synth", default="", help="their synthetic sessions (data/synth/<person>.jsonl) mixed in")
    ap.add_argument("--synth-frac", type=float, default=0.25, help="share of each batch from --synth")
    ap.add_argument("--boost", default="", help="regex: their real messages that match are drilled more")
    ap.add_argument("--boost-frac", type=float, default=0.2, help="share of each batch from --boost snippets")
    ap.add_argument("--gen", default=str(T.TOK / "train.bin"), help="general chat stream for --mix")
    ap.add_argument("--kd", type=float, default=0.0,
                    help="weight of the writer's next-token guesses (data/people/<person>/kd_train.pkl)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.init, map_location="cuda")
    c = dict(ck["config"], dropout=a.dropout)
    net = Net(c).cuda()
    net.load_state_dict(ck["state"])
    net.set_qat(True)
    tok = T.load_chat()
    slots = T.TOK / "slots.json"
    if T.CHAT_TOK.exists() and slots.exists() and not a.start_step:
        with torch.no_grad():
            for tid, _, old in json.loads(slots.read_text()):
                net.emb.weight[tid] = net.emb.weight[old].mean(0)
    d = ROOT / "data" / "people" / a.person
    tr, va = encode_sessions(tok, d / "train.jsonl"), encode_sessions(tok, d / "val.jsonl")
    print(f"{a.person}: train {sum(len(i) for i, _ in tr)} tokens ({sum(l.sum() for _, l in tr)} labelled), "
          f"val {sum(len(i) for i, _ in va)}", flush=True)
    kd = load_kd(tok, d / "kd_train.pkl", tr) if a.kd else None
    if kd:
        print(f"kd: {sum(int((k[:, 0] >= 0).sum()) for k, _ in kd)} positions with a writer target, weight {a.kd}",
              flush=True)
    data = People(tr, c["seq"], a.seed, kd)
    gen = Stream(a.gen, c["seq"], a.seed)
    n_gen = int(round(a.batch * a.mix))
    syn = None
    if a.synth:
        ss = encode_sessions(tok, a.synth)
        ss = [(np.concatenate([i for i, _ in ss[k:k + 12]]), np.concatenate([l for _, l in ss[k:k + 12]]))
              for k in range(0, len(ss), 12)]
        syn = People(ss, c["seq"], a.seed + 3)
    n_syn = int(round(a.batch * a.synth_frac)) if syn else 0
    bst = None
    if a.boost:
        rx, snip = re.compile(a.boost, re.I), []
        for line in open(d / "train.jsonl"):
            turns = json.loads(line)["turns"]
            snip += [turns[max(0, j - 4):j + 1] for j, (s_, t) in enumerate(turns) if s_ == 1 and rx.search(t)]
        tmp = d / "boost.jsonl"
        tmp.write_text("".join(json.dumps({"turns": t}) + chr(10) for t in snip))
        bs = encode_sessions(tok, tmp)
        bs = [(np.concatenate([i for i, _ in bs[k:k + 12]]), np.concatenate([l for _, l in bs[k:k + 12]]))
              for k in range(0, len(bs), 12)]
        bst = People(bs, c["seq"], a.seed + 5)
        print(f"boost: {len(snip)} snippets matching {a.boost!r}", flush=True)
    n_bst = int(round(a.batch * a.boost_frac)) if bst else 0
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    best, best_state, bad, log = float("inf"), None, 0, []
    t0 = time.time()
    for step in range(a.start_step, a.steps * (4 if a.until_overfit else 1) + 1):
        lr = a.lr * min(1.0, (step - a.start_step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        if step % a.eval_every == 0:
            vl, nv = eval_people(net, va, c["seq"])
            log.append({"step": step, "val": vl})
            mark = ""
            if vl < best:
                best, bad, mark = vl, 0, " *"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            print(f"step {step} val {vl:.4f} ppl {math.exp(vl):.2f} ({nv} tok) {time.time() - t0:.0f}s{mark}", flush=True)
            if bad >= a.patience:
                break
        x, y, *kdb = data.batch(a.batch - n_gen - n_syn - n_bst)
        if n_bst:
            bx, by = bst.batch(n_bst)
            x, y = torch.cat([x, bx]), torch.cat([y, by])
        if n_syn:
            sx, sy = syn.batch(n_syn)
            x, y = torch.cat([x, sx]), torch.cat([y, sy])
        if n_gen:
            gx, gy = gen.batch(n_gen)
            x, y = torch.cat([x, gx]), torch.cat([y, gy])
        with torch.autocast("cuda", torch.bfloat16):
            lg = net(x).float()
        loss = F.cross_entropy(lg.flatten(0, 1), y.flatten(), ignore_index=-1)
        if kdb:
            ki, kp = kdb
            lp = lg[:ki.shape[0]].log_softmax(-1)
            has = ki[..., 0] >= 0
            if has.any():
                ce = -(kp * lp.gather(-1, ki.clamp(min=0))).sum(-1)
                loss = loss + a.kd * ce[has].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    net.load_state_dict(best_state)
    net.eval()
    torch.save({"config": c, "state": net.state_dict(), "qat": True}, out / f"{a.person}.pt")
    cal = [torch.from_numpy(i[:c["seq"]][None]).cuda() for i, _ in va[:64]]
    fmt = {k: 0 for k in B.FMT_KEYS}
    amax = {k: 0.0 for k in B.FMT_KEYS}
    for x in cal + [gen.batch(16)[0]]:
        _, am = B.calibrate(net, x)
        amax = {k: max(amax[k], am[k]) for k in amax}
    fmt = {k: B.frac_bits(v) for k, v in amax.items()}
    p = B.quantize(net, fmt)
    B.kv8_calibrate(p, [i[:c["seq"]] for i, _ in tr[:48]])
    chk = int_check(p, net, va)
    mm = B.memory_map(p)
    with open(out / f"{a.person}_board.pkl", "wb") as f:
        pickle.dump(p, f)
    samples = chat_samples(net, tok, ["yo what are you up to", "wanna get food later", "lol thats crazy",
                                      "did you finish the homework", "im so tired"])
    rep = {"person": a.person, "best_val_nll": best, "best_val_ppl": math.exp(best), "fmt": fmt, "amax": amax,
           "int_check": chk, "memory": mm, "log": log}
    (out / f"{a.person}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["best_val_ppl", "fmt", "int_check", "memory"]}, indent=1))
    for s in samples:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
