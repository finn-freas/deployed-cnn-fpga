import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import board as B
import tok as T

ROOT = Path(__file__).resolve().parent.parent
SIZES = {
    "s": (192, 512, 6, 6, 2),
    "m": (256, 704, 8, 8, 2),
    "l": (320, 864, 10, 10, 2),
    "b": (256, 704, 10, 8, 2),
    "b12": (256, 704, 12, 8, 2),
    "b14": (256, 704, 14, 8, 2),
    "b16": (256, 704, 16, 8, 2),
}


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


def unit_rms(x):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).type_as(x)


def rope_tables(hs, seq):
    freq = 10000.0 ** (-torch.arange(0, hs, 2).float() / hs)
    ang = torch.arange(seq).float()[:, None] * freq[None, :]
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :x.shape[1], None, :], sin[None, :x.shape[1], None, :]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hs = c["heads"], c["kv_heads"], c["dim"] // c["heads"]
        d = c["dim"]
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, self.nkv * self.hs, bias=False)
        self.wv = nn.Linear(d, self.nkv * self.hs, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, c["hidden"], bias=False)
        self.w2 = nn.Linear(c["hidden"], d, bias=False)
        self.w3 = nn.Linear(d, c["hidden"], bias=False)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.drop = nn.Dropout(c["dropout"])
        self.qat = False

    def lin(self, layer, h, gain=None):
        w = layer.weight if gain is None else layer.weight * gain[None, :]
        return F.linear(h, B.fq4(w) if self.qat else w)

    def forward(self, x, cos, sin):
        Bs, L, _ = x.shape
        h = unit_rms(x)
        q = apply_rope(self.lin(self.wq, h, self.n1.weight).view(Bs, L, self.nh, self.hs), cos, sin)
        k = apply_rope(self.lin(self.wk, h, self.n1.weight).view(Bs, L, self.nkv, self.hs), cos, sin)
        v = self.lin(self.wv, h, self.n1.weight).view(Bs, L, self.nkv, self.hs)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True, enable_gqa=True)
        x = x + self.drop(self.lin(self.wo, o.transpose(1, 2).reshape(Bs, L, -1)))
        h = unit_rms(x)
        return x + self.drop(self.lin(self.w2, F.silu(self.lin(self.w1, h, self.n2.weight))
                                      * self.lin(self.w3, h, self.n2.weight)))


class Net(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qat = False
        self.emb = nn.Embedding(c["vocab"], c["dim"])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["layers"]))
        self.norm = RMSNorm(c["dim"])
        self.drop = nn.Dropout(c["dropout"])
        cos, sin = rope_tables(c["dim"] // c["heads"], c["seq"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c["layers"]))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def set_qat(self, on=True):
        self.qat = on
        for b in self.blocks:
            b.qat = on

    def table(self):
        return B.fq8(self.emb.weight) if self.qat else self.emb.weight

    def embed(self, idx):
        return F.embedding(idx, self.table())

    def forward(self, idx):
        x = self.drop(self.embed(idx))
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return (unit_rms(x) * self.norm.weight) @ self.table().T


def export(net, path):
    c = net.c
    hs = c["dim"] // c["heads"]
    out = [struct.pack("7i", c["dim"], c["hidden"], c["layers"], c["heads"], c["kv_heads"], c["vocab"], c["seq"])]

    def put(t):
        out.append(t.detach().float().cpu().numpy().astype(np.float32).tobytes())

    put(net.emb.weight)
    for name in ["n1", "wq", "wk", "wv", "wo", "n2", "w1", "w2", "w3"]:
        for b in net.blocks:
            put(getattr(b, name).weight)
    put(net.norm.weight)
    put(torch.zeros(c["seq"] * hs // 2))
    put(torch.zeros(c["seq"] * hs // 2))
    Path(path).write_bytes(b"".join(out))


class Stream:
    def __init__(self, path, seq, seed):
        self.d = np.memmap(path, np.uint16, "r")
        self.seq = seq
        self.rng = np.random.default_rng(seed)

    def batch(self, n):
        i = self.rng.integers(0, len(self.d) - self.seq - 1, n)
        x = np.stack([self.d[j:j + self.seq + 1] for j in i]).astype(np.int64)
        x = torch.from_numpy(x).pin_memory().cuda(non_blocking=True)
        return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate(net, data, n=40, bs=32):
    net.eval()
    losses = []
    for _ in range(n):
        x, y = data.batch(bs)
        with torch.autocast("cuda", torch.bfloat16):
            losses.append(F.cross_entropy(net(x).float().flatten(0, 1), y.flatten()).item())
    net.train()
    return float(np.mean(losses))


@torch.no_grad()
def chat_samples(net, tok, prompts, temp=0.8, top_k=40, max_new=60, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    net.eval()
    outs = []
    for p in prompts:
        ids = [T.BOS, T.YOU] + tok.encode(p, add_special_tokens=False).ids + [T.HIM]
        seq = torch.tensor([ids], device="cuda")
        for _ in range(max_new):
            with torch.autocast("cuda", torch.bfloat16):
                lg = net(seq[:, -net.c["seq"]:])[0, -1].float() / temp
            v, _ = torch.topk(lg, top_k)
            lg[lg < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1, generator=g)
            if nxt.item() in (T.YOU, T.BOS):
                break
            seq = torch.cat([seq, nxt[None]], 1)
        reply = seq[0, len(ids):].tolist()
        msgs, cur = [], []
        for t in reply + [T.HIM]:
            if t == T.HIM:
                if cur:
                    msgs.append(tok.decode(cur))
                cur = []
            else:
                cur.append(t)
        outs.append({"you": p, "him": msgs})
    net.train()
    return outs


PROMPTS = ["yo what are you up to", "did you see the game last night", "wanna get food later",
           "bro i failed my exam", "whats your plan for the weekend", "lol thats crazy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="b", choices=SIZES)
    ap.add_argument("--name")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tokens", type=float, default=300e6, help="training tokens")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--qat-frac", type=float, default=0.15, help="final fraction of steps with 4-bit fake quant")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synth", default="", help="packed synthetic dialogues (data/tok/synth.bin) mixed into batches")
    ap.add_argument("--synth-frac", type=float, default=0.15, help="share of each batch from --synth")
    a = ap.parse_args()
    name = a.name or f"pre_{a.size}"
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    dim, hidden, layers, heads, kv = SIZES[a.size]
    c = dict(dim=dim, hidden=hidden, layers=layers, heads=heads, kv_heads=kv, vocab=T.VOCAB, seq=a.seq,
             dropout=a.dropout)
    net = Net(c).cuda()
    params = sum(p.numel() for p in net.parameters())
    steps = int(a.tokens // (a.batch * a.seq))
    print(f"{name}: {params / 1e6:.2f}M params, {steps} steps, {steps * a.batch * a.seq / 1e6:.0f}M tokens", flush=True)
    model = torch.compile(net) if a.compile else net
    decay = [p for p in net.parameters() if p.dim() >= 2]
    other = [p for p in net.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    tr = Stream(T.TOK / "train.bin", a.seq, a.seed)
    syn = Stream(a.synth, a.seq, a.seed + 7) if a.synth else None
    n_syn = int(round(a.batch * a.synth_frac)) if syn else 0
    va = Stream(T.TOK / "val.bin", a.seq, 1234)
    tok = T.load()
    log = []
    t0 = time.time()
    warm = min(1000, steps // 20)
    qat_at = int(steps * (1 - a.qat_frac)) if a.qat_frac > 0 else steps + 1
    ckpt = out / f"{name}.ckpt"
    start = 0
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cuda")
        net.load_state_dict(s["state"])
        opt.load_state_dict(s["opt"])
        start, log = s["step"], s["log"]
        tr.rng = np.random.default_rng(a.seed + start)
        net.set_qat(start >= qat_at)
        t0 -= log[-1]["time"] if log else 0
        print(f"resumed at step {start}", flush=True)
    for step in range(start, steps + 1):
        if step > start and step % 2000 == 0:
            torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "step": step, "log": log}, ckpt)
        if step == qat_at:
            net.set_qat(True)
            print(f"step {step}: 4-bit QAT on", flush=True)
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        if step % a.eval_every == 0 or step == steps:
            vl = evaluate(model, va)
            el = time.time() - t0
            log.append({"step": step, "tokens": step * a.batch * a.seq, "val": vl, "time": el})
            print(f"step {step}/{steps} val {vl:.4f} ppl {math.exp(vl):.2f} lr {lr:.1e} "
                  f"{step * a.batch * a.seq / max(el, 1e-9) / 1e3:.0f}k tok/s {el / 60:.1f}min", flush=True)
        if step == steps:
            break
        x, y = tr.batch(a.batch - n_syn)
        if n_syn:
            sx, sy = syn.batch(n_syn)
            x, y = torch.cat([x, sx]), torch.cat([y, sy])
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

    torch.save({"config": c, "state": net.state_dict(), "qat": net.qat}, out / f"{name}.pt")
    ckpt.unlink(missing_ok=True)
    export(net, out / f"{name}.bin")
    samples = chat_samples(net, tok, PROMPTS)
    for s in samples:
        print(json.dumps(s, ensure_ascii=False))
    (out / f"{name}.json").write_text(json.dumps({"args": vars(a), "config": c, "params": params, "log": log,
                                                  "samples": samples}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
