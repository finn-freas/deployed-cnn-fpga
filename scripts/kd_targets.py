import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tok as T
from finetune import encode_sessions

HEAD = "Text chat between a teenager (F) and a savage roast bot (B).\n"


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BD = byte_decoder()
BE = {b: c for c, b in BD.items()}


def piece_bytes(s, added):
    return s.encode() if added else bytes(BD[c] for c in s)


class Ours:

    def __init__(self, tok):
        self.tok = tok
        js = json.loads(T.CHAT_TOK.read_text() if T.CHAT_TOK.exists() else (T.TOK / "tokenizer.json").read_text())
        self.added = {a["id"]: a["content"] for a in js["added_tokens"]}
        self.by_content = {c.encode(): i for i, c in self.added.items() if i >= 4}
        self.memo = {}

    def bytes(self, i):
        return self.added[i].encode() if i in self.added else piece_bytes(self.tok.id_to_token(i), False)

    def first(self, b):
        if b not in self.memo:
            r = None
            for c, i in self.by_content.items():
                if b.startswith(c):
                    r = i
            if r is None and b:
                toks = self.tok.model.tokenize("".join(BE[x] for x in b))
                r = toks[0].id if toks else None
            self.memo[b] = r
        return self.memo[b]


def targets(person="pr", K=20, batch=8):
    from writer import WRITER, q4
    tok = T.load_chat()
    ours = Ours(tok)
    wtok, net = q4(WRITER)
    wbytes = {}

    special = set(wtok.all_special_ids)

    def wb(i):
        if i not in wbytes:
            s = wtok.convert_ids_to_tokens(i)
            wbytes[i] = b"" if i in special or any(c not in BD for c in s) else piece_bytes(s, False)
        return wbytes[i]

    d = ROOT / "data" / "people" / person
    for split in ("val", "train"):
        convs = [json.loads(line)["turns"] for line in open(d / f"{split}.jsonl")]
        sess = encode_sessions(tok, d / f"{split}.jsonl")
        out, n_pos, n_tgt, n_skip, t0 = [], 0, 0, 0, time.time()
        jobs = []
        for turns, (ids, _) in zip(convs, sess):
            text, want, k = HEAD.encode(), {}, 1
            for s, t in turns:
                k += 1
                text += b"B:" if s == 1 else b"F:"
                start = len(text)
                body = b" " + t.encode()
                n = len(tok.encode(t, add_special_tokens=False).ids)
                if s == 1:
                    got = b"".join(ours.bytes(int(i)) for i in ids[k:k + n])
                    if got == body:
                        off = start
                        for i in ids[k:k + n]:
                            want[off] = k
                            off += len(ours.bytes(int(i)))
                            k += 1
                        k -= n
                    else:
                        n_skip += 1
                text += body + b"\n"
                k += n
            jobs.append((text, want))
        for b in range(0, len(jobs), batch):
            part = jobs[b:b + batch]
            enc = [wtok(j[0].decode(), add_special_tokens=False).input_ids for j in part]
            L = max(map(len, enc))
            ids_t = torch.tensor([e + [wtok.pad_token_id or 0] * (L - len(e)) for e in enc], device="cuda:0")
            att = torch.tensor([[1] * len(e) + [0] * (L - len(e)) for e in enc], device="cuda:0")
            with torch.no_grad():
                lg = net(input_ids=ids_t, attention_mask=att).logits
            for r, ((text, want), e) in enumerate(zip(part, enc)):
                pos, idx, prob, off = [], [], [], 0
                for j, wi in enumerate(e):
                    if j and off in want:
                        p = lg[r, j - 1].float().softmax(-1)
                        pv, pi = p.topk(K)
                        acc = {}
                        for v, i in zip(pv.tolist(), pi.tolist()):
                            o = ours.first(wb(i))
                            if o is not None and o >= 4:
                                acc[o] = acc.get(o, 0.0) + v
                        if acc:
                            tot = sum(acc.values())
                            items = sorted(acc.items(), key=lambda x: -x[1])[:K]
                            pos.append(want[off])
                            idx.append([i for i, _ in items] + [-1] * (K - len(items)))
                            prob.append([v / tot for _, v in items] + [0.0] * (K - len(items)))
                            n_tgt += 1
                    off += len(wb(wi))
                n_pos += len(want)
                out.append((np.array(pos, np.int32), np.array(idx, np.int16).reshape(-1, K),
                            np.array(prob, np.float16).reshape(-1, K)))
            del lg
            if b % (batch * 100) == 0:
                print(f"{split}: {b + len(part)}/{len(jobs)} chats, {n_tgt}/{n_pos} bot tokens with a target "
                      f"({time.time() - t0:.0f}s)", flush=True)
        with open(d / f"kd_{split}.pkl", "wb") as f:
            pickle.dump({"K": K, "rows": out}, f)
        print(f"{split}: {n_tgt} of {n_pos} bot tokens have a writer target ({n_tgt / max(n_pos, 1):.0%}); "
              f"{n_skip} bot lines skipped (bytes did not match)", flush=True)


if __name__ == "__main__":
    targets(sys.argv[1] if len(sys.argv) > 1 else "pr", int(sys.argv[2]) if len(sys.argv) > 2 else 20)
