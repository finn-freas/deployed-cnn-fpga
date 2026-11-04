import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent.parent
CORPUS, TOK = ROOT / "data" / "corpus", ROOT / "data" / "tok"
VOCAB = 3072
N_RESERVED = 128
PAD, BOS, YOU, HIM = 0, 1, 2, 3
SPECIAL = ["<pad>", "<bos>", "<you>", "<him>"] + [f"<r{i}>" for i in range(N_RESERVED)]


CHAT_TOK = TOK / "tokenizer_chat.json"


def load(path=TOK / "tokenizer.json"):
    return Tokenizer.from_file(str(path))


def load_chat():
    return load(CHAT_TOK if CHAT_TOK.exists() else TOK / "tokenizer.json")


def fill_slots(strings):
    base = load()
    assert len(strings) <= N_RESERVED
    js = json.loads((TOK / "tokenizer.json").read_text())
    vocab = js["model"]["vocab"]
    out = []
    for i, s in enumerate(strings):
        old = base.encode(s, add_special_tokens=False).ids
        tid = 4 + i
        del vocab[f"<r{i}>"]
        vocab[s] = tid
        at = next(a for a in js["added_tokens"] if a["id"] == tid)
        at.update(content=s, special=False)
        out.append((tid, s, old))
    CHAT_TOK.write_text(json.dumps(js, ensure_ascii=False))
    return out


def turns_text(path, limit=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            for _, t in json.loads(line)["turns"]:
                yield t


def train_tokenizer(sample=400_000):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    lines = open(CORPUS / "train.jsonl").readlines()
    step = max(1, len(lines) // sample)
    texts = (t for line in lines[::step] for _, t in json.loads(line)["turns"])
    tok.train_from_iterator(texts, tr)
    assert tok.token_to_id("<him>") == HIM and tok.get_vocab_size() == VOCAB
    TOK.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOK / "tokenizer.json"))
    return tok


def encode_dialogue(tok, turns, swap=False):
    out = [BOS]
    enc = tok.encode_batch([t for _, t in turns], add_special_tokens=False)
    for (s, _), e in zip(turns, enc):
        out.append(HIM if (s == 1) != swap else YOU)
        out += e.ids
    return out


def encode_split(tok, name, rng):
    stats = {}
    chunk = []
    n = 0

    def flush(out):
        nonlocal n
        enc = tok.encode_batch([t for d in chunk for _, t in d["turns"]], add_special_tokens=False)
        ids = []
        k = 0
        for d in chunk:
            swap = bool(rng.integers(2))
            ids.append(BOS)
            for s, _ in d["turns"]:
                ids.append(HIM if (s == 1) != swap else YOU)
                ids.extend(enc[k].ids)
                k += 1
            stats[d["src"]] = stats.get(d["src"], 0) + 1
        np.array(ids, np.uint16).tofile(out)
        n += len(ids)
        chunk.clear()

    with open(CORPUS / f"{name}.jsonl") as f, open(TOK / f"{name}.bin", "wb") as out:
        for line in f:
            chunk.append(json.loads(line))
            if len(chunk) == 20000:
                flush(out)
        flush(out)
    return n, stats


def main():
    tok = train_tokenizer() if "--reuse" not in sys.argv else load()
    rng = np.random.default_rng(0)
    info = {}
    for name in ["val", "train"]:
        n, st = encode_split(tok, name, rng)
        info[name] = {"tokens": n, "dialogues": st}
        print(name, info[name], flush=True)
    chars = json.loads((CORPUS / "stats.json").read_text())
    info["chars_per_token"] = (chars.get("soda_chars", 0) + chars.get("reddit_chars", 0)) / info["train"]["tokens"]
    (TOK / "info.json").write_text(json.dumps(info, indent=1))
    for s in ["lmao bro what are you doing tonight", "nah im good, see u tmrw 😂"]:
        e = tok.encode(s, add_special_tokens=False)
        print(len(s), "chars ->", len(e.ids), "tokens:", e.tokens)


if __name__ == "__main__":
    main()
