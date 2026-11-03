import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

import board as B
import tok as T
from engine import CLK_HZ, token_cycles

ROOT = Path(__file__).resolve().parent
HELP = ("/reset, /temp 0.6, /cut 4, /seed 7, /emoji 0.5 (0.5 = the usual amount, higher = more), "
        "/best 6 (candidates per reply, 1 = off), /info, /quit")
MODELS = ROOT / "models"
MAX_REPLY = 80
BEST_N = 6
BEST_BUDGET = 80
BUDGET = {"pr": 150}
BEST_RESERVE = 20
VOICE_LAM = 1.0
STYLE = {"pr": (6, 1.0, 0.0)}
TEMP = {"pr": 0.6}
EMOJI_LEVEL = {"pr": 0.5}
FRAGMENT = -3.0
SAY_MORE = {"pr": {" fuck": 1.0, " fucking": 1.0, " shit": 1.0, " damn": 1.0, " ass": 1.0}}
REPEAT = {"pr": 6.0}
REPEAT_N, REPEAT_COMMON, REPEAT_EMOJI = 4, 200, 4.0
EMOJI_SCORE = {"pr": -1.0}

KEEP = 256
EMOJI_PRIOR = 3
EMOJI_GAIN_MAX = 64.0
EMOJI_GAIN_MIN = 0.02
EMOJI_HORIZON = 6
EMOJI_RECENT = 8


def board_file(mdir, key):
    f = mdir / f"{key}_board.pkl"
    xz = mdir / f"{key}_board.pkl.xz"
    if not f.exists() and xz.exists():
        import lzma
        f.write_bytes(lzma.decompress(xz.read_bytes()))
    return f


def make_rng(seed):
    x = seed & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= x >> 16
    return B.XorShift(x)


def emoji_scale(emoji):
    return (2 * max(0.0, emoji)) ** 2


def emoji_mask():
    mask = np.zeros(T.VOCAB, bool)
    slots = T.TOK / "slots.json"
    if slots.exists():
        for tid, s, _ in json.loads(slots.read_text()):
            if not ("︀" <= s <= "️" or "\U0001F3FB" <= s <= "\U0001F3FF"):
                mask[tid] = True
    return mask


EMOJI = emoji_mask()


class Person:
    def __init__(self, key, mdir=None):
        self.key = key
        mdir = Path(mdir) if mdir else MODELS
        self.p = pickle.loads(board_file(mdir, key).read_bytes())
        self.m = B.IntModel(self.p)
        self.name = "roast bot"
        cal = mdir / f"{key}_emoji.json"
        cal = json.loads(cal.read_text()) if cal.exists() else {}
        self.emoji_corr, self.emoji_rate = cal.get("corr", 1.0), cal.get("rate")
        self.emoji_cold = cal.get("cold_mass")
        self.n_best, alpha, lam = STYLE.get(key, (BEST_N, 1.0, VOICE_LAM))
        tk = T.load_chat()
        piece = {(tuple(tk.encode(t, add_special_tokens=False).ids) if t.startswith(" ") else (tk.token_to_id(t),)): b
                 for t, b in SAY_MORE.get(key, {}).items()}
        extra = {i: FRAGMENT for i in range(T.VOCAB) if "�" in tk.decode([i])}
        extra.update({i[0]: b for i, b in piece.items() if len(i) == 1 and i[0] is not None})
        self.w_q = B.score_table(mdir, key, T.VOCAB, lam, alpha, extra)
        if key in EMOJI_SCORE:
            words = self.w_q[4:][~EMOJI[4:]]
            self.w_q[EMOJI] = round(np.median(words) + EMOJI_SCORE[key] * math.log2(math.e) * 256)
        self.p["score_w"] = self.w_base = self.w_q.copy()
        u = mdir / "generic_unigram.npy"
        self.common = set(np.argsort(-np.load(u))[:REPEAT_COMMON].tolist()) if u.exists() else set()
        self.common |= {i for i in range(T.VOCAB) if not tk.decode([i]).strip(" .,!?'’\"-")}
        self.emoji_bias = np.zeros(T.VOCAB, np.int64)
        for tid, nats in cal.get("bias", []):
            self.emoji_bias[tid] = round(nats * 2 ** self.p["fmt"]["logit"])
        self.reset()

    def emoji_gain(self, emoji):
        scale = emoji_scale(emoji)
        if not self.emoji_rate or scale == 0:
            return self.emoji_corr * scale
        target = min(0.9, self.emoji_rate * scale)
        need = (target * (self.bot_msgs + EMOJI_HORIZON) - self.bot_emoji) / EMOJI_HORIZON
        if need <= 0:
            return EMOJI_GAIN_MIN
        need = min(need, 0.9)
        prior = self.emoji_cold or -math.log(1 - target) / (self.emoji_corr * scale)
        m = np.array(self.bot_mass[-EMOJI_RECENT:] + [prior] * max(0, EMOJI_PRIOR - len(self.bot_mass)))
        lo, hi = math.log(EMOJI_GAIN_MIN), math.log(EMOJI_GAIN_MAX)
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.mean(1 - np.exp(-math.exp(mid) * m)) < need else (lo, mid)
        return math.exp((lo + hi) / 2)

    def repeat_table(self):
        w, pen, q = self.w_base.copy(), REPEAT.get(self.key, 0.0), math.log2(math.e) * 256
        if pen:
            for age, toks in enumerate(reversed(self.said[-REPEAT_N:])):
                f = (REPEAT_N - age) / REPEAT_N
                for t in set(toks) - self.common:
                    if t > 3:
                        w[t] = min(w[t], self.w_base[t] - round(q * f * (REPEAT_EMOJI if EMOJI[t] else pen)))
        w = np.clip(w, -32768, 32767)
        changed = np.flatnonzero(w != self.w_q)
        self.w_q = w
        self.push_table(changed)

    def push_table(self, changed):
        pass

    def reset(self):
        self.said = []
        self.bot_msgs = self.bot_emoji = 0
        self.bot_mass = []
        self.m.reset()
        self.hist = []
        self.pos = 0
        self.last = None

    def feed(self, ids):
        for t in ids:
            self.last = self.m.forward(int(t), self.pos)
            self.hist.append(int(t))
            self.pos += 1

    def settings(self, inv_t, cut_q, gain_q, rng):
        self.inv_t, self.cut_q, self.gain_q, self.rng = inv_t, cut_q, gain_q, rng

    def rewind(self, pos):
        self.pos = pos
        del self.hist[pos:]

    def step(self, t):
        self.feed([t])
        f = self.p["fmt"]["logit"]
        em, total = B.class_mass(self.last, EMOJI, f, self.p["exp2"])
        g = 1 << B.GAIN_BITS if EMOJI[t] else self.gain_q
        nxt = B.pick(self.last, self.inv_t, self.cut_q, self.rng, self.p["gumbel"], EMOJI, g, f, self.p["exp2"],
                     self.emoji_bias)
        return nxt, em, total, B.token_score(self.last, nxt, f, total, self.w_q)

    def best_reset(self):
        self.best = None
        self.cand = 0
        self.acc = [0, 0]

    def score(self, s):
        self.acc[0] += s
        self.acc[1] += 1

    def cand_end(self, keep=True):
        s, n = self.acc
        if keep and n and (self.best is None or B.better(s, n, self.best[0], self.best[1])):
            self.best = (s, n, self.cand)
        self.cand += 1
        self.acc = [0, 0]
        return self.best[2] if self.best else 0

    def make_room(self, need):
        if self.pos + need <= self.p["cfg"].seq_len:
            return
        cut = [i for i, t in enumerate(self.hist) if t in (T.YOU, T.HIM) and len(self.hist) - i <= KEEP]
        keep = self.hist[cut[0]:] if cut else []
        self.m.reset()
        self.hist, self.pos = [], 0
        self.feed([T.BOS] + keep)


def reply(person, tok, text, temp, cut, rng, emoji=0.5):
    ids = ([T.BOS] if person.pos == 0 else []) + [T.YOU] + tok.encode(text, add_special_tokens=False).ids + [T.HIM]
    person.make_room(len(ids) + MAX_REPLY)
    cyc = sum(token_cycles(person.p, person.pos + i) for i in range(len(ids)))
    person.feed(ids)
    inv_t = max(1, round(256 / temp))
    cut_q = round(cut * 2 ** person.p["fmt"]["logit"])
    f_logit = person.p["fmt"]["logit"]
    gain_q = round(person.emoji_gain(emoji) * 2 ** B.GAIN_BITS)
    msgs, cur, n, prev, mass, cur_mass = [], [], 0, T.HIM, [], 0.0
    while n < MAX_REPLY:
        g = 1 << B.GAIN_BITS if EMOJI[prev] else gain_q
        if not EMOJI[prev]:
            em, total = B.class_mass(person.last, EMOJI, f_logit, person.p["exp2"])
            cur_mass += em / max(total, 1)
        nxt = B.pick(person.last, inv_t, cut_q, rng, person.p["gumbel"], EMOJI, g, f_logit, person.p["exp2"],
                     person.emoji_bias)
        cyc_tok = token_cycles(person.p, person.pos)
        n += 1
        if nxt in (T.YOU, T.BOS, T.PAD):
            break
        cyc += cyc_tok
        person.feed([nxt])
        prev = nxt
        if nxt == T.HIM:
            msgs.append(cur)
            mass.append(cur_mass)
            cur, cur_mass = [], 0.0
        else:
            cur.append(nxt)
    msgs.append(cur)
    mass.append(cur_mass)
    person.bot_mass += [ms for m, ms in zip(msgs, mass) if m]
    msgs = [m for m in msgs if m]
    person.bot_msgs += len(msgs)
    person.bot_emoji += sum(any(EMOJI[t] for t in m) for m in msgs)
    texts = [tok.decode(m).strip() for m in msgs]
    return texts, n, cyc / CLK_HZ


def loops(toks):
    w = [t for t in toks if t != T.HIM and not EMOJI[t]]
    tri = list(zip(w, w[1:], w[2:]))
    return len(set(tri)) < len(tri) or any(w[i] == w[i + 1] == w[i + 2] for i in range(len(w) - 2))


MOM = re.compile(r"\b(your|ur|yo|you're|you’re) (mom|mama|momma|moms)\b")
MOM_GAP = 6
MOM_SLACK = float(os.environ.get("FB_MOM_SLACK", "260"))
AGGRO = re.compile(r"\b(fuck|f\*ck|fk|stfu|shut (up|it)|hate|stupid|dumb|trash|suck|ugly|annoying|bitch|loser|idiot|"
                   r"cringe|mid|lame|weak|garbage|go away|stop talking|screw|unplug|fry|beat you|not funny|"
                   r"(just|only|a|little|lil|dumb|stupid) (a )?bot|toaster|calculator|ratio|my mom|my mama|my moms|"
                   r"ur mom|your mom|yo mama)\b", re.I)


def has_mom(tok, toks):
    return bool(MOM.search(tok.decode(toks).lower()))


def mom_bad(person, tok, toks, text):
    said = lambda t: bool(MOM.search(tok.decode(t).lower()))
    return said(toks) and (not AGGRO.search(text.replace("’", "'")) or any(said(r) for r in person.said[-MOM_GAP:]))


def reply_best(person, tok, text, temp, cut, rng=None, emoji=None, n_best=BEST_N, budget=None):
    emoji = EMOJI_LEVEL.get(person.key, 0.5) if emoji is None else emoji
    budget = BUDGET.get(person.key, BEST_BUDGET) if budget is None else budget
    ids = ([T.BOS] if person.pos == 0 else []) + [T.YOU] + tok.encode(text, add_special_tokens=False).ids + [T.HIM]
    person.make_room(len(ids) + MAX_REPLY)
    person.repeat_table()
    person.settings(max(1, round(256 / temp)), round(cut * 2 ** person.p["fmt"]["logit"]),
                    round(person.emoji_gain(emoji) * 2 ** B.GAIN_BITS), rng)
    person.feed(ids[:-1])
    p0 = person.pos
    person.best_reset()
    cands, used, last, moms, means = [], 0, -1, [], {}
    for k in range(n_best):
        if k and used > budget - BEST_RESERVE:
            break
        person.rewind(p0)
        nxt, em, total, sc = person.step(T.HIM)
        fed, msgs, cur, n, prev, mass, cur_mass, ok = [], [], [], 0, T.HIM, [], 0.0, True
        while n < MAX_REPLY:
            if not EMOJI[prev]:
                cur_mass += em / max(total, 1)
            person.score(sc)
            n += 1
            if used + n > budget:
                ok = False
                break
            if nxt in (T.YOU, T.BOS, T.PAD):
                break
            prev = nxt
            fed.append(nxt)
            if n < MAX_REPLY:
                following = person.step(nxt)
            else:
                person.feed([nxt])
            if nxt == T.HIM:
                msgs.append(cur)
                mass.append(cur_mass)
                cur, cur_mass = [], 0.0
            else:
                cur.append(nxt)
            if n < MAX_REPLY:
                nxt, em, total, sc = following
        msgs.append(cur)
        mass.append(cur_mass)
        used += n
        last = k
        good = ok and not loops(fed) and not mom_bad(person, tok, fed, text)
        if good and person.acc[1] and has_mom(tok, fed):
            moms.append((person.acc[0] / person.acc[1], k))
        if good and person.acc[1]:
            means[k] = person.acc[0] / person.acc[1]
        win = person.cand_end(good)
        cands.append((fed, msgs, mass, ok))
    if moms and win in means and max(moms)[0] >= means[win] - MOM_SLACK:
        win = max(moms)[1]
    if win != last:
        person.rewind(p0)
        person.feed([T.HIM] + cands[win][0])
    _, msgs, mass, _ = cands[win]
    person.said.append([t for m in msgs for t in m])
    person.bot_mass += [ms for m, ms in zip(msgs, mass) if m]
    msgs = [m for m in msgs if m]
    person.bot_msgs += len(msgs)
    person.bot_emoji += sum(any(EMOJI[t] for t in m) for m in msgs)
    person.last_cands = [(" / ".join(tok.decode(m).strip() for m in c[1] if m), c[3]) for c in cands]
    return [tok.decode(m).strip() for m in msgs], used, win


class BoardPerson(Person):

    def __init__(self, key, link, log=print):
        self.link = link
        super().__init__(key)
        import pack as P
        from link import checksum
        im = P.build(self.p)[0]
        img, words = im.bytes(), len(im.words)
        self.w_addr = im.sym["score_w"][0]
        link.write_bytes(self.w_addr, img[8 * self.w_addr:8 * (self.w_addr + T.VOCAB)])
        if link.checksum(0, words) != checksum(img):
            log(f"uploading {self.name}'s model to the board (about a minute)...")
            link.write_bytes(0, img)
            if link.checksum(0, words) != checksum(img):
                raise IOError("image upload failed its checksum")
        link.init_engine()
        link.emoji_table(EMOJI[:128], self.emoji_bias[:128])
        link.new_conversation()

    def push_table(self, changed):
        if not len(changed):
            return
        runs, start = [], changed[0]
        for a, b in zip(changed, list(changed[1:]) + [None]):
            if b is None or b - a > 8:
                runs.append((start, a + 1))
                start = b
        for a, b in runs:
            data = b"".join((int(v) & (2 ** 64 - 1)).to_bytes(8, "little") for v in self.w_q[a:b])
            self.link.write_bytes(self.w_addr + int(a), data)
        self.link.init_engine()
        self.link.emoji_table(EMOJI[:128], self.emoji_bias[:128])

    def reset(self):
        super().reset()
        if hasattr(self, "link"):
            self.link.new_conversation()

    def feed(self, ids):
        for t in ids:
            self.pos = self.link.feed(int(t))
            self.hist.append(int(t))

    def step(self, t):
        r = self.link.step(int(t))
        self.hist.append(int(t))
        self.pos += 1
        return r

    def settings(self, inv_t, cut_q, gain_q, rng):
        self.link.settings(inv_t, cut_q, gain_q)

    def rewind(self, pos):
        self.link.jump(pos)
        super().rewind(pos)

    def best_reset(self):
        super().best_reset()
        self.link.best(clear=True)

    def cand_end(self, keep=True):
        want = super().cand_end(keep)
        k, s, n = self.link.best(end=True, keep=keep)
        if self.best and (k, s, n) != (self.best[2], self.best[0], self.best[1]):
            raise IOError(f"board best {k, s, n} != host {self.best[2], self.best[0], self.best[1]}")
        return k if self.best else want

    def make_room(self, need):
        if self.pos + need <= self.p["cfg"].seq_len:
            return
        cut = [i for i, t in enumerate(self.hist) if t in (T.YOU, T.HIM) and len(self.hist) - i <= KEEP]
        keep = self.hist[cut[0]:] if cut else []
        self.link.new_conversation()
        self.hist, self.pos = [], 0
        self.feed([T.BOS] + keep)


def board_reply(person, tok, text, temp, cut, emoji=0.5):
    ids = ([T.BOS] if person.pos == 0 else []) + [T.YOU] + tok.encode(text, add_special_tokens=False).ids + [T.HIM]
    person.make_room(len(ids) + MAX_REPLY)
    person.link.settings(max(1, round(256 / temp)), round(cut * 2 ** person.p["fmt"]["logit"]),
                         round(person.emoji_gain(emoji) * 2 ** B.GAIN_BITS))
    person.feed(ids[:-1])
    nxt, em, total, _ = person.step(ids[-1])
    msgs, cur, n, prev, mass, cur_mass = [], [], 0, T.HIM, [], 0.0
    while n < MAX_REPLY:
        if not EMOJI[prev]:
            cur_mass += em / max(total, 1)
        n += 1
        if nxt in (T.YOU, T.BOS, T.PAD):
            break
        prev = nxt
        if n < MAX_REPLY:
            following = person.step(nxt)
        else:
            person.feed([nxt])
        if nxt == T.HIM:
            msgs.append(cur)
            mass.append(cur_mass)
            cur, cur_mass = [], 0.0
        else:
            cur.append(nxt)
        if n < MAX_REPLY:
            nxt, em, total, _ = following
    msgs.append(cur)
    mass.append(cur_mass)
    person.bot_mass += [ms for m, ms in zip(msgs, mass) if m]
    msgs = [m for m in msgs if m]
    person.bot_msgs += len(msgs)
    person.bot_emoji += sum(any(EMOJI[t] for t in m) for m in msgs)
    return [tok.decode(m).strip() for m in msgs], n, 0.0


def main():
    tok = T.load_chat()
    keys = [k for k in ["pr"] if board_file(MODELS, k).exists()]
    if not keys:
        sys.exit("no model yet (models/pr_board.pkl)")
    link = None
    if "--board" in sys.argv:
        from link import Link
        i = sys.argv.index("--board")
        first = "pr"
        link = Link()
        people = {first: BoardPerson(first, link)}
    else:
        people = {k: Person(k) for k in keys}
    cur = next(iter(people.values()))
    temp, cut, seed, emoji, best = None, 4.0, 1, None, None
    rng = make_rng(seed)
    if link:
        link.seed(rng.x)
        if link.ping()["version"] < 2:
            best = 1
            print("(the board has an older bitstream: one reply per message, /best is off)")
    where = "on the Tang Nano" if link else "on this computer"
    print(f"roastbot {where}: /help for commands")
    while True:
        try:
            line = input("\033[36myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, *arg = line[1:].split(maxsplit=1) or ["help"]
            arg = arg[0] if arg else ""
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "reset":
                cur.reset()
                print("conversation cleared")
            elif cmd == "temp" and arg:
                temp = None if arg == "auto" else float(arg)
                print("each person's own temperature" if temp is None else
                      f"temperature {temp} (board inv_t {round(256 / temp)})")
            elif cmd == "cut" and arg:
                cut = float(arg)
                print(f"cutoff: only words within {cut} of the best (lower = safer, higher = wilder)")
            elif cmd == "emoji" and arg:
                emoji = float(arg)
                print(f"emoji {emoji} (0.5 = their usual amount, 1 = more, 0 = none)")
            elif cmd == "seed" and arg:
                rng = make_rng(int(arg))
                if link:
                    link.seed(rng.x)
                print(f"seed {arg}")
            elif cmd == "best" and arg:
                best = None if arg == "auto" else max(1, min(BEST_N, int(arg)))
                print("each person's own number of candidates" if best is None else
                      f"writing up to {best} replies per message and showing the one most like them" if best > 1
                      else "one reply per message")
            elif cmd == "info":
                c = cur.p["cfg"]
                mm = B.memory_map(cur.p)
                print(f"{cur.name} {where}: {c.n_layers} layers x {c.dim}, 4-bit weights, {cur.pos}/{c.seq_len} "
                      f"tokens used, temp {TEMP.get(cur.key, 0.8) if temp is None else temp}, cut {cut}, emoji {EMOJI_LEVEL.get(cur.key, 0.5) if emoji is None else emoji}, best of {best or cur.n_best}, SDRAM {mm['total'] / 2 ** 20:.2f} MB")
            else:
                print(HELP)
            continue
        t0 = time.time()
        nb = best or cur.n_best
        t_ = TEMP.get(cur.key, 0.8) if temp is None else temp
        if nb > 1:
            texts, n, _ = reply_best(cur, tok, line, t_, cut, None if link else rng, emoji, n_best=nb)
            secs = n * token_cycles(cur.p, cur.pos) / CLK_HZ
        elif link:
            texts, n, _ = board_reply(cur, tok, line, t_, cut, EMOJI_LEVEL.get(cur.key, 0.5) if emoji is None else emoji)
        else:
            texts, n, secs = reply(cur, tok, line, t_, cut, rng, EMOJI_LEVEL.get(cur.key, 0.5) if emoji is None else emoji)
        texts = [t.replace("�", "").strip() for t in texts]
        for t in [t for t in texts if t] or ["..."]:
            print(f"\033[33m{cur.name}>\033[0m {t}")
        took = f"took {time.time() - t0:.2f} s on the board" if link else f"board ~{secs * 1e3:.0f} ms"
        print(f"\033[90m  [{n} tokens, {took} incl. reading your message]\033[0m")


if __name__ == "__main__":
    main()
