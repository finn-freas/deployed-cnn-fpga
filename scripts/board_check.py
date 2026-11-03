import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import chat
import tok as T
from link import Link

PROMPTS = ["how are you", "what are you doing", "did you eat yet", "lol ok", "you coming tonight?",
           "where are you", "want to get food later"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    best = "--best" in sys.argv
    key = args[0] if args else "pr"
    tok = T.load_chat()
    link = Link()
    print("ping", link.ping())
    brd = chat.BoardPerson(key, link)
    cpu = chat.Person(key)
    same, n, times = 0, 0, []
    for seed in (1, 2):
        cpu.reset()
        brd.reset()
        rng = chat.make_rng(seed)
        link.seed(chat.make_rng(seed).x)
        for q in PROMPTS:
            t0 = time.time()
            if best:
                a, _, wa = chat.reply_best(cpu, tok, q, chat.TEMP.get(key, 0.8), 4.0, rng, n_best=cpu.n_best)
                t0 = time.time()
                b, ntok, wb = chat.reply_best(brd, tok, q, chat.TEMP.get(key, 0.8), 4.0, n_best=brd.n_best)
                a, b = (a, wa, cpu.hist), (b, wb, brd.hist)
            else:
                a = chat.reply(cpu, tok, q, 0.8, 4.0, rng)[0]
                t0 = time.time()
                b, ntok, _ = chat.board_reply(brd, tok, q, 0.8, 4.0)
            times.append((time.time() - t0, ntok))
            same += a == b
            n += 1
            print(f"seed {seed} {q!r}: {'same' if a == b else 'DIFFERENT'} {times[-1][0]:.2f}s ({ntok} tokens)  "
                  f"{b[:2] if best else b}" + ("" if a == b else f"  cpu: {a[:2] if best else a}"), flush=True)
    tot = sum(t for t, _ in times)
    print(f"{same}/{n} replies identical to the CPU; {tot / n:.2f} s per reply on average, "
          f"{max(t for t, _ in times):.2f} s at most, {sum(k for _, k in times) / tot:.1f} tokens/s")


if __name__ == "__main__":
    main()
