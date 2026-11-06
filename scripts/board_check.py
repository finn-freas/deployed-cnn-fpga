import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chat
import tok as T
from link import Link

PROMPTS = ["how are you", "what are you doing", "did you eat yet", "lol ok", "you coming tonight?",
           "where are you", "want to get food later"]
ARGUE = ["yo", "fuck you", "ur so dumb", "what are you doing", "shut up", "i hate you", "ur a toaster"]


USAGE = "usage: python scripts/board_check.py [model key] [--best] [--argue]"
FLAGS = ("--best", "--argue")


def parse_args(argv):
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    bad = [a for a in flags if a not in FLAGS]
    if bad or len(args) > 1:
        sys.exit(f"{USAGE}" + (f"\nunknown option: {bad[0]}" if bad else ""))
    key = args[0] if args else "pr"
    if not chat.board_file(chat.MODELS, key).exists():
        sys.exit(f"no model {key!r}: {chat.MODELS / (key + '_board.pkl')} not found")
    return key, "--best" in flags, "--argue" in flags


def main():
    key, best, argue = parse_args(sys.argv[1:])
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
        for q in ARGUE if argue else PROMPTS:
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
