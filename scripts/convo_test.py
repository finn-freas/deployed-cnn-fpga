import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import chat
import tok as T

CHATS = [
    ["yo i just lost 5 games in a row in fortnite", "it wasnt my fault my teammates were trash", "ok fine maybe i suck a little", "wanna play later tho"],
    ["i got a new haircut today", "my mom said it looks good", "why are you being so mean", "ok what should i get next time"],
    ["what should we eat tonight", "i was thinking pizza", "pineapple on it obviously", "youre just jealous"],
    ["goku would beat superman easy", "superman doesnt even have ultra instinct", "name one thing superman beat", "thats not even canon"],
    ["i have a math test tomorrow", "i havent studied at all", "can you help me", "its on fractions"],
    ["yo", "woah woah", "ur a bot bro", "ok shut up", "how are you", "calm down", "whatever"],
]


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "pr"
    if "norepeat" in sys.argv:
        chat.REPEAT = {}
    for a in sys.argv[2:]:
        k, _, v = a.partition("=")
        if k == "repeat":
            chat.REPEAT = {key: float(v)}
        elif k == "emoji":
            chat.REPEAT_EMOJI = float(v)
    tok = T.load_chat()
    p = chat.Person(key)
    for seed, msgs in enumerate(CHATS, 1):
        p.reset()
        rng = chat.make_rng(seed)
        print(f"--- chat {seed}")
        for m in msgs:
            texts, used, _ = chat.reply_best(p, tok, m, 0.8, 4.0, rng, n_best=p.n_best)
            print(f"  you: {m}")
            for t in texts:
                print(f"  bot: {t.replace(chr(0xfffd), '')}")
            print(f"       [{used} tokens, ~{used / 23:.1f} s on the board]")


if __name__ == "__main__":
    main()
