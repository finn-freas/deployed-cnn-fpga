import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from writer import GROUPS, NEVER, TIKTOK, WRITER, chat, q4

HAND = [
    "ur so gay 😭", "fuck u", "goku solos btw 💀", "who asked 💀", "nobody asked 😭", "skill issue", "touch grass",
    "ratio", "L + ratio", "cope", "stay mad", "cry about it", "ok and? 💀", "bro is not him 😭", "bro thinks he's him 💀",
    "who let bro cook 💀", "bro fell off", "mid", "this ain't it chief", "the yapping 😭", "bro is yapping",
    "stop yapping 💀", "negative aura 💀", "-1000 aura", "bro glazing", "stop glazing 😭", "no shot 💀",
    "ain't no way 😭", "bro said 💀", "fr? 💀", "stfu 😭", "shut tf up", "dude shut tf up", "nah ur weird for that",
    "why is bro like this 😭", "chill 😭", "ur actually so annoying", "least delusional fan 💀", "average L enjoyer",
    "bro is cooked 💀", "ur cooked", "it's giving loser 😭", "not you saying that 😭", "say less ig 💀",
    "imagine 💀", "couldn't be me", "keep crying", "go outside 😭", "seek help 💀", "bro needs therapy 😭",
    "ur mom", "ur mom gay 😭", "fuck off 😭", "suck it", "get good", "u wish", "be so fr", "ur so dumb for that 💀",
    "lmaooo shut up", "not funny didn't laugh", "ur not funny 😭", "i'm telling ur mom", "bro's down bad 💀",
    "caught in 4k 📸", "zero rizz 💀", "bro has no rizz", "who hurt u 😭", "emotional damage 💀", "bro is the problem",
    "touch grass fr", "ur so lame 😭", "ok boomer", "npc behavior 💀", "bro thought 💀", "the audacity 😭",
]
RATE = ("Could this comment be thrown as a roast or comeback in reply to almost anything a friend texts, without "
        "needing the original video? Score 0 to 9. 9 = works anywhere (like \"who asked\", \"skill issue\"); "
        "0 = only makes sense for one specific video or product.\nComment: {c}\nAnswer with only one digit.")
MIN_SCORE, MAX_WORDS = 6.0, 10


def main():
    mined = [c for c in json.loads(TIKTOK.read_text()) if len(c.split()) <= MAX_WORDS]
    tok, net = q4(WRITER)
    digits = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    keep = []
    for b in range(0, len(mined), 16):
        part = mined[b:b + 16]
        x = tok([chat(tok, None, RATE.format(c=c)) for c in part], return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            p = net(**x, logits_to_keep=1).logits[:, -1, digits].float().softmax(-1)
        for c, v in zip(part, (p * torch.arange(10, device=p.device)).sum(-1).tolist()):
            if v >= MIN_SCORE:
                keep.append(c)
    bank = [c for c in dict.fromkeys(HAND + keep) if not NEVER.search(c) and not GROUPS.search(c.replace("gay", ""))]
    out = ROOT / "data" / "style" / "roast_bank.json"
    out.write_text(json.dumps(bank, ensure_ascii=False, indent=0))
    print(f"{len(HAND)} hand-written + {len(keep)} of {len(mined)} TikTok comments -> {len(bank)} in {out}")
    for c in keep[:30]:
        print("  ", c)


if __name__ == "__main__":
    main()
