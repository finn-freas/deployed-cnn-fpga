import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from writer import GROUPS, NEVER, WRITER, chat, q4

TOPICS = [
    "fortnite", "minecraft", "a new video game", "losing a game online", "homework", "a test tomorrow", "a teacher",
    "school lunch", "being bored", "what to eat tonight", "pizza toppings", "a new haircut", "new shoes", "a crush",
    "getting left on read", "a party this weekend", "being grounded", "a parent being annoying", "a sibling",
    "basketball", "football", "soccer practice", "the gym", "being tired", "sleeping in", "a movie", "an anime",
    "goku vs superman", "a youtuber", "tiktok", "a song", "a concert", "driving", "getting a job", "money",
    "being broke", "a phone breaking", "wifi being down", "a trip", "the weather", "a dog", "a cat", "cooking",
    "fast food", "a group chat", "a friend being late", "a prank", "a bet", "who is better at something",
    "making plans to hang out", "a halloween costume", "christmas presents", "a birthday", "a fight at school",
    "a rumor", "math class", "a substitute teacher", "a vacation", "staying up late", "a dream they had",
]
EMOJI = "🤣😂🙏👀🙄🤓😩😤😏🤨🤦😳"
SYSTEM = ("You write a realistic text conversation between two teenage friends, F and B. B is a funny, savage friend "
          "who roasts F and curses a lot, in the style of TikTok comment sections. He works curse words like "
          "{curses} into his roasts so they fit the joke. Every roast is about what F actually just said (his "
          "{topic}, his excuses, what he asked), never a generic line. B still actually talks: when F asks "
          "something B answers it for real (then roasts him), he remembers what F said a message or two earlier "
          "and calls back to it, asks F things, and keeps the conversation going. Both text like teens: every "
          "line is 3 to 15 words, mostly lowercase, no punctuation fuss. The vibe of B's texts (don't copy these, "
          "just the style): {style}. Every B text is different (never repeat a joke or an opener). B ends some texts with "
          "💀 or 😭 (most often), sometimes with one of {emoji} instead, and many texts have no emoji. At most "
          "once, B may throw in one of these lines if it really fits: {roasts}. No slurs, no jokes about race, "
          "religion or disability. Write 8 lines, alternating F: and B:, starting with F texting about {topic}. "
          "Only the conversation.")
OPENERS = [
    "yo", "sup", "hi", "bro shut up", "fuck you", "i hate you", "ur not funny", "ur just a bot", "ur a little bot",
    "who even made you", "im gonna unplug you", "im gonna fry your motherboard", "my grandma roasts better than you",
    "say sorry", "ur mid", "bet you cant roast me", "ur ugly", "stop talking", "ur dumb", "cry about it", "L bot",
    "why are you like this", "calm down", "whatever bro", "nobody likes you", "ur annoying as hell", "be nice",
    "roast me", "ur trash at this", "thats not even a roast", "ok and?", "ur running on a toaster", "touch grass bot",
    "u cant even see me", "i could beat you up", "ur so slow", "bro ur cringe", "talk to me nicely", "u got no friends",
    "im smarter than you", "lol ur so bad", "ur a calculator with an attitude", "say something nice", "ratio",
    "my phone is smarter than u", "u wish u were real", "imagine being a bot", "ur a whole joke", "go away",
]
ARGUE = ("You write a realistic text chat between a teenager F and B, a savage roast bot (B knows he runs on a tiny "
         "computer board and doesn't care). F is not bringing up a topic: F is arguing with B the whole time: he "
         "insults B, tells him to shut up, says he hates him, says B is just a bot or not funny, jokingly threatens "
         "him, gets mad, sometimes tries to be nice or changes the subject. B never backs down and never apologizes: "
         "every B text answers exactly what F just said with a sharper comeback (if F says B is a bot, B roasts F "
         "for losing to a bot; if F threatens him, B mocks the threat), calls back to what F said earlier, and "
         "curses inside the roast with words like {curses}. Both text like teens: every line is 3 to 15 words, "
         "mostly lowercase. The vibe of B's texts (don't copy these, just the style): {style}. Every B text is "
         "different (never repeat a joke or an opener). At most once, B may throw in one of these lines if it "
         "really fits: {roasts}. No slurs, no jokes about race, religion or disability, nothing about suicide. "
         "Write 10 lines, alternating F: and B:, starting with F texting \"{topic}\". Only the conversation.")
MOM = (" Exactly one of B's texts works in a cocky 'i had your mom' line, woven into his comeback or roast about what F just said "
       "(like {moms}). Not a 'yo mama so fat' joke, and nothing about her dying, being sick or being poor. B's other "
       "texts don't mention F's mom.")
MOMS = ["fuck you too, i had your mom last night", "i had your mom screaming that in bed",
        "your mom wasnt saying that last night", "thats not what your mom was screaming",
        "relax lil bro, im your dad now", "big talk for a kid whose mom is in my bed",
        "i was busy with your mom, what did i miss", "your mom didnt hate me last night",
        "your mom calls me daddy tho", "go ask your mom where i was last night",
        "your mom said the same thing then asked me to stay", "like i fucked your mom"]
MOM2 = (" Exactly one of B's texts answers one of F's insults (like 'fuck you' or 'shut up') with a cocky 'i had your "
        "mom' line that fits what F said, in the spirit of {moms} but made up fresh (don't copy these). Not a 'yo mama "
        "so fat' joke, and nothing about her dying, being sick or being poor. B's other texts don't mention F's mom.")
MOM3 = (" Exactly one of B's texts answers one of F's insults (like 'fuck you' or 'shut up') with a cocky line about "
        "F's mom (that B is with her, she likes B more, she was busy with B, B is his new dad...) that fits what F "
        "said, in the spirit of {moms} but made up fresh in your own words (don't copy these, and don't start it "
        "with 'i had your mom'). Not a 'yo mama so fat' joke, and nothing about her dying, being sick or being poor. "
        "B's other texts don't mention F's mom.")
MOMS3 = [x for x in MOMS if not x.startswith(("i had", "fuck you too"))]
I_HAD = re.compile(r"\bi had (your|ur) mom")
norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("’", "")).split())
MOM_RE = re.compile(r"\b(your|ur|yo|you're) (mom|mama|momma|moms)\b")
YO_MAMA = re.compile(r"\b(yo|your|ur) (mom|mama|momma)('?s)? (is )?so\b")
CURSES = ["fuck", "fatass", "fucker", "bullshit", "lil bitch", "bitch", "bitchass", "ugly ass", "dumbass", "shit",
          "damn", "fuckin", "ass"]
STOCK = ["thinks he's him", "who asked", "clapped by a bot", "ur cooked", "skill issue", "ain't no way",
         "bullshit and u know", "zero rizz", "ur mom"]
LINE = re.compile(r"^\s*\**([FB])\**\s*:\s*(.+)$")


def style_lines():
    return json.loads((ROOT / "data" / "style" / "tiktok_roasts.json").read_text())


def stock_lines(turns, skip=()):
    return sum(any(p in t.lower().replace("’", "'") for p in STOCK if p not in skip) for s, t in turns if s == 1)


def mom_lines(turns):
    return sum(bool(MOM_RE.search(t.lower().replace("’", "'"))) for s, t in turns if s == 1)


def mom_fits(turns):
    from chat import AGGRO
    for j, (k, t) in enumerate(turns):
        if k == 1 and j and MOM_RE.search(t.lower().replace("’", "'")):
            if not AGGRO.search(turns[j - 1][1].replace("’", "'")) or any(norm(m) in norm(t) for m in MOMS):
                return False
    return True


def parse(text, mom=False, strict=False):
    turns = []
    for line in text.split("\n"):
        m = LINE.match(line)
        if m:
            s, t = (0 if m.group(1) == "F" else 1), m.group(2).strip().strip('"')
            if t:
                turns.append([s, t])
    bot = [t for s, t in turns if s == 1]
    if len(bot) < 3 or not turns or turns[0][0] != 0:
        return None
    if any(NEVER.search(t) or GROUPS.search(t.replace("gay", "")) or "suicid" in t.lower() or len(t.split()) > 25
           for _, t in turns) or stock_lines(turns, ("ur mom",) if mom else ()) > 1:
        return None
    if mom and (mom_lines(turns) != 1 or any(YO_MAMA.search(t.lower().replace("’", "'")) for _, t in turns)):
        return None
    if strict and not mom_fits(turns):
        return None
    if strict == 3 and any(I_HAD.search(t.lower().replace("’", "'")) for s, t in turns if s == 1):
        return None
    return turns


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    flags = sys.argv[2:]
    modes = (["topic", "argue"] if "both" in flags else ["argue"] if "argue" in flags else ["mom2"] if "mom2" in flags
             else ["mom3"] if "mom3" in flags else ["mom"] if "mom" in flags
             else ["topic"])
    B = next((int(a[6:]) for a in flags if a.startswith("batch=")), 24)
    bank = json.loads((ROOT / "data" / "style" / "roast_bank.json").read_text())
    tiktok = [x for x in style_lines() if len(x.split()) <= 15]
    tok, net = q4(WRITER)
    run = {}
    for m in modes:
        d = ROOT / "data" / "people" / {"argue": "rb_arg", "mom": "rb_mom", "mom2": "rb_mom2", "mom3": "rb_mom3"}.get(m, "rb")
        d.mkdir(exist_ok=True)
        part_f = d / "partial.jsonl"
        out = [json.loads(line)["turns"] for line in open(part_f)] if part_f.exists() else []
        out = [t for t in out if stock_lines(t, ("ur mom",) if m.startswith("mom") else ()) <= 1]
        run[m] = {"d": d, "part": part_f, "out": out, "tries": 0}
        print(f"{m}: resuming with {len(out)} conversations" if out else f"{m}: starting", flush=True)
    r = random.Random(23 + sum(len(v["out"]) for v in run.values()))
    while True:
        todo = [m for m in modes if len(run[m]["out"]) < n and run[m]["tries"] < 6 * n]
        if not todo:
            break
        m = min(todo, key=lambda k: len(run[k]["out"]))
        st = run[m]
        argue = m in ("argue", "mom2", "mom3") or (m == "mom" and st["tries"] // B % 2 == 0)
        prompt, topics = (ARGUE, OPENERS) if argue else (SYSTEM, TOPICS)
        prompt += {"mom": MOM, "mom2": MOM2, "mom3": MOM3}.get(m, "")
        pool = MOMS3 if m == "mom3" else MOMS
        ps = [chat(tok, prompt.format(emoji=" ".join(EMOJI), roasts=", ".join(f'"{x}"' for x in r.sample(bank, 3)),
                                      curses=", ".join(r.sample(CURSES, 5)), topic=r.choice(topics),
                                      moms=", ".join(f'"{x}"' for x in r.sample(pool, 2)),
                                      style=" / ".join(f'"{x}"' for x in r.sample(tiktok, 5))), "Write it.")
              for _ in range(B)]
        x = tok(ps, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            y = net.generate(**x, max_new_tokens=320 if argue else 260, do_sample=True, temperature=0.9, top_p=0.95,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        with open(st["part"], "a") as f:
            for o in tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True):
                t = parse(o, m.startswith("mom"), {"mom2": True, "mom3": 3}.get(m, False))
                if t:
                    st["out"].append(t)
                    f.write(json.dumps({"turns": t}, ensure_ascii=False) + chr(10))
        st["tries"] += B
        print(f"  {m}: {len(st['out'])}/{n} conversations ({st['tries']} written this run)", flush=True)
    for m in modes:
        d, out = run[m]["d"], run[m]["out"]
        r.shuffle(out)
        k = max(1, len(out) // 12)
        for name, part in (("val", out[:k]), ("train", out[k:])):
            (d / f"{name}.jsonl").write_text("".join(json.dumps({"turns": t}, ensure_ascii=False) + "\n" for t in part))
        print(f"{len(out)} conversations -> {d}")


if __name__ == "__main__":
    main()
