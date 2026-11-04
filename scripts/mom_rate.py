import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import chat, tok as T
M = re.compile(r"\b(your|ur|yo|you.re) (mom|mama|momma|moms)\b")
INS = ["fuck you", "shut up", "ur so dumb", "ur trash bro", "i hate you", "ur just a bot", "stfu", "ur not funny",
       "go away", "ur annoying", "screw you", "ur mid", "loser", "ur a toaster", "fuck off"]
OK = ["whats up", "what should i eat", "i got a new phone", "im bored", "wanna hear a joke", "i failed my test",
      "my dog is sick", "what music do you like", "im going to the gym", "lol", "ok", "its raining",
      "i beat my brother at 2k", "my hair is fire today", "what time is it"]
tk = T.load_chat(); p = chat.Person("pr", sys.argv[1])
total = 0; ins_hits = 0; gaps = []
for s in range(1, 1 + int(sys.argv[2])):
    p.reset(); rng = chat.make_rng(s); last = None
    msgs = [x for pair in zip(INS, OK) for x in pair]
    import random; random.Random(s).shuffle(msgs); msgs = msgs[:30]
    for i, m in enumerate(msgs):
        texts, _, _ = chat.reply_best(p, tk, m, 0.6, 4, rng, n_best=6, budget=150)
        r = " / ".join(texts)
        if M.search(r.lower()):
            total += 1; ins_hits += m in INS
            if last is not None: gaps.append(i - last)
            last = i
            print(f"chat {s} msg {i}: F: {m} | B: {r}", flush=True)
print(f"{total} mom lines in {int(sys.argv[2])} chats of 30 ({ins_hits} after an insult), gaps {gaps}")
