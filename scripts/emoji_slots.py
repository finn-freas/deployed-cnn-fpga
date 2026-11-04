import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import tok as T

EMOJI_SLOTS = ["️", "🤣", "🏻", "💙", "☺", "😂", "😍", "🤩", "🎉", "😭", "😃", "🤪", "😆", "💕", "🫶", "💗", "❤", "🎂", "🙏", "⚽", "😊", "😳", "😡", "🥳", "🤨", "😋", "🙂", "😔", "👌", "😜", "😁", "🥰", "👍", "😘", "💀", "😢", "😞", "😉", "😅", "🇸", "🤔", "🇺", "🇵", "👋", "♀", "🙌", "🦅", "🤞", "🇯", "👨", "👀", "😈", "🙃", "😩", "😎", "😬", "😠", "😤", "🥴", "🌺", "🌼", "🙄", "🍣", "😚", "🤷", "🤬", "😫", "🤓", "🏈", "😏", "✈", "😛", "🇷", "😗", "🎓", "😵", "👾", "🌚", "😣", "🤦", "👿", "💦", "🍕", "💩", "🎾", "💰", "🍳", "🇮", "😴", "✅", "😕", "🌸", "🐶"]


def main():
    picked = EMOJI_SLOTS
    slots = T.fill_slots(picked)
    (T.TOK / "slots.json").write_text(json.dumps(slots, ensure_ascii=False))
    print(f"{len(picked)} emoji in slots")

    tok, base = T.load_chat(), T.load()
    assert tok.get_vocab_size() == T.VOCAB
    for tid, s, _ in slots:
        assert tok.token_to_id(s) == tid and tok.encode(s, add_special_tokens=False).ids == [tid], s
    for i in range(len(slots) + 4, T.VOCAB):
        assert tok.id_to_token(i) == base.id_to_token(i), i
    for s in ["omg 😭😭 no way", "ok 👍🏻", "lol 😂 see u", "hearing that tho"]:
        e = tok.encode(s, add_special_tokens=False)
        assert tok.decode(e.ids).strip() == s, (s, tok.decode(e.ids))
        print(f"{len(base.encode(s, add_special_tokens=False).ids)} -> {len(e.ids)} tokens, round trip ok")


if __name__ == "__main__":
    main()
