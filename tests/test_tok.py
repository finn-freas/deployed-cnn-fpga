import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tok


TEXTS = ["hello there my friend", "what are you doing tonight", "nothing much just relaxing"] * 20


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=tok.SPECIAL, min_frequency=1,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t.train_from_iterator(TEXTS, tr)
    (tmp_path / "tok").mkdir()
    (tmp_path / "corpus").mkdir()
    t.save(str(tmp_path / "tok" / "tokenizer.json"))
    monkeypatch.setattr(tok, "TOK", tmp_path / "tok")
    monkeypatch.setattr(tok, "CORPUS", tmp_path / "corpus")
    monkeypatch.setattr(tok.load, "__defaults__", (tmp_path / "tok" / "tokenizer.json",))
    monkeypatch.setattr(tok, "CHAT_TOK", tmp_path / "tok" / "tokenizer_chat.json")
    return t


def test_special_ids():
    assert tok.SPECIAL[tok.PAD] == "<pad>"
    assert tok.SPECIAL[tok.BOS] == "<bos>"
    assert tok.SPECIAL[tok.YOU] == "<you>"
    assert tok.SPECIAL[tok.HIM] == "<him>"
    assert len(tok.SPECIAL) == 4 + tok.N_RESERVED


def test_encode_dialogue_speaker_markers(tiny):
    turns = [(0, "hello there"), (1, "my friend")]
    ids = tok.encode_dialogue(tiny, turns)
    a = tiny.encode("hello there", add_special_tokens=False).ids
    b = tiny.encode("my friend", add_special_tokens=False).ids
    assert ids == [tok.BOS, tok.YOU] + a + [tok.HIM] + b


def test_encode_dialogue_swap(tiny):
    turns = [(0, "hello there"), (1, "my friend")]
    plain = tok.encode_dialogue(tiny, turns)
    swapped = tok.encode_dialogue(tiny, turns, swap=True)
    assert len(plain) == len(swapped)
    assert plain[1] == tok.YOU and swapped[1] == tok.HIM


def test_turns_text_limit(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [{"turns": [[0, f"a{i}"], [1, f"b{i}"]]} for i in range(3)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert list(tok.turns_text(p)) == ["a0", "b0", "a1", "b1", "a2", "b2"]
    assert list(tok.turns_text(p, limit=2)) == ["a0", "b0", "a1", "b1"]


def test_fill_slots_assigns_reserved_ids(tiny):
    out = tok.fill_slots(["<emoji_a>", "<emoji_b>"])
    assert [(tid, s) for tid, s, _ in out] == [(4, "<emoji_a>"), (5, "<emoji_b>")]
    chat = tok.load_chat()
    assert chat.token_to_id("<emoji_a>") == 4
    assert chat.token_to_id("<emoji_b>") == 5
    assert chat.token_to_id("<r0>") is None
    assert chat.token_to_id("<r2>") == 6
    assert chat.encode("<emoji_a>", add_special_tokens=False).ids == [4]


def test_load_chat_falls_back_to_base(tiny):
    assert tok.load_chat().get_vocab() == tiny.get_vocab()


def test_encode_split_counts_and_layout(tiny):
    dialogues = [
        {"src": "a", "turns": [[0, "hello there"], [1, "my friend"]]},
        {"src": "b", "turns": [[1, "what are you doing tonight"]]},
        {"src": "a", "turns": [[0, "nothing much"], [1, "just relaxing"]]},
    ]
    (tok.CORPUS / "val.jsonl").write_text("\n".join(json.dumps(d) for d in dialogues) + "\n")
    n, stats = tok.encode_split(tiny, "val", np.random.default_rng(0))
    ids = np.fromfile(tok.TOK / "val.bin", np.uint16)
    assert n == len(ids)
    assert stats == {"a": 2, "b": 1}
    assert int((ids == tok.BOS).sum()) == 3
    assert ids[0] == tok.BOS
    assert int(((ids == tok.YOU) | (ids == tok.HIM)).sum()) == 5
