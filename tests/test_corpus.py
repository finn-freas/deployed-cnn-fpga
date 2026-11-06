import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import corpus


def test_clean_collapses_whitespace_and_zero_width():
    assert corpus.clean("  a \n\t b​ c ") == "a b c"


def test_clean_rejects_empty_and_long():
    assert corpus.clean("   ") is None
    assert corpus.clean("x" * corpus.MAX_CHARS) == "x" * corpus.MAX_CHARS
    assert corpus.clean("x" * (corpus.MAX_CHARS + 1)) is None


@pytest.mark.parametrize("text", ["see https://a.b", "www.x.com", "[a](b)", "&gt; quote", "ask /u/bob", "in r/test", "u/bob"])
def test_url_pattern_matches(text):
    assert corpus.URL.search(text)


def test_url_pattern_ignores_plain_text():
    assert not corpus.URL.search("our summer was great")


def test_textify_lowercases_and_is_deterministic():
    a = corpus.textify("Hello, It's Me.", random.Random(3))
    b = corpus.textify("Hello, It's Me.", random.Random(3))
    assert a == b
    assert a == a.lower()


def test_textify_only_removes_trailing_period_and_apostrophes():
    for seed in range(20):
        out = corpus.textify("Don't stop, ok.", random.Random(seed))
        assert set(out) <= set("don't stop, ok.")
        assert out.startswith("dont" if "'" not in out else "don't")


def write_soda(path, rows):
    pq.write_table(pa.table({"dialogue": [r[0] for r in rows], "speakers": [r[1] for r in rows]}), path)


def test_soda_yields_two_speaker_dialogues(tmp_path):
    p = tmp_path / "s.parquet"
    write_soda(p, [
        (["Hi", "Hello", "Bye"], ["A", "B", "A"]),
        (["a", "b", "c"], ["A", "B", "C"]),
        (["only one"], ["A"]),
        (["fine", "x" * 400], ["A", "B"]),
    ])
    out = list(corpus.soda(str(p), random.Random(0), styled=0))
    assert out == [{"src": "soda", "turns": [[0, "Hi"], [1, "Hello"], [0, "Bye"]]}]


def test_soda_styled_lowercases(tmp_path):
    p = tmp_path / "s.parquet"
    write_soda(p, [(["Hi There", "Oh Yes"], ["A", "B"])])
    out = list(corpus.soda(str(p), random.Random(0), styled=1.0))
    assert [t for _, t in out[0]["turns"]] == ["hi there", "oh yes"]


def write_reddit(path, rows):
    cols = list(zip(*rows))
    pq.write_table(pa.table(dict(zip(["id", "parent_id", "author", "body"], cols))), path)


def test_reddit_chain_alternating_authors(tmp_path):
    p = tmp_path / "r.parquet"
    write_reddit(p, [
        ("a", "t3_post", "u1", "first"),
        ("b", "t1_a", "u2", "second"),
        ("c", "t1_b", "u1", "third"),
        ("d", "t1_c", "u3", "fourth"),
    ])
    out = list(corpus.reddit_chains(str(p)))
    assert out == [{"src": "reddit", "turns": [[0, "first"], [1, "second"], [0, "third"]]}]


def test_reddit_skips_deleted_urls_and_short_chains(tmp_path):
    p = tmp_path / "r.parquet"
    write_reddit(p, [
        ("a", "t3_post", "u1", "alone"),
        ("b", "t3_post", "[deleted]", "gone"),
        ("c", "t1_b", "u2", "reply to deleted"),
        ("d", "t3_post", "u1", "see https://x.y"),
        ("e", "t1_d", "u2", "reply to link"),
        ("f", "t3_post", "u4", "[removed]"),
    ])
    assert list(corpus.reddit_chains(str(p))) == []


def test_reddit_same_author_does_not_form_chain(tmp_path):
    p = tmp_path / "r.parquet"
    write_reddit(p, [
        ("a", "t3_post", "u1", "one"),
        ("b", "t1_a", "u1", "two"),
    ])
    assert list(corpus.reddit_chains(str(p))) == []


def test_reddit_respects_max_len(tmp_path):
    p = tmp_path / "r.parquet"
    rows = [("c0", "t3_post", "u1", "m0")]
    for i in range(1, 6):
        rows.append((f"c{i}", f"t1_c{i - 1}", "u2" if i % 2 else "u1", f"m{i}"))
    write_reddit(p, rows)
    out = list(corpus.reddit_chains(str(p), max_len=4))
    assert len(out) == 1 and len(out[0]["turns"]) == 4
