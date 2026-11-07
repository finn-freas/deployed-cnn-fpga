import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import copy_check


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(copy_check, "ROOT", tmp_path)
    (tmp_path / "data" / "style").mkdir(parents=True)
    (tmp_path / "data" / "style" / "tiktok_roasts.json").write_text(json.dumps(["one two three four five"]))
    return tmp_path


def people(root, key, convs):
    d = root / "data" / "people" / key
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (d / f"{split}.jsonl").write_text("".join(json.dumps({"turns": c}) + "\n" for c in convs))
    return d


@pytest.mark.parametrize("argv", [[], ["a", "b", "c"], ["a", "keep"], [","]])
def test_bad_arguments_print_usage(root, argv):
    with pytest.raises(SystemExit) as e:
        copy_check.parse_args(argv)
    assert "usage" in str(e.value)


def test_missing_style_file_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(copy_check, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        copy_check.parse_args(["a"])
    assert "tiktok_roasts.json not found" in str(e.value)


def test_missing_split_is_named(root):
    with pytest.raises(SystemExit) as e:
        copy_check.parse_args(["zz"])
    assert "train.jsonl not found" in str(e.value)


def test_valid_arguments(root):
    people(root, "a", [])
    people(root, "b", [])
    sources, drop, _ = copy_check.parse_args(["a,b", "drop"])
    assert sources == ["a", "b"] and drop


def test_main_reports_copies_and_drops_them(root, monkeypatch, capsys):
    convs = [
        [[0, "hi"], [1, "ok then"], [0, "again"], [1, "fine"]],
        [[0, "hey"], [1, "One two three four five!"], [0, "x"], [1, "y"]],
    ]
    d = people(root, "a", convs)
    monkeypatch.setattr(sys, "argv", ["copy_check.py", "a", "drop"])
    copy_check.main()
    assert "1 of 4 bot lines copy a style example, in 1 of 2 conversations" in capsys.readouterr().out
    kept = [json.loads(line)["turns"] for line in open(d / "train.jsonl")]
    assert kept == [convs[0]]
