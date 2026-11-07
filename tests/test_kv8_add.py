import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import kv8_add


@pytest.mark.parametrize("argv", [[], ["a"], ["a", "b", "c", "d"]])
def test_wrong_argument_count_prints_usage(argv):
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args(argv)
    assert "usage" in str(e.value)


def test_missing_source_is_named(tmp_path):
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args([str(tmp_path / "pr_board.pkl"), str(tmp_path / "out")])
    assert "pr_board.pkl not found" in str(e.value)


def test_output_next_to_the_source_is_rejected(tmp_path):
    src = tmp_path / "pr_board.pkl"
    src.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args([str(src), str(tmp_path)])
    assert "somewhere else" in str(e.value)


def test_missing_training_data_is_named(tmp_path):
    src = tmp_path / "zz_nokey_board.pkl"
    src.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        kv8_add.parse_args([str(src), str(tmp_path / "out")])
    assert "train.jsonl not found" in str(e.value)


def test_key_defaults_to_the_file_name_prefix(tmp_path, monkeypatch):
    src = tmp_path / "zz_board.pkl"
    src.write_bytes(b"")
    data = tmp_path / "data" / "people" / "zz"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("")
    monkeypatch.setattr(kv8_add, "ROOT", tmp_path)
    assert kv8_add.parse_args([str(src), str(tmp_path / "out")]) == (src, tmp_path / "out", "zz")
    assert kv8_add.parse_args([str(src), str(tmp_path / "out"), "zz"])[2] == "zz"
