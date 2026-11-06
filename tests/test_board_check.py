import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chat
from scripts import board_check


@pytest.fixture
def models(tmp_path, monkeypatch):
    (tmp_path / "pr_board.pkl").write_bytes(b"")
    monkeypatch.setattr(chat, "MODELS", tmp_path)
    return tmp_path


def test_defaults(models):
    assert board_check.parse_args([]) == ("pr", False, False)


def test_flags(models):
    assert board_check.parse_args(["pr", "--best", "--argue"]) == ("pr", True, True)


def test_unknown_option(models):
    with pytest.raises(SystemExit) as e:
        board_check.parse_args(["--bogus"])
    assert "unknown option: --bogus" in str(e.value)


def test_unknown_model(models):
    with pytest.raises(SystemExit) as e:
        board_check.parse_args(["zz"])
    assert "no model 'zz'" in str(e.value)


def test_too_many_models(models):
    with pytest.raises(SystemExit):
        board_check.parse_args(["pr", "pr"])
