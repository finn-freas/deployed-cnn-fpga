import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tok as T
from scripts import unigram


def test_missing_token_file_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TOK", tmp_path)
    monkeypatch.setattr(sys, "argv", ["unigram.py", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as e:
        unigram.main()
    assert "val.bin not found" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_writes_a_normalised_distribution(tmp_path, monkeypatch):
    np.array([1, 1, 2, 5], np.uint16).tofile(tmp_path / "val.bin")
    monkeypatch.setattr(T, "TOK", tmp_path)
    monkeypatch.setattr(sys, "argv", ["unigram.py", str(tmp_path / "out")])
    unigram.main()
    p = np.load(tmp_path / "out" / "generic_unigram.npy")
    assert p.shape == (T.VOCAB,)
    assert abs(p.sum() - 1) < 1e-9
    assert p[1] > p[2] > p[0] > 0
