import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import sdram_diag


def test_defaults():
    assert sdram_diag.parse_args([]) == (20, 65536)


def test_explicit_values():
    assert sdram_diag.parse_args(["0"]) == (0, 65536)
    assert sdram_diag.parse_args(["5", "4096"]) == (5, 4096)


@pytest.mark.parametrize("argv", [["x"], ["1", "y"], ["-1"], ["1", "100"], ["1", "4096", "3"]])
def test_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit) as e:
        sdram_diag.parse_args(argv)
    assert "usage" in str(e.value)
