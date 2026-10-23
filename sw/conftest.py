# Makes the `npu` package importable when running pytest from the repo root or sw/.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
