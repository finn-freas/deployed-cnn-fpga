import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import board as B
import pack as P
import tok as T
from finetune import encode_sessions


USAGE = "usage: python scripts/kv8_add.py models/<key>_board.pkl new_models_dir [key]"


def parse_args(argv):
    if not 2 <= len(argv) <= 3:
        sys.exit(USAGE)
    src, out = Path(argv[0]), Path(argv[1])
    key = argv[2] if len(argv) > 2 else src.name.split("_")[0]
    if not src.is_file():
        sys.exit(f"{src} not found")
    if out.resolve() == src.parent.resolve():
        sys.exit("write the new model somewhere else than the directory of the source model")
    train = ROOT / "data" / "people" / key / "train.jsonl"
    if not train.is_file():
        sys.exit(f"{train} not found")
    return src, out, key


def main():
    src, out, key = parse_args(sys.argv[1:])
    p = pickle.loads(src.read_bytes())
    tr = encode_sessions(T.load_chat(), ROOT / "data" / "people" / key / "train.jsonl")
    B.kv8_calibrate(p, [i for i, _ in tr[:48]])
    print(f"{p['cfg'].n_layers} layers, kv8 shifts {p['kv8']}")
    img, _ = P.build(p)
    print(P.memory_report(img))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}_board.pkl").write_bytes(pickle.dumps(p))
    for f in src.parent.iterdir():
        if f.name != src.name and (f.name.startswith(key + "_") or f.name == "generic_unigram.npy"):
            link = out / f.name
            if not link.exists():
                link.symlink_to(f.resolve())


if __name__ == "__main__":
    main()
