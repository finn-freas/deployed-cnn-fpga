import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import board as B
import pack as P
import tok as T
from finetune import encode_sessions


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    key = sys.argv[3] if len(sys.argv) > 3 else src.name.split("_")[0]
    assert out.resolve() != src.parent.resolve(), "write the new model somewhere else"
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
