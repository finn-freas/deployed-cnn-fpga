import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from link import Link, checksum


USAGE = "usage: python scripts/sdram_diag.py [rounds [words]]  (rounds >= 0, words >= 4096)"


def parse_args(argv):
    try:
        rounds = int(argv[0]) if len(argv) > 0 else 20
        n = int(argv[1]) if len(argv) > 1 else 65536
    except ValueError:
        sys.exit(USAGE)
    if len(argv) > 2 or rounds < 0 or n < 4096:
        sys.exit(USAGE)
    return rounds, n


def main():
    rounds, n = parse_args(sys.argv[1:])
    lk = Link()
    print("ping", lk.ping())
    lk.send(b"D")
    r = lk.recv(3)
    d = int.from_bytes(r[1:3], "little")
    ok, pick = d >> 4, d & 15
    print(f"calibration: passing read latencies (first = CAS - 1) "
          f"{''.join('X' if ok >> t & 1 else '.' for t in range(4))}  pick {pick}")
    if not rounds:
        return
    rnd = random.Random(1)
    a = 0xE0000
    words = [rnd.getrandbits(64) for _ in range(n)]
    data = struct.pack(f"<{n}Q", *words)
    lk.write_bytes(a, data)
    want = checksum(data)
    bad = sum(lk.checksum(a, n) != want for _ in range(rounds))
    got = []
    for i in range(0, 4096, 256):
        got += lk.read(a + i, 256)
    werr = sum(g != w for g, w in zip(got, words))
    print(f"checksum fails {bad}/{rounds}, word errors {werr}/4096")


if __name__ == "__main__":
    main()
