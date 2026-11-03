import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from link import Link, checksum


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
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
    a, n = 0xE0000, int(sys.argv[2]) if len(sys.argv) > 2 else 65536
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
