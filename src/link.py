import glob
import os
import struct
import sys
import termios
import time
from pathlib import Path

PORT = next(iter(sorted(glob.glob("/dev/serial/by-id/*SIPEED*if01*"))), "/dev/ttyUSB1")
BAUD = 1_000_000
CHUNK = 4096


class Link:
    def __init__(self, port=PORT, baud=BAUD):
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
        a = termios.tcgetattr(self.fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = getattr(termios, f"B{baud}")
        a[6][termios.VMIN], a[6][termios.VTIME] = 0, 1
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.drain()

    def drain(self, quiet=0.2):
        end = time.time() + quiet
        while time.time() < end:
            if os.read(self.fd, 256):
                end = time.time() + quiet

    def send(self, data):
        view = memoryview(data)
        while view:
            n = os.write(self.fd, view)
            view = view[n:]

    def recv(self, n, timeout=5.0):
        out = bytearray()
        end = time.time() + timeout
        while len(out) < n:
            chunk = os.read(self.fd, n - len(out))
            if chunk:
                out += chunk
                end = time.time() + timeout
            elif time.time() > end:
                raise TimeoutError(f"board sent {len(out)} of {n} bytes: {bytes(out[:16])!r}")
        return bytes(out)

    @staticmethod
    def _hdr(cmd, addr, n):
        return cmd + addr.to_bytes(3, "little") + n.to_bytes(3, "little")

    def ping(self, tries=2):
        for i in range(tries):
            self.send(b"P")
            r = self.recv(3)
            if r[0:1] == b"K":
                break
            self.drain()
        else:
            raise IOError(f"bad ping reply {r!r}")
        return {"version": r[1], "sdram_ready": bool(r[2] & 1), "cal_err": bool(r[2] & 2)}

    def write(self, addr, words):
        self.send(self._hdr(b"W", addr, len(words)) + struct.pack(f"<{len(words)}Q", *words))
        if self.recv(1, timeout=10) != b"w":
            raise IOError("write not acknowledged")

    def write_bytes(self, addr, data, progress=None):
        data = data + bytes(-len(data) % 8)
        n = len(data) // 8
        for i in range(0, n, CHUNK):
            k = min(CHUNK, n - i)
            self.send(self._hdr(b"W", addr + i, k) + data[8 * i:8 * (i + k)])
            if self.recv(1, timeout=10) != b"w":
                raise IOError(f"write at word {addr + i:#x} not acknowledged")
            if progress:
                progress(i + k, n)

    def read(self, addr, n):
        self.send(self._hdr(b"R", addr, n))
        return list(struct.unpack(f"<{n}Q", self.recv(8 * n)))

    def checksum(self, addr, n):
        self.send(self._hdr(b"C", addr, n))
        return struct.unpack("<I", self.recv(4, timeout=10))[0]

    def _ack(self, cmd, want, timeout=5.0):
        self.send(cmd)
        r = self.recv(1, timeout)
        if r != want:
            raise IOError(f"{cmd[:1]!r}: expected {want!r}, got {r!r}")

    def init_engine(self):
        self._ack(b"I", b"i")

    def new_conversation(self):
        self._ack(b"Z", b"z")

    def seed(self, state):
        self._ack(b"S" + struct.pack("<I", state & 0xFFFFFFFF), b"s")

    def settings(self, inv_t, cut_q, gain_q):
        self._ack(b"G" + struct.pack("<HHH", inv_t, cut_q, min(gain_q, 0xFFFF)), b"g")

    def emoji_table(self, mask, bias):
        body = b"".join(struct.pack("<Bh", int(bool(mask[t])), int(bias[t])) for t in range(128))
        self._ack(b"E" + body, b"e")

    def feed(self, tok):
        self.send(struct.pack("<cBHH", b"F", 0, tok, 0))
        r = self.recv(3)
        if r[0:1] != b"a":
            raise IOError(f"feed: bad reply {r!r}")
        return struct.unpack("<H", r[1:3])[0]

    def step(self, tok):
        return self._f(1, tok, 0)

    def score_after(self, tok, nxt):
        return self._f(2, tok, nxt)

    def _f(self, flags, tok, nxt):
        self.send(struct.pack("<cBHH", b"F", flags, tok, nxt))
        r = self.recv(18)
        if r[0:1] != b"f":
            raise IOError(f"step: bad reply {r!r}")
        return (struct.unpack("<H", r[1:3])[0], int.from_bytes(r[3:9], "little"),
                int.from_bytes(r[9:15], "little"), int.from_bytes(r[15:18], "little", signed=True))

    def window(self, base):
        self._ack(b"X" + struct.pack("<H", base), b"x")

    def jump(self, pos):
        self._ack(b"J" + struct.pack("<H", pos), b"j")

    def best(self, clear=False, end=False, keep=False):
        self.send(b"Y" + bytes([int(clear) | int(end) << 1 | int(keep) << 2]))
        r = self.recv(7)
        if r[0:1] != b"y":
            raise IOError(f"best: bad reply {r!r}")
        return r[1], struct.unpack("<i", r[2:6])[0], r[6]


def checksum(data):
    data = data + bytes(-len(data) % 8)
    return sum(struct.unpack(f"<{len(data) // 4}I", data)) & 0xFFFFFFFF


USAGE = "usage: python src/link.py ping | load models/pr_board"


def read_image(argv):
    if len(argv) < 3:
        sys.exit(USAGE)
    path = Path(argv[2]).with_suffix(".sdram.bin")
    if not path.is_file():
        sys.exit(f"{path} not found; build it with pack.py first")
    return path.read_bytes()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ping"
    if cmd not in ("ping", "load"):
        sys.exit(USAGE)
    img = read_image(sys.argv) if cmd == "load" else None
    link = Link()
    if cmd == "ping":
        print(link.ping())
    elif cmd == "load":
        t = time.time()

        def progress(done, n):
            print(f"\r  {done * 8 / 2 ** 20:.2f} / {n * 8 / 2 ** 20:.2f} MB", end="", flush=True)

        link.write_bytes(0, img, progress)
        dt = time.time() - t
        want, got = checksum(img), link.checksum(0, -(-len(img) // 8))
        print(f"\nuploaded {len(img) / 2 ** 20:.2f} MB in {dt:.1f}s ({len(img) / dt / 1e3:.0f} KB/s), "
              f"checksum {'OK' if want == got else f'MISMATCH {got:#x} != {want:#x}'}")


if __name__ == "__main__":
    main()
