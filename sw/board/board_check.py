import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from npu.board import boot_words, image_sum  # noqa: E402
from npu.compiler import compile_ir, load_ir  # noqa: E402

NPU_HZ = 54_000_000
FAILED = []

def report(name, ok, detail="", hint=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
        if hint:
            print("       -> " + hint.replace("\n", "\n          "))
    return ok

def read_line(port, timeout):
    end = time.monotonic() + timeout
    buf = bytearray()
    while time.monotonic() < end:
        b = port.read(1)
        if not b:
            continue
        if b == b"\n":
            text = buf.decode(errors="replace").strip()
            if text:
                return text
            buf.clear()
        else:
            buf += b
    return buf.decode(errors="replace").strip()

def read_exact(port, n, timeout):
    end = time.monotonic() + timeout
    buf = b""
    while len(buf) < n and time.monotonic() < end:
        buf += port.read(n - len(buf))
    return buf

def ask(port, cmd: bytes, timeout=1.0):
    port.reset_input_buffer()
    port.write(cmd)
    return read_line(port, timeout)

def dq_lines(got: int, exp: int) -> str:
    diff = got ^ exp
    lines = sorted({b % 32 for b in range(64) if diff >> b & 1})
    if len(lines) > 24:
        return "almost every bit"
    return ", ".join(f"DQ{n}" for n in lines)

def expected_flash(model_dir: Path):
    prog = compile_ir(*load_ir(model_dir))
    m = prog.meta["input"]
    header = boot_words(prog, np.zeros((m["h"], m["w"], m["c"]), np.uint8))[0]
    ext = prog.ext_words()
    return prog, header, len(ext), image_sum(ext)

def check_bringup(port, model_dir, extra=0.0):
    r = ask(port, b"p", 1 + extra)
    if not report("UART", r == "pong", f"ping -> {r!r}" if r else "no reply",
                  "nothing came back: wrong serial port, the bring-up bitstream not loaded (LED 0 "
                  "should blink), or the UART pins (69 TX, 70 RX)" if not r else
                  "garbled reply: baud rate (1,000,000) or the 54 MHz PLL (LED 1 should blink)"):
        return

    r = ask(port, b"s", 1 + extra)
    m = re.fullmatch(r"SDRAM: ready, read latency CAS-1\+([0-3])", r)
    if not report("SDRAM calibration", bool(m), r,
                  "the controller wrote a pattern and no read latency returned it: check the SDRAM\n"
                  "pin constraints (IO names in tangnano20k.cst) and the clock phase (gw_sdram_clk)"):
        return
    if m[1] in ("0", "3"):
        print(f"       (latency at the edge of the calibrated range, CAS-1+{m[1]}: works, but worth noting)")

    r = ask(port, b"m", 10 + extra)
    m = re.fullmatch(r"mem: FAIL, 0x([0-9a-f]{8}) errors, first @([0-9a-f]{5}) got ([0-9a-f]{16}) exp ([0-9a-f]{16})", r)
    if m:
        n, addr, got, exp = int(m[1], 16), int(m[2], 16), int(m[3], 16), int(m[4], 16)
        report("SDRAM test", False, f"{n} errors, first at word 0x{addr:05x}: got {got:016x}, expected {exp:016x}",
               f"differing lines: {dq_lines(got, exp)}. A few lines -> those DQ pins; every word ->\n"
               "read timing (latency/clock phase); some addresses only -> address/bank pins")
    else:
        report("SDRAM test", r.startswith("mem: PASS"), r or "no reply (a hang during the test)")

    r = ask(port, b"f", 10 + extra)
    if r.startswith("flash: no model image"):
        report("flash", False, r, "no model in flash (or the flash pins don't work): write it with\n"
               "  cd sw && python -m npu.board ../models/ir/vww --image photo.jpg --program")
        return
    m = re.fullmatch(r"flash: image found, header ([0-9a-f]{16}), 0x([0-9a-f]{5}) words, sum ([0-9a-f]{8})", r)
    if not report("flash read", bool(m), r or "no reply"):
        return
    if model_dir:
        _, header, n, s = expected_flash(model_dir)
        same = (int(m[1], 16), int(m[2], 16), int(m[3], 16)) == (header, n, s)
        report(f"flash holds {model_dir}", same,
               f"{int(m[2], 16)} words, sum {m[3]}" + ("" if same else f" (expected {n} words, sum {s:08x})"),
               "the flash holds another model or a corrupted write: write it again (with --verify)")

def check_npu(port, model_dir, n_images, extra_timeout, seed=0):
    from npu.sim import NpuSim
    prog = compile_ir(*load_ir(model_dir))
    m = prog.meta["input"]
    rng = np.random.default_rng(seed)
    times = []
    for i in range(n_images):
        img = rng.integers(0, 256, (m["h"], m["w"], m["c"])).astype(np.uint8)
        sim = NpuSim.from_program(prog)
        cls, score = sim.run(img)
        port.reset_input_buffer()
        port.write(b"I" + img.tobytes())
        reply = read_exact(port, 7, (img.size + 1) * 10 / port.baudrate + extra_timeout)
        if len(reply) != 7 or reply[0] != 0xA5:
            report(f"inference {i}", False, f"reply {reply!r}",
                   "no answer: is npu_board_top loaded and booted (LED 3)? LED 4 = boot error (flash)")
            return
        b_cls, b_score = reply[1], int(np.int8(reply[2]))
        cycles = int.from_bytes(reply[3:7], "little")
        times.append(cycles / NPU_HZ * 1e3)
        report(f"inference {i}", (b_cls, b_score) == (cls, score),
               f"class {b_cls}, score {b_score:+d}, {cycles:,} cycles = {times[-1]:.2f} ms"
               + ("" if (b_cls, b_score) == (cls, score) else f"; simulator: class {cls}, score {score:+d}"),
               "the board differs from the simulator: the flash may hold another model\n"
               "(board_check.py with the bring-up bitstream and --model checks it), or SDRAM reads are\n"
               "marginal (run the bring-up SDRAM test several times)")
    if times:
        print(f"       NPU latency {np.mean(times):.2f} ms per image at 54 MHz")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True, help="the board's serial port (COM5, /dev/ttyUSB1, ...)")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--model", type=Path, help="model IR directory (models/ir/vww, or an npuc output dir)")
    ap.add_argument("--npu", action="store_true", help="the NPU bitstream is loaded: run inferences")
    ap.add_argument("--images", type=int, default=5, help="--npu: random images to compare")
    ap.add_argument("--timeout", type=float, default=1.0, help="extra seconds to wait for replies")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("needs pyserial: pip install pyserial")
    with serial.Serial(args.port, args.baud, timeout=0.05) as port:
        if args.npu:
            if not args.model:
                sys.exit("--npu needs --model (the model that is in the flash)")
            check_npu(port, args.model, args.images, args.timeout)
        else:
            check_bringup(port, args.model, args.timeout)
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: " + ", ".join(FAILED))
        sys.exit(1)
    print("\nall checks passed" + ("" if args.npu else
          ": next, load npu_board_top (make -C fpga/tangnano20k TOP=npu_board_top FREQ_MHZ=54 prog)\n"
          "and run this script with --npu --model <the model in flash>"))

if __name__ == "__main__":
    main()
