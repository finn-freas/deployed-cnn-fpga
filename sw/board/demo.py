import argparse
import sys
import time

import numpy as np

SIZE = 96
CLK_HZ = 54_000_000

def to_input(bgr) -> np.ndarray:
    import cv2
    h, w = bgr.shape[:2]
    s = min(h, w)
    crop = bgr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def infer(port, rgb: np.ndarray):
    t0 = time.perf_counter()
    port.write(b"I" + np.ascontiguousarray(rgb, np.uint8).tobytes())
    reply = port.read(7)
    if len(reply) != 7 or reply[0] != 0xA5:
        raise IOError(f"bad reply from the board: {reply!r}")
    cycles = int.from_bytes(reply[3:7], "little")
    score = int.from_bytes(reply[2:3], "little", signed=True)
    return reply[1] == 1, score, cycles / CLK_HZ * 1e3, (time.perf_counter() - t0) * 1e3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="the board's USB serial port (e.g. COM5, /dev/ttyUSB1)")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("images", nargs="*")
    args = ap.parse_args()

    import cv2
    import serial
    try:
        port = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as e:
        sys.exit(f"cannot open {args.port}: {e}")

    if args.images:
        for path in args.images:
            img = cv2.imread(path)
            if img is None:
                print(f"{path}: cannot read", file=sys.stderr)
                continue
            person, score, lat, rt = infer(port, to_input(img))
            print(f"{path}: {'PERSON' if person else 'no person'} (score {score:+d}), "
                  f"NPU {lat:.2f} ms, round trip {rt:.0f} ms")
        return

    cam = cv2.VideoCapture(args.camera)
    print("q to quit")
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        person, score, lat, rt = infer(port, to_input(frame))
        label = f"{'PERSON' if person else 'no person'} {score:+d}  NPU {lat:.1f} ms  ({1000 / rt:.1f} fps)"
        colour = (60, 220, 60) if person else (60, 60, 230)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.imshow("Tang Nano 20K NPU", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
