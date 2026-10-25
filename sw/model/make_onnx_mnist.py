"""Train a small MobileNet-style MNIST CNN in PyTorch, export it to ONNX and quantize it
with onnxruntime (QDQ, int8 per-channel weights): an example input for npuc.

    .venv/bin/python sw/model/make_onnx_mnist.py --out models/onnx

Writes mnist_cnn_float.onnx and mnist_cnn_int8.onnx, plus mnist_test.npz (1000 test images
for `npuc --images`). The network only uses what the NPU's board build supports (strided
convolutions instead of max pooling), and takes x = pixel / 255 so the quantized input
codes are the raw pixels.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist():
    path = Path.home() / ".keras" / "datasets" / "mnist.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, path)
    d = np.load(path)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def build():
    import torch.nn as nn

    def dw(c, s, p=1):
        return [nn.Conv2d(c, c, 3, s, p, groups=c), nn.ReLU()]

    def pw(i, o):
        return [nn.Conv2d(i, o, 1), nn.ReLU()]

    return nn.Sequential(
        nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),     # 28 -> 14
        *dw(16, 1), *pw(16, 32),
        *dw(32, 2), *pw(32, 32),                   # -> 7
        *dw(32, 2, 0), *pw(32, 64),                # -> 3 (3x3 global pool: depthwise unit)
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))


def train(model, x, y, epochs, seed=0):
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-2, total_steps=epochs * (len(x) // 128))
    xt = torch.from_numpy(x[:, None].astype(np.float32) / 255)
    yt = torch.from_numpy(y.astype(np.int64))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(len(x) // 128):
            idx = perm[i * 128:(i + 1) * 128]
            loss = torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}: loss {loss.item():.4f}")
    model.eval()


def ort_accuracy(path, x, y):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    pred = [np.argmax(s.run(None, {name: (img[None, None] / 255).astype(np.float32)})[0]) for img in x]
    return float(np.mean(np.array(pred) == y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/onnx"))
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    import torch
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)

    xtr, ytr, xte, yte = load_mnist()
    model = build()
    train(model, xtr, ytr, args.epochs)
    args.out.mkdir(parents=True, exist_ok=True)
    f32 = args.out / "mnist_cnn_float.onnx"
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), str(f32), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False)

    calib = xtr[np.random.default_rng(0).choice(len(xtr), 256, replace=False)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            img = next(self.it, None)
            return None if img is None else {"image": (img[None, None] / 255).astype(np.float32)}

    q8 = args.out / "mnist_cnn_int8.onnx"
    quantize_static(str(f32), str(q8), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    np.savez_compressed(args.out / "mnist_test.npz", images=xte[:1000, :, :, None], labels=yte[:1000])
    print(f"float accuracy {ort_accuracy(f32, xte[:1000], yte[:1000]):.4f}, "
          f"int8 (onnxruntime) {ort_accuracy(q8, xte[:1000], yte[:1000]):.4f} on 1000 test images -> {args.out}")


if __name__ == "__main__":
    main()
