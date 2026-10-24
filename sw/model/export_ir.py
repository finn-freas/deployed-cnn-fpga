"""Export a fully-INT8 TFLite model to a framework-free IR, plus golden tensors.

Run with the TensorFlow environment:
    .venv-tf/bin/python sw/model/export_ir.py models/mnist_int8.tflite build/model --golden 1000

Writes:
    <out>/ir.json      layer list: op, shapes, conv/pool geometry, quant params (no arrays)
    <out>/ir.npz       weights (int8), biases (int32), per-channel weight scales (float32)
    <out>/golden.npz   'images' (uint8), 'labels', 'input' (int8), and every layer's output
                       tensor from the TFLite interpreter using reference kernels (BUILTIN_REF)

Everything downstream (compiler, NumPy simulator, RTL tests) reads only these files.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as sfb

OPNAMES = {v: k for k, v in vars(sfb.BuiltinOperator).items() if not k.startswith("_")}
SHAPE_OPS = {"SHAPE", "STRIDED_SLICE", "PACK"}  # flatten bookkeeping on int32 shape tensors
ACT = {0: "none", 1: "relu", 3: "relu6"}


def same_pad(in_size, k, stride):
    out = -(-in_size // stride)
    total = max((out - 1) * stride + k - in_size, 0)
    return out, total // 2


def tensor_array(model, t):
    data = model.buffers[t.buffer].data
    dtype = {9: np.int8, 2: np.int32, 0: np.float32}[t.type]
    return np.frombuffer(bytes(data), dtype=dtype).reshape(t.shape)


def quant(t):
    q = t.quantization
    return (np.asarray(q.scale, dtype=np.float32), np.asarray(q.zeroPoint, dtype=np.int64))


def export(tflite_path: Path):
    buf = tflite_path.read_bytes()
    m = sfb.ModelT.InitFromObj(sfb.Model.GetRootAsModel(buf, 0))
    g = m.subgraphs[0]
    T = g.tensors

    def shape_hwc(ti):
        return [int(d) for d in T[ti].shape[1:]]  # drop batch dim

    in_scale, in_zp = quant(T[g.inputs[0]])
    ir = {
        "source": tflite_path.name,
        "input": {"tensor": int(g.inputs[0]), "shape": shape_hwc(g.inputs[0]),
                  "scale": float(in_scale[0]), "zp": int(in_zp[0])},
        "output": {"tensor": int(g.outputs[0])},
        "layers": [],
    }
    arrays = {}
    alias = {}  # reshape output -> input tensor (layout is already NHWC-flattened)

    for op in g.operators:
        code = m.operatorCodes[op.opcodeIndex]
        name = OPNAMES[max(code.builtinCode, code.deprecatedBuiltinCode)]
        opt = op.builtinOptions
        ins = [int(i) for i in op.inputs]
        out = int(op.outputs[0])
        if name in SHAPE_OPS:
            continue
        if name == "RESHAPE":
            alias[out] = alias.get(ins[0], ins[0])
            continue
        if name == "SOFTMAX":
            # Monotonic, so argmax is unchanged: the NPU stops at the logits. Only valid as
            # the final op; recorded so reports can say scores were computed off-device.
            assert out == g.outputs[0], "SOFTMAX is only supported as the final op"
            ir["output"]["softmax_dropped"] = True
            continue
        src = alias.get(ins[0], ins[0])
        s_in, z_in = quant(T[src])
        s_out, z_out = quant(T[out])
        L = {"name": f"L{len(ir['layers'])}_{name.lower()}", "in_tensor": src, "out_tensor": out,
             "in_shape": shape_hwc(src), "out_shape": shape_hwc(out),
             "in_scale": float(s_in[0]), "in_zp": int(z_in[0]),
             "out_scale": float(s_out[0]), "out_zp": int(z_out[0])}

        if name in ("CONV_2D", "FULLY_CONNECTED"):
            w = tensor_array(m, T[ins[1]])
            b = tensor_array(m, T[ins[2]])
            w_scale, w_zp = quant(T[ins[1]])
            assert np.all(w_zp == 0), "weights must be symmetric"
            if name == "CONV_2D":
                assert opt.dilationHFactor == 1 and opt.dilationWFactor == 1
                kh, kw = w.shape[1:3]
                H, W, _ = L["in_shape"]
                if opt.padding == 0:  # SAME
                    _, pt = same_pad(H, kh, opt.strideH)
                    _, pl = same_pad(W, kw, opt.strideW)
                else:
                    pt = pl = 0
                L.update(op="conv", kh=int(kh), kw=int(kw), stride=int(opt.strideH),
                         pad=int(pt), act=ACT[opt.fusedActivationFunction])
                assert opt.strideH == opt.strideW and pt == pl
            else:
                # FC over an HxWxC input == conv with kernel HxW, VALID padding.
                H, W, C = L["in_shape"]
                w = w.reshape(w.shape[0], H, W, C)
                L.update(op="conv", kh=H, kw=W, stride=1, pad=0,
                         act=ACT[opt.fusedActivationFunction], from_fc=True)
                L["out_shape"] = [1, 1, int(w.shape[0])]
            arrays[L["name"] + ".w"] = w.astype(np.int8)        # [O, kh, kw, I]
            arrays[L["name"] + ".b"] = b.astype(np.int32)       # [O]
            arrays[L["name"] + ".w_scale"] = np.broadcast_to(w_scale, (w.shape[0],)).astype(np.float32)
        elif name == "DEPTHWISE_CONV_2D":
            w = tensor_array(m, T[ins[1]])                       # [1, kh, kw, C*mult]
            b = tensor_array(m, T[ins[2]])
            w_scale, w_zp = quant(T[ins[1]])
            assert np.all(w_zp == 0), "weights must be symmetric"
            assert opt.depthMultiplier == 1, "depth multiplier must be 1"
            assert opt.dilationHFactor == 1 and opt.dilationWFactor == 1
            kh, kw = w.shape[1:3]
            H, W, C = L["in_shape"]
            assert kh == kw and opt.strideH == opt.strideW
            pt = same_pad(H, kh, opt.strideH)[1] if opt.padding == 0 else 0
            L.update(op="dwconv", kh=int(kh), kw=int(kw), stride=int(opt.strideH), pad=int(pt),
                     act=ACT[opt.fusedActivationFunction])
            arrays[L["name"] + ".w"] = w[0].transpose(2, 0, 1).astype(np.int8)   # [C, kh, kw]
            arrays[L["name"] + ".b"] = b.astype(np.int32)
            arrays[L["name"] + ".w_scale"] = np.broadcast_to(w_scale, (C,)).astype(np.float32)
        elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            assert s_in[0] == s_out[0] and z_in[0] == z_out[0], "pooling must preserve quantization"
            assert opt.filterHeight == opt.filterWidth and opt.strideH == opt.strideW
            k = int(opt.filterHeight)
            pt = same_pad(L["in_shape"][0], k, opt.strideH)[1] if opt.padding == 0 else 0
            L.update(op="maxpool" if name == "MAX_POOL_2D" else "avgpool", k=k,
                     stride=int(opt.strideH), pad=int(pt), act=ACT[opt.fusedActivationFunction])
        else:
            raise NotImplementedError(f"op {name} is not supported by the NPU")
        ir["layers"].append(L)

    return buf, ir, arrays


def load_images(spec: str, n: int):
    """'mnist' -> MNIST test set; otherwise an .npz with 'images' (uint8 NHWC) and 'labels'."""
    if spec == "mnist":
        (_, _), (xt, yt) = tf.keras.datasets.mnist.load_data()
        return xt[:n, :, :, None].astype(np.uint8), yt[:n].astype(np.int64)
    d = np.load(spec)
    return d["images"][:n].astype(np.uint8), d["labels"][:n].astype(np.int64)


def golden(buf: bytes, ir: dict, images: np.ndarray, labels: np.ndarray):
    # Input quantization is scale=1/255, zp=-128, so int8 = uint8 - 128 exactly; this is
    # what the NPU's packed-input CONV assumes.
    assert ir["input"]["zp"] == -128 and abs(ir["input"]["scale"] - 1 / 255) < 1e-9
    assert list(images.shape[1:]) == ir["input"]["shape"], (images.shape, ir["input"]["shape"])
    x_int8 = (images.astype(np.int16) - 128).astype(np.int8)

    itp = tf.lite.Interpreter(
        model_content=buf,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True)
    itp.allocate_tensors()
    outs = {L["name"]: [] for L in ir["layers"]}
    for img in x_int8:
        itp.set_tensor(ir["input"]["tensor"], img[None])
        itp.invoke()
        for L in ir["layers"]:
            outs[L["name"]].append(itp.get_tensor(L["out_tensor"])[0].reshape(L["out_shape"]).copy())
    res = {"images": images, "labels": labels, "input": x_int8}
    res.update({k: np.stack(v) for k, v in outs.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--golden", type=int, default=1000, help="number of test images to dump")
    ap.add_argument("--images", default="mnist", help="'mnist' or an .npz with images/labels")
    args = ap.parse_args()

    buf, ir, arrays = export(args.tflite)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ir.json").write_text(json.dumps(ir, indent=2))
    np.savez(args.out / "ir.npz", **arrays)
    images, labels = load_images(args.images, args.golden)
    gold = golden(buf, ir, images, labels)
    np.savez_compressed(args.out / "golden.npz", **gold)

    final = gold[ir["layers"][-1]["name"]].reshape(len(labels), -1)
    acc = float(np.mean(final.argmax(1) == gold["labels"]))
    for L in ir["layers"]:
        print(f"{L['name']:28s} {str(L['in_shape']):14s} -> {str(L['out_shape']):12s} "
              f"zp {L['in_zp']:4d}->{L['out_zp']:4d}")
    print(f"golden: {len(labels)} images, TFLite-ref accuracy {acc:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
