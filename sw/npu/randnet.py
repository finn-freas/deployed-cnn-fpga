"""Random small int8 networks in the exported-IR format, for corner-case tests.

random_ir(seed) builds a chain that starts with a packed-input CONV and mixes CONV,
DWCONV (k 1..3, stride 1/2, valid or same padding), MAXPOOL (possibly padded) and 3x3
AVGPOOL with odd widths, then reduces to 1x1 and ends with an FC (single-rounding CONV). Weights, biases and scales
are random, with output scales set so values mostly land inside int8 rather than all
saturating. The NPU simulator is the reference for such programs (it is checked against
TFLite on the real models); the RTL test compares the two layer by layer.
"""

import math

import numpy as np



def _out_and_pad(size, k, stride, same):
    if same:
        out = -(-size // stride)
        pad = max((out - 1) * stride + k - size, 0) // 2
    else:
        out = (size - k) // stride + 1
        pad = 0
    return out, pad


def random_ir(seed: int, max_hw: int = 20):
    rng = np.random.default_rng(seed)
    H = W = int(rng.integers(6, max_hw + 1))
    C = int(rng.integers(1, 5))
    arrays = {}
    layers = []
    cur = {"shape": [H, W, C], "scale": 1 / 255, "zp": -128}

    def new_q(k_eff):
        # output scale ~ |acc| spread so requantized values use the int8 range
        return float(cur["scale"] * 0.02 * math.sqrt(k_eff) * 60 * rng.uniform(0.5, 2.0))

    def add_conv(name, k, stride, same, oc, fc=False):
        h, w, ic = cur["shape"]
        oh, ph = _out_and_pad(h, k, stride, same)
        ow, _ = _out_and_pad(w, k, stride, same)
        arrays[name + ".w"] = rng.integers(-127, 128, (oc, k, k, ic)).astype(np.int8)
        arrays[name + ".b"] = rng.integers(-3000, 3000, oc).astype(np.int32)
        arrays[name + ".w_scale"] = (0.02 * rng.uniform(0.5, 1.5, oc)).astype(np.float32)
        out_scale = new_q(k * k * ic) / (8 if fc else 1)
        L = {"name": name, "op": "conv", "in_shape": [h, w, ic], "out_shape": [oh, ow, oc],
             "in_scale": cur["scale"], "in_zp": cur["zp"], "out_scale": out_scale,
             "out_zp": int(rng.integers(-128, 0)), "kh": k, "kw": k, "stride": stride,
             "pad": ph, "act": "none" if fc else str(rng.choice(["relu", "relu6", "none"]))}
        if fc:
            L["from_fc"] = True
        layers.append(L)
        cur.update(shape=[oh, ow, oc], scale=out_scale, zp=L["out_zp"])

    def add_dw(name, k, stride, same):
        h, w, c = cur["shape"]
        oh, ph = _out_and_pad(h, k, stride, same)
        ow, _ = _out_and_pad(w, k, stride, same)
        arrays[name + ".w"] = rng.integers(-127, 128, (c, k, k)).astype(np.int8)
        arrays[name + ".b"] = rng.integers(-3000, 3000, c).astype(np.int32)
        arrays[name + ".w_scale"] = (0.02 * rng.uniform(0.5, 1.5, c)).astype(np.float32)
        out_scale = new_q(k * k)
        L = {"name": name, "op": "dwconv", "in_shape": [h, w, c], "out_shape": [oh, ow, c],
             "in_scale": cur["scale"], "in_zp": cur["zp"], "out_scale": out_scale,
             "out_zp": int(rng.integers(-128, 0)), "kh": k, "kw": k, "stride": stride,
             "pad": ph, "act": str(rng.choice(["relu", "relu6", "none"]))}
        layers.append(L)
        cur.update(shape=[oh, ow, c], scale=out_scale, zp=L["out_zp"])

    def add_pool(name, op, k, stride, same):
        h, w, c = cur["shape"]
        oh, ph = _out_and_pad(h, k, stride, same)
        ow, _ = _out_and_pad(w, k, stride, same)
        layers.append({"name": name, "op": op, "in_shape": [h, w, c], "out_shape": [oh, ow, c],
                       "in_scale": cur["scale"], "in_zp": cur["zp"], "out_scale": cur["scale"],
                       "out_zp": cur["zp"], "k": k, "stride": stride, "pad": ph, "act": "none"})
        cur.update(shape=[oh, ow, c])

    # packed first conv
    add_conv("c0", int(rng.choice([1, 3])), int(rng.choice([1, 2])), bool(rng.integers(2)),
             int(rng.integers(4, 17)))
    n_mid = int(rng.integers(3, 7))
    for i in range(n_mid):
        h = cur["shape"][0]
        kind = rng.choice(["dw", "dw", "conv", "pool"])
        if kind == "dw":
            k = int(rng.choice([1, 2, 3])) if h >= 3 else 1
            add_dw(f"d{i}", k, int(rng.choice([1, 2])) if h >= 4 else 1, bool(rng.integers(2)))
        elif kind == "conv":
            k = int(rng.choice([1, 3])) if h >= 3 else 1
            add_conv(f"c{i + 1}", k, 1, True, int(rng.integers(4, 41)))
        elif h >= 4:
            if rng.integers(2):     # max pool, possibly padded
                add_pool(f"p{i}", "maxpool", int(rng.choice([2, 3])), 2, bool(rng.integers(2)))
            else:                   # average pool: lowered to DWCONV (3x3, valid)
                add_pool(f"p{i}", "avgpool", 3, int(rng.choice([1, 2])), False)
    # reduce to 1x1, then a classifier (a global average pool must be 3x3 to be lowered)
    while cur["shape"][0] > 3 or cur["shape"][0] == 2:
        add_dw(f"r{len(layers)}", 3, 2, True)
    if cur["shape"][0] == 3:
        add_pool("gap", "avgpool", 3, 1, False)
    add_conv("fc", 1, 1, False, int(rng.integers(2, 11)), fc=True)

    ir = {"input": {"tensor": 0, "shape": [H, W, C], "scale": 1 / 255, "zp": -128},
          "output": {"tensor": 1, "softmax_dropped": True}, "layers": layers}
    return ir, arrays


def random_image(ir, seed: int):
    return np.random.default_rng(seed).integers(0, 256, ir["input"]["shape"]).astype(np.uint8)


if __name__ == "__main__":   # print a few nets
    for s in range(3):
        ir, _ = random_ir(s)
        print(f"seed {s}: input {ir['input']['shape']}")
        for L in ir["layers"]:
            print(f"   {L['op']:8s} {L['in_shape']} -> {L['out_shape']} "
                  f"k={L.get('kh', L.get('k'))} s={L['stride']} pad={L['pad']}")
