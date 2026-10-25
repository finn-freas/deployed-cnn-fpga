"""Small float ONNX classifiers, quantized with onnxruntime, for testing npuc.

A spec is a list of layers:
    ("conv", out_ch, k, stride, pad, act)    act: "relu" | "relu6" | "none"
    ("dw", k, stride, pad, act)
    ("maxpool", k, stride, pad) / ("avgpool", k, stride) / ("gap",)
    ("flatten",) / ("fc", out) / ("softmax",)
"""

import numpy as np


def float_model(spec, input_shape, seed=0, opset=17):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(seed)
    H, W, C = input_shape
    nodes, inits = [], []
    cur, n = "image", 0

    def t(prefix):
        nonlocal n
        n += 1
        return f"{prefix}{n}"

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, np.float32), name))
        return name

    def act(x, kind):
        if kind == "none":
            return x
        y = t("act")
        if kind == "relu":
            nodes.append(helper.make_node("Relu", [x], [y]))
        else:
            nodes.append(helper.make_node("Clip", [x, init(t("lo"), 0.0), init(t("hi"), 6.0)], [y]))
        return y

    for L in spec:
        kind = L[0]
        y = t(kind)
        if kind == "conv":
            _, o, k, s, p, a = L
            w = rng.normal(0, np.sqrt(2 / (k * k * C)), (o, C, k, k))
            nodes.append(helper.make_node("Conv", [cur, init(t("w"), w), init(t("b"), rng.normal(0, 0.1, o))],
                                          [y], kernel_shape=[k, k], strides=[s, s], pads=[p] * 4))
            cur, C = act(y, a), o
        elif kind == "dw":
            _, k, s, p, a = L
            w = rng.normal(0, np.sqrt(2 / (k * k)), (C, 1, k, k))
            nodes.append(helper.make_node("Conv", [cur, init(t("w"), w), init(t("b"), rng.normal(0, 0.1, C))],
                                          [y], kernel_shape=[k, k], strides=[s, s], pads=[p] * 4, group=C))
            cur = act(y, a)
        elif kind == "maxpool":
            _, k, s, p = L
            nodes.append(helper.make_node("MaxPool", [cur], [y], kernel_shape=[k, k], strides=[s, s], pads=[p] * 4))
            cur = y
        elif kind == "avgpool":
            _, k, s = L
            nodes.append(helper.make_node("AveragePool", [cur], [y], kernel_shape=[k, k], strides=[s, s]))
            cur = y
        elif kind == "gap":
            nodes.append(helper.make_node("GlobalAveragePool", [cur], [y]))
            cur = y
        elif kind == "flatten":
            nodes.append(helper.make_node("Flatten", [cur], [y], axis=1))
            cur = y
        elif kind == "fc":
            _, o = L
            # input width from shape inference of what we have so far
            g = helper.make_graph(nodes, "tmp", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, input_shape[2], H, W])],
                                  [helper.make_tensor_value_info(cur, TensorProto.FLOAT, None)], inits)
            m = onnx.shape_inference.infer_shapes(helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)]))
            dims = next(v for v in list(m.graph.value_info) + list(m.graph.output) if v.name == cur)
            fan_in = int(np.prod([d.dim_value for d in dims.type.tensor_type.shape.dim][1:]))
            w = rng.normal(0, np.sqrt(1 / fan_in), (o, fan_in))
            nodes.append(helper.make_node("Gemm", [cur, init(t("w"), w), init(t("b"), rng.normal(0, 0.1, o))],
                                          [y], transB=1))
            cur = y
        elif kind == "softmax":
            nodes.append(helper.make_node("Softmax", [cur], [y], axis=1))
            cur = y
        else:
            raise ValueError(kind)
    g = helper.make_graph(nodes, "testnet",
                          [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, input_shape[2], H, W])],
                          [helper.make_tensor_value_info(cur, TensorProto.FLOAT, None)], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    return m


def calibration_images(input_shape, n=32, seed=1):
    """Random pixels: full 0..255 range, so the input quantizes to scale 1/255, zero point 0."""
    return np.random.default_rng(seed).integers(0, 256, (n, *input_shape)).astype(np.uint8)


def quantize(float_path, out_path, images_u8, fmt="QDQ", act="uint8", per_channel=True):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(images_u8)

        def get_next(self):
            img = next(self.it, None)
            if img is None:
                return None
            return {"image": (img.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)}

    quantize_static(str(float_path), str(out_path), Reader(),
                    quant_format=QuantFormat.QDQ if fmt == "QDQ" else QuantFormat.QOperator,
                    activation_type=QuantType.QUInt8 if act == "uint8" else QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=per_channel)


SPECS = {
    # MobileNet-style: packed first conv, depthwise/pointwise pairs, pools, global pool, FC
    "mobile": ([("conv", 8, 3, 2, 1, "relu6"), ("dw", 3, 1, 1, "relu6"), ("conv", 16, 1, 1, 0, "relu6"),
                ("dw", 3, 2, 1, "relu6"), ("conv", 24, 1, 1, 0, "relu6"), ("avgpool", 3, 1),
                ("gap",), ("flatten",), ("fc", 5), ("softmax",)], (24, 24, 3)),
    # LeNet-style with max pooling and a flattened (spatial) FC
    "lenet": ([("conv", 6, 3, 1, 1, "relu"), ("maxpool", 2, 2, 0), ("conv", 12, 3, 1, 0, "relu"),
               ("maxpool", 2, 2, 0), ("flatten",), ("fc", 10)], (14, 14, 1)),
    # odd sizes, stride-2 depthwise without padding, 2x2 depthwise, plain ReLU
    "odd": ([("conv", 5, 3, 1, 0, "none"), ("dw", 2, 1, 0, "relu"), ("dw", 3, 2, 0, "relu"),
             ("conv", 11, 1, 1, 0, "relu"), ("gap",), ("flatten",), ("fc", 3)], (13, 13, 2)),
}
