"""ONNX front end: a quantized ONNX classifier -> the compiler's IR.

Produces the same (ir, arrays) pair as sw/model/export_ir.py, so everything downstream
(compiler, simulator, RTL tests, board images) is shared with the TFLite path.

Accepted graphs are sequential int8/uint8 classifiers in either quantization format:
  * QDQ (onnxruntime.quantization default): DequantizeLinear -> Conv/Gemm/MatMul/pool ->
    [Relu/Clip] -> QuantizeLinear, weights as DequantizeLinear(int8 initializer)
  * QOperator: QLinearConv, QLinearMatMul, QGemm, QLinearAveragePool,
    QLinearGlobalAveragePool, MaxPool on quantized tensors
uint8 activations are mapped to int8 (code - 128, zero point - 128), which is exact.
Weights must be symmetric (int8 zero point 0, or uint8 zero point 128).

Rounding: ONNX requantizes in float with round-half-to-even. The NPU's requant is TFLite's
fixed-point multiply; with rounding="onnx" each layer uses single rounding (one rounding of
acc * scale, round half up) where the scale allows it, which differs from ONNX only on
exact ties and in the 31-bit multiplier. rounding="tflite" uses TFLite's per-op modes.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from npu.quant import effective_scale, quantize_multiplier


class UnsupportedModel(Exception):
    """The model uses something the NPU (or this front end) cannot run; the message says what."""


ACT_OPS = ("Relu", "Clip")
LAYOUT_OPS = ("Flatten", "Reshape", "Squeeze", "Unsqueeze")


def _attrs(node):
    import onnx
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


class _Graph:
    def __init__(self, model):
        import onnx
        from onnx import numpy_helper
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception:               # keep going with whatever value_info exists
            pass
        g = model.graph
        self.init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        self.nodes = []
        for n in g.node:
            if n.op_type == "Constant":
                self.init[n.output[0]] = numpy_helper.to_array(_attrs(n)["value"])
            else:
                self.nodes.append(n)
        self.producer = {o: n for n in self.nodes for o in n.output}
        self.consumers = defaultdict(list)
        for n in self.nodes:
            for i in n.input:
                if i:
                    self.consumers[i].append(n)
        self.shapes = {}
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            t = vi.type.tensor_type
            if t.HasField("shape"):
                self.shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else None
                                        for d in t.shape.dim]
        self.elem = {vi.name: vi.type.tensor_type.elem_type
                     for vi in list(g.value_info) + list(g.input) + list(g.output)}
        self.inputs = [i for i in g.input if i.name not in self.init]
        self.outputs = [o.name for o in g.output]
        self.opset = {o.domain: o.version for o in model.opset_import}

    def const(self, name, what):
        if name not in self.init:
            raise UnsupportedModel(f"{what} ({name!r}) must be a constant initializer")
        return self.init[name]

    def sole_consumer(self, t):
        c = self.consumers.get(t, [])
        if t in self.outputs and c:
            raise UnsupportedModel(f"tensor {t!r} is both a graph output and used further")
        if len(c) > 1:
            raise UnsupportedModel(f"tensor {t!r} feeds {len(c)} nodes ({', '.join(n.op_type for n in c)}): "
                                   "only sequential networks are supported (no residual "
                                   "connections, concatenation or multiple heads)")
        return c[0] if c else None


def _qparams(g, scale_name, zp_name, what):
    """(scale, int8 zero point, input dtype) for a per-tensor activation quantization."""
    s = np.asarray(g.const(scale_name, f"{what} scale"), np.float64).reshape(-1)
    if s.size != 1:
        raise UnsupportedModel(f"{what}: activations must be quantized per tensor")
    if zp_name:
        z = g.const(zp_name, f"{what} zero point")
        dtype = z.dtype
        z = int(np.asarray(z).reshape(-1)[0])
    else:
        dtype, z = np.dtype(np.uint8), 0
    if dtype == np.uint8:
        z -= 128
    elif dtype != np.int8:
        raise UnsupportedModel(f"{what}: {dtype} activations (int8/uint8 only)")
    return float(s[0]), z, dtype


def _weights(g, w_name, s_name, zp_name, what):
    """Symmetric int8 weights + per-output-channel (axis 0) scales."""
    w = g.const(w_name, f"{what} weights")
    s = np.asarray(g.const(s_name, f"{what} weight scale"), np.float32).reshape(-1)
    zp = np.asarray(g.const(zp_name, f"{what} weight zero point")) if zp_name else np.zeros(1, w.dtype)
    if w.dtype == np.uint8 and np.all(zp == 128):
        w = w.astype(np.int16) - 128
    elif w.dtype != np.int8 or np.any(zp != 0):
        raise UnsupportedModel(f"{what}: weights must be symmetric int8 (zero point 0); got "
                               f"{w.dtype} with zero point(s) {np.unique(zp)[:4]}")
    return w.astype(np.int8), s


def _dq_weights(g, name, what, axis_default=1):
    """Weights given as DequantizeLinear(initializer) in a QDQ graph."""
    dq = g.producer.get(name)
    if name in g.init or dq is None or dq.op_type != "DequantizeLinear":
        raise UnsupportedModel(f"{what}: weights are not quantized; quantize the model first "
                               "(e.g. onnxruntime.quantization.quantize_static)")
    zp = dq.input[2] if len(dq.input) > 2 and dq.input[2] else None
    w, s = _weights(g, dq.input[0], dq.input[1], zp, what)
    return w, s, _attrs(dq).get("axis", axis_default)


def _bias(g, name, in_scale, w_scale, what):
    """int32 bias at scale in_scale * w_scale, from DQ(int32), an int32 or a float initializer."""
    if not name:
        return None
    dq = g.producer.get(name)
    if dq is not None and dq.op_type == "DequantizeLinear":
        b = g.const(dq.input[0], f"{what} bias")
        bs = np.asarray(g.const(dq.input[1], f"{what} bias scale"), np.float64).reshape(-1)
        expect = in_scale * w_scale.astype(np.float64)
        if b.dtype != np.int32 or not np.allclose(bs, expect, rtol=1e-5, atol=0):
            raise UnsupportedModel(f"{what}: bias must be int32 at scale input_scale * weight_scale")
        return b.astype(np.int64).reshape(-1)
    b = g.const(name, f"{what} bias")
    if b.dtype == np.int32:
        return b.astype(np.int64).reshape(-1)
    q = np.round(b.astype(np.float64) / (in_scale * w_scale.astype(np.float64)))
    if np.any(np.abs(q) >= 2 ** 31):
        raise UnsupportedModel(f"{what}: bias does not fit int32 at the accumulator scale")
    return q.astype(np.int64)


def _square(vals, what, name):
    vals = list(vals)
    if len(set(vals)) != 1:
        raise UnsupportedModel(f"{name}: non-square {what} {vals}")
    return int(vals[0])


def _geometry(a, name, H, W, k_default=None):
    """kernel, stride, pad (top == left) from Conv/Pool attributes."""
    k = _square(a.get("kernel_shape", k_default), "kernel", name)
    stride = _square(a.get("strides", [1, 1]), "stride", name)
    if any(d != 1 for d in a.get("dilations", [1, 1])):
        raise UnsupportedModel(f"{name}: dilated kernels are not supported")
    auto = a.get("auto_pad", b"NOTSET")
    auto = auto.decode() if isinstance(auto, bytes) else auto
    if auto in ("SAME_UPPER", "SAME_LOWER"):
        oh = -(-H // stride)
        total = max((oh - 1) * stride + k - H, 0)
        pad = total // 2 if auto == "SAME_UPPER" else total - total // 2
        ow = -(-W // stride)
        tw = max((ow - 1) * stride + k - W, 0)
        if (tw // 2 if auto == "SAME_UPPER" else tw - tw // 2) != pad:
            raise UnsupportedModel(f"{name}: unequal top/left padding")
    elif auto == "VALID":
        pad, oh, ow = 0, (H - k) // stride + 1, (W - k) // stride + 1
    else:
        p = list(a.get("pads", [0, 0, 0, 0]))
        if p[0] != p[1]:
            raise UnsupportedModel(f"{name}: top padding {p[0]} != left padding {p[1]}")
        pad = int(p[0])
        rnd = (lambda x: -(-x // stride)) if a.get("ceil_mode", 0) else (lambda x: x // stride)
        oh = rnd(H + p[0] + p[2] - k) + 1
        ow = rnd(W + p[1] + p[3] - k) + 1
    return k, stride, pad, oh, ow


class OnnxImporter:
    def __init__(self, model, rounding="onnx"):
        if rounding not in ("onnx", "tflite"):
            raise ValueError(rounding)
        self.g = _Graph(model)
        self.rounding = rounding
        self.layers, self.arrays = [], {}
        self.notes = []                  # things the user should know (shown in the report)

    # ------------------------------------------------------------ state helpers
    def _layer(self, op, name, **kw):
        h, w, c = self.shape
        L = {"name": f"L{len(self.layers)}_{op}", "onnx_node": name, "op": op,
             "in_shape": [h, w, c], "in_scale": self.q[0], "in_zp": self.q[1]}
        L.update(kw)
        self.layers.append(L)
        return L

    def _finish(self, L, out_q, out_shape, out_tensor, out_dtype, acts=()):
        L.update(out_shape=list(out_shape), out_scale=out_q[0], out_zp=out_q[1],
                 out_tensor=out_tensor, out_dtype=str(np.dtype(out_dtype)))
        lo, hi = -128, 127
        for kind, a, b in acts:          # clamps in real units -> output codes
            s, z = out_q[0], out_q[1]
            if a is not None:
                lo = max(lo, z + int(np.round(a / s)))
            if b is not None:
                hi = min(hi, z + int(np.round(b / s)))
        if lo > hi:
            raise UnsupportedModel(f"{L['onnx_node']}: activation range is empty after quantization")
        L["act_min"], L["act_max"] = lo, hi
        self.shape, self.q, self.flat = list(out_shape), tuple(out_q[:2]), False
        self.dtype = np.dtype(out_dtype)

    def _set_rmode(self, L, scales):
        if self.rounding == "tflite":
            return
        shifts = [quantize_multiplier(effective_scale(L["in_scale"], s, L["out_scale"]))[1] for s in scales]
        L["rmode"] = "single" if max(shifts) <= 0 else "double"
        if L["rmode"] == "double":
            self.notes.append(f"{L['name']}: requant scale >= 1, using double rounding (may differ "
                              "from ONNX by 1 LSB more often)")

    # ------------------------------------------------------------ ops
    def _spatial(self, name):
        if self.flat:
            raise UnsupportedModel(f"{name}: needs a feature map, but its input was flattened")

    def conv(self, name, a, w, w_scale, bias):
        self._spatial(name)
        H, W, C = self.shape
        group = int(a.get("group", 1))
        k, stride, pad, oh, ow = _geometry(a, name, H, W, list(w.shape[2:]))
        O = w.shape[0]
        if w.shape[2] != w.shape[3]:
            raise UnsupportedModel(f"{name}: non-square kernel {list(w.shape[2:])}")
        if group == 1:
            if w.shape[1] != C:
                raise UnsupportedModel(f"{name}: weight shape {list(w.shape)} vs {C} input channels")
            L = self._layer("conv", name, kh=k, kw=k, stride=stride, pad=pad)
            self.arrays[L["name"] + ".w"] = w.transpose(0, 2, 3, 1).copy()      # [O, k, k, I]
        elif group == C and O == C and w.shape[1] == 1:
            L = self._layer("dwconv", name, kh=k, kw=k, stride=stride, pad=pad)
            self.arrays[L["name"] + ".w"] = w[:, 0].copy()                      # [C, k, k]
        else:
            raise UnsupportedModel(f"{name}: grouped convolution (group={group}) other than "
                                   "depthwise with multiplier 1")
        w_scale = np.broadcast_to(w_scale, (O,)).astype(np.float32)
        self.arrays[L["name"] + ".b"] = (np.zeros(O, np.int64) if bias is None else bias).astype(np.int32)
        self.arrays[L["name"] + ".w_scale"] = w_scale
        return L, (oh, ow, O)

    def fc(self, name, w_oi, w_scale, bias):
        """w_oi: [O, I] with I in the ONNX flattening order (C, H, W)."""
        H, W, C = self.shape
        O, I = w_oi.shape
        if I != H * W * C:
            raise UnsupportedModel(f"{name}: {I} inputs, but the tensor has {H}x{W}x{C}")
        if H != W:
            raise UnsupportedModel(f"{name}: fully connected over a non-square {H}x{W} map")
        L = self._layer("conv", name, kh=H, kw=W, stride=1, pad=0, from_fc=True)
        self.arrays[L["name"] + ".w"] = w_oi.reshape(O, C, H, W).transpose(0, 2, 3, 1).copy()
        w_scale = np.broadcast_to(w_scale, (O,)).astype(np.float32)
        self.arrays[L["name"] + ".b"] = (np.zeros(O, np.int64) if bias is None else bias).astype(np.int32)
        self.arrays[L["name"] + ".w_scale"] = w_scale
        return L, (1, 1, O)

    def pool(self, name, op_type, a):
        self._spatial(name)
        H, W, C = self.shape
        if op_type == "GlobalAveragePool":
            if H != W:
                raise UnsupportedModel(f"{name}: global pooling over a non-square {H}x{W} map")
            k, stride, pad, oh, ow = H, 1, 0, 1, 1
        else:
            k, stride, pad, oh, ow = _geometry(a, name, H, W)
        op = "maxpool" if op_type == "MaxPool" else "avgpool"
        if op == "avgpool" and pad and a.get("count_include_pad", 0) == 0:
            raise UnsupportedModel(f"{name}: padded average pooling")
        L = self._layer(op, name, k=k, stride=stride, pad=pad)
        if op == "avgpool":
            L["approx"] = True
            if k > 3:
                self.notes.append(f"{L['name']}: a {k}x{k} average pool runs as a CONV with identity "
                                  "weights (slow); windows up to 3x3 use the depthwise unit")
        return L, (oh, ow, C)

    # ------------------------------------------------------------ graph walk
    def run(self):
        g = self.g
        if len(g.inputs) != 1:
            raise UnsupportedModel(f"the model has {len(g.inputs)} inputs (one image input only)")
        inp = g.inputs[0]
        shp = g.shapes.get(inp.name)
        if not shp or len(shp) != 4:
            raise UnsupportedModel(f"input {inp.name!r} must be a 4-D image, got shape {shp}")
        self.input_name, self.input_float = inp.name, g.elem[inp.name] == 1
        t = inp.name
        # NHWC input followed by a transpose to NCHW (e.g. converted from TFLite)
        c = g.sole_consumer(t)
        self.input_layout = "NCHW"
        if c is not None and c.op_type == "Transpose" and list(_attrs(c).get("perm", [])) == [0, 3, 1, 2]:
            self.input_layout, t = "NHWC", c.output[0]
            _, H, W, C = [d or 1 for d in shp]
        else:
            _, C, H, W = [d or 1 for d in shp]
        self.shape, self.flat = [H, W, C], False

        # the first quantized tensor
        if self.input_float:
            q = g.sole_consumer(t)
            if q is None or q.op_type != "QuantizeLinear":
                raise UnsupportedModel("the float input must go straight into QuantizeLinear")
            self.q = _qparams(g, q.input[1], q.input[2] if len(q.input) > 2 else None, "input")
            t = q.output[0]
        else:
            # integer input feeding a DequantizeLinear or a QLinear op: find its parameters
            c = g.sole_consumer(t)
            if c is None:
                raise UnsupportedModel("the input is not used")
            if c.op_type == "DequantizeLinear":
                self.q = _qparams(g, c.input[1], c.input[2] if len(c.input) > 2 else None, "input")
            elif c.op_type.startswith("QLinear"):
                self.q = _qparams(g, c.input[1], c.input[2], "input")
            else:
                raise UnsupportedModel(f"integer input feeds {c.op_type}")
        s_in, z_in, in_dtype = self.q
        if z_in != -128:
            raise UnsupportedModel(
                f"input quantization zero point is {z_in + 128} (uint8): the NPU feeds raw pixels, "
                "so input codes must equal pixel values (uint8 zero point 0 / int8 -128). Scale "
                "inputs to [0, 1] (x = pixel / 255) instead of mean/std normalization")
        if abs(s_in * 255 - 1) > 1e-6:
            self.notes.append(f"input scale is 1/{1 / s_in:.2f}: the NPU feeds pixel p as code p, so "
                              f"the model sees x = p * {s_in:.6g} rather than p / 255")
        self.input_scale = s_in
        self.dtype = np.dtype(self.q[2])
        self.q = tuple(self.q[:2])
        self.softmax_dropped = False

        while True:
            if t in g.outputs:
                break
            n = g.sole_consumer(t)
            if n is None:
                raise UnsupportedModel(f"tensor {t!r} is neither consumed nor a graph output")
            if n.op_type == "DequantizeLinear":
                t = self._qdq_step(n)
            elif n.op_type in LAYOUT_OPS or n.op_type == "MaxPool":
                t = self._int_step(n)
            elif n.op_type == "QLinearSoftmax":
                t = self._final_softmax(n)
            elif n.op_type in ("QLinearConv", "QLinearMatMul", "QGemm", "QLinearAveragePool",
                               "QLinearGlobalAveragePool"):
                t = self._qop_step(n)
            else:
                raise UnsupportedModel(f"{n.name or n.op_type}: {n.op_type} is not supported by the NPU")
            if t is None:               # reached the (float) graph output
                break

        if not self.layers:
            raise UnsupportedModel("the model has no compute layers")
        if self.shape[:2] != [1, 1]:
            raise UnsupportedModel(f"the network must end in a 1x1xC classifier output "
                                   f"(it ends at {self.shape[0]}x{self.shape[1]}x{self.shape[2]})")
        ir = {"source": "onnx", "rounding": self.rounding,
              "input": {"tensor": self.input_name, "shape": [H, W, C], "scale": self.input_scale,
                        "zp": -128, "float": self.input_float, "layout": self.input_layout,
                        "dtype": str(np.dtype(in_dtype))},
              "output": {"softmax_dropped": self.softmax_dropped},
              "layers": self.layers, "notes": self.notes}
        return ir, self.arrays

    def _tail(self, t, allow_end=True):
        """Walk float ops after a compute node: activations, a bias Add, then QuantizeLinear.
        Returns (QuantizeLinear node or None at the graph output, acts, bias name)."""
        g = self.g
        acts, bias = [], None
        while True:
            n = g.sole_consumer(t)
            if n is None or n.op_type == "Softmax":
                raise UnsupportedModel(f"the output of {g.producer[t].op_type} is not quantized "
                                       "(every layer, including the last, needs a QuantizeLinear)")
            if n.op_type == "QuantizeLinear":
                return n, acts, bias
            if n.op_type in ACT_OPS:
                acts.append(self._act(n))
            elif n.op_type == "Add" and bias is None and not acts:
                other = [i for i in n.input if i != t]
                if len(other) != 1 or (other[0] not in g.init and other[0] not in g.producer):
                    raise UnsupportedModel(f"{n.name}: Add of two tensors (residual) is not supported")
                bias = other[0]
            else:
                raise UnsupportedModel(f"{n.name or n.op_type}: {n.op_type} after a compute op is not supported")
            t = n.output[0]

    def _final_softmax(self, n):
        """A Softmax (float or quantized) is dropped if only Q/DQ follow it to the output:
        argmax is unchanged, the NPU returns the logits."""
        g, t = self.g, n.output[0]
        while t not in g.outputs:
            c = g.sole_consumer(t)
            if c is None or c.op_type not in ("QuantizeLinear", "DequantizeLinear"):
                raise UnsupportedModel("Softmax is only supported as the final op")
            t = c.output[0]
        self.softmax_dropped = True
        return None

    def _act(self, n):
        """Relu / Clip -> (kind, low, high) in real units (None = unbounded)."""
        if n.op_type == "Relu":
            return ("relu", 0.0, None)
        g, a = self.g, _attrs(n)
        lo = a.get("min") if len(n.input) < 2 or not n.input[1] else float(g.const(n.input[1], "Clip min"))
        hi = a.get("max") if len(n.input) < 3 or not n.input[2] else float(g.const(n.input[2], "Clip max"))
        return ("clip", lo, hi)

    def _qdq_step(self, dq):
        g = self.g
        f = dq.output[0]
        c = g.sole_consumer(f)
        if c is None:
            if f in g.outputs:
                return None
            raise UnsupportedModel(f"{f!r} is unused")
        name = c.name or c.op_type
        a = _attrs(c)
        if c.op_type == "Softmax":
            return self._final_softmax(c)
        if c.op_type in LAYOUT_OPS:
            return self._layout(c, float_domain=True)
        if c.op_type in ACT_OPS:
            # a separately quantized activation: fold it into the previous layer's clamp
            q, acts, _ = self._tail(c.output[0])
            oq = _qparams(g, q.input[1], q.input[2] if len(q.input) > 2 else None, f"{name} output")
            prev = self.layers[-1] if self.layers else None
            if prev is None or prev["out_tensor"] != dq.input[0] or oq[:2] != tuple(self.q):
                raise UnsupportedModel(f"{name}: a standalone {c.op_type} that changes the "
                                       "quantization is not supported")
            self.shape = prev["in_shape"]
            self.q = (prev["in_scale"], prev["in_zp"])
            lo, hi = prev["act_min"], prev["act_max"]
            self._finish(prev, oq, prev["out_shape"], q.output[0], oq[2], [self._act(c)] + acts)
            prev["act_min"], prev["act_max"] = max(lo, prev["act_min"]), min(hi, prev["act_max"])
            return q.output[0]
        if c.op_type in ("Conv", "Gemm", "MatMul"):
            w, w_scale, axis = _dq_weights(g, c.input[1], name, 1 if c.op_type == "MatMul" else 0)
            if c.op_type == "Conv":
                if axis != 0 and w_scale.size > 1:
                    raise UnsupportedModel(f"{name}: per-channel weight scales must be on axis 0")
                q, acts, bname = self._tail(c.output[0])
                bias_in = c.input[2] if len(c.input) > 2 and c.input[2] else bname
                bias = _bias(g, bias_in, self.q[0], np.broadcast_to(w_scale, (w.shape[0],)), name)
                L, shape = self.conv(name, a, w, w_scale, bias)
            else:
                w_oi, w_scale = self._fc_weights(c, a, w, w_scale, axis)
                q, acts, bname = self._tail(c.output[0])
                bias_in = c.input[2] if c.op_type == "Gemm" and len(c.input) > 2 and c.input[2] else bname
                bias = _bias(g, bias_in, self.q[0], np.broadcast_to(w_scale, (w_oi.shape[0],)), name)
                L, shape = self.fc(name, w_oi, w_scale, bias)
        elif c.op_type in ("MaxPool", "AveragePool", "GlobalAveragePool"):
            q, acts, _ = self._tail(c.output[0])
            L, shape = self.pool(name, c.op_type, a)
        else:
            raise UnsupportedModel(f"{name}: {c.op_type} is not supported by the NPU")
        out_q = _qparams(g, q.input[1], q.input[2] if len(q.input) > 2 else None, f"{name} output")
        self._finish(L, out_q, shape, q.output[0], out_q[2], acts)
        if L["op"] in ("conv", "dwconv"):
            self._set_rmode(L, self.arrays[L["name"] + ".w_scale"])
        if L["op"] == "maxpool" and (L["in_scale"], L["in_zp"]) != (L["out_scale"], L["out_zp"]):
            raise UnsupportedModel(f"{name}: MaxPool must keep its input quantization")
        return q.output[0]

    def _fc_weights(self, c, a, w, w_scale, axis):
        name = c.name or c.op_type
        if c.op_type == "Gemm":
            if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                raise UnsupportedModel(f"{name}: Gemm needs transA=0, alpha=beta=1")
            w_oi = w if a.get("transB", 0) else w.T
            if w_scale.size > 1 and axis != (0 if a.get("transB", 0) else 1):
                raise UnsupportedModel(f"{name}: per-channel scales must be per output")
        else:                            # MatMul: [I, O]
            w_oi = w.T
            if w_scale.size > 1 and axis != 1:
                raise UnsupportedModel(f"{name}: per-channel scales must be per output")
        return np.ascontiguousarray(w_oi), w_scale

    def _layout(self, n, float_domain):
        """Flatten/Reshape/Squeeze: no data movement (the NPU's HWC order is handled by
        permuting the next FC's weights). In the float domain a re-quantization must follow
        with the same parameters."""
        g = self.g
        H, W, C = self.shape
        out = g.shapes.get(n.output[0])
        if H * W > 1:
            # only a flatten to [1, C*H*W] (ONNX order C, H, W) for a following FC
            if out is None or len(out) != 2 or out[1] != H * W * C:
                raise UnsupportedModel(f"{n.name or n.op_type}: reshape of a {H}x{W}x{C} map to "
                                       f"{out} (only a flatten before a fully connected layer)")
            self.flat = True
        t = n.output[0]
        if float_domain:
            q = g.sole_consumer(t)
            if q is None or q.op_type != "QuantizeLinear":
                if q is not None and q.op_type in ("Gemm", "MatMul"):
                    raise UnsupportedModel(f"{q.name}: input is not quantized")
                raise UnsupportedModel(f"{n.name or n.op_type}: expected QuantizeLinear after it")
            oq = _qparams(g, q.input[1], q.input[2] if len(q.input) > 2 else None, "reshape output")
            if abs(oq[0] - self.q[0]) > 1e-9 * self.q[0] or oq[1] != self.q[1]:
                raise UnsupportedModel(f"{n.name or n.op_type}: re-quantization across a reshape")
            t = q.output[0]
        return t

    def _int_step(self, n):
        if n.op_type in LAYOUT_OPS:
            return self._layout(n, float_domain=False)
        L, shape = self.pool(n.name or n.op_type, "MaxPool", _attrs(n))
        self._finish(L, self.q, shape, n.output[0], self.dtype)
        return n.output[0]

    def _qop_step(self, n):
        g = self.g
        name = n.name or n.op_type
        a = _attrs(n)
        I = list(n.input) + [""] * 10
        if n.op_type == "QLinearConv":
            w, w_scale = _weights(g, I[3], I[4], I[5] or None, name)
            bias = _bias(g, I[8], self.q[0], np.broadcast_to(w_scale, (w.shape[0],)), name) if I[8] else None
            L, shape = self.conv(name, a, w, w_scale, bias)
            oq = _qparams(g, I[6], I[7] or None, f"{name} output")
        elif n.op_type == "QLinearMatMul":
            w, w_scale = _weights(g, I[3], I[4], I[5] or None, name)
            oq = _qparams(g, I[6], I[7] or None, f"{name} output")
            L, shape = self.fc(name, np.ascontiguousarray(w.T), w_scale, None)
        elif n.op_type == "QGemm":
            w, w_scale = _weights(g, I[3], I[4], I[5] or None, name)
            if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0:
                raise UnsupportedModel(f"{name}: QGemm needs transA=0, alpha=1")
            w_oi = np.ascontiguousarray(w if a.get("transB", 0) else w.T)
            bias = _bias(g, I[6], self.q[0], np.broadcast_to(w_scale, (w_oi.shape[0],)), name) if I[6] else None
            if not I[7]:
                raise UnsupportedModel(f"{name}: QGemm with a float output")
            oq = _qparams(g, I[7], I[8] or None, f"{name} output")
            L, shape = self.fc(name, w_oi, w_scale, bias)
        else:                            # QLinearAveragePool / QLinearGlobalAveragePool
            if a.get("channels_last", 0):
                raise UnsupportedModel(f"{name}: channels_last pooling")
            L, shape = self.pool(name, n.op_type[len("QLinear"):], a)
            oq = _qparams(g, I[3], I[4] or None, f"{name} output")
        self._finish(L, oq, shape, n.output[0], oq[2])
        if L["op"] in ("conv", "dwconv"):
            self._set_rmode(L, self.arrays[L["name"] + ".w_scale"])
        return n.output[0]


def import_onnx(path_or_model, rounding="onnx"):
    """-> (ir, arrays). Raises UnsupportedModel with a readable reason."""
    import onnx
    model = onnx.load(str(path_or_model)) if isinstance(path_or_model, (str, Path)) else path_or_model
    ir, arrays = OnnxImporter(model, rounding).run()
    if isinstance(path_or_model, (str, Path)):
        ir["source"] = Path(path_or_model).name
    return ir, arrays
