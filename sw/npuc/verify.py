"""Run the compiled program in the NPU simulator and compare it with ONNX Runtime.

The simulator is cycle-exact with the RTL and bit-exact with it on every layer (see
tb/), so agreement here is agreement with the hardware.
"""

from dataclasses import dataclass, field

import numpy as np

from npu.sim import NpuSim, total_cycles


@dataclass
class LayerDiff:
    name: str
    values: int = 0
    exact: int = 0
    max_diff: int = 0


@dataclass
class Verification:
    images: int = 0
    argmax_agree: int = 0
    layers: list[LayerDiff] = field(default_factory=list)

    @property
    def bit_exact(self):
        return all(d.exact == d.values for d in self.layers)


def run_sim(prog, image_u8):
    """-> (class, score, per-layer outputs [h, w, c] int8, per-instruction cycles, total cycles)."""
    sim = NpuSim.from_program(prog)
    layer_at = {L["instr"]: L for L in prog.meta["layers"]}
    outs = {}

    def snap(pc, ins):
        if pc in layer_at:
            L = layer_at[pc]
            outs[L["name"]] = sim.read_tensor(L["base"], L["h"], L["w"], L["cb"], L["c"]).copy()

    cls, score = sim.run(image_u8, trace=snap)
    return cls, score, outs, sim.stats, total_cycles(sim.stats, len(prog.instrs))


class OrtReference:
    """The original ONNX model in onnxruntime, with every layer's quantized output exposed."""

    def __init__(self, model_path, ir):
        import onnx
        import onnxruntime as ort
        m = onnx.load(str(model_path))
        self.names = [L["out_tensor"] for L in ir["layers"]]
        have = {o.name for o in m.graph.output}
        for L in ir["layers"]:
            if L["out_tensor"] not in have:
                et = onnx.TensorProto.UINT8 if L["out_dtype"] == "uint8" else onnx.TensorProto.INT8
                m.graph.output.append(onnx.helper.make_tensor_value_info(L["out_tensor"], et, None))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
        self.input = ir["input"]
        self.layers = ir["layers"]

    def run(self, image_u8):
        x = np.asarray(image_u8).transpose(2, 0, 1)[None]
        x = (x.astype(np.float32) * np.float32(self.input["scale"])) if self.input["float"] else x.astype(np.uint8)
        res = self.sess.run(self.names, {self.input["tensor"]: x})
        outs = {}
        for L, r in zip(self.layers, res):
            r = r.astype(np.int16) - (128 if r.dtype == np.uint8 else 0)
            r = r[0].transpose(1, 2, 0) if r.ndim == 4 else r.reshape(1, 1, -1)
            outs[L["name"]] = r.astype(np.int8)
        return outs


def verify(prog, ir, model_path, images):
    ref = OrtReference(model_path, ir)
    v = Verification(layers=[LayerDiff(L["name"]) for L in ir["layers"]])
    names = [L["name"] for L in ir["layers"]]
    last = names[-1]
    for img in images:
        cls, _, outs, _, _ = run_sim(prog, img)
        r = ref.run(img)
        for d, n in zip(v.layers, names):
            a, b = outs[n].astype(np.int16), r[n].astype(np.int16)
            assert a.shape == b.shape, f"{n}: NPU {a.shape} vs ONNX {b.shape}"
            diff = np.abs(a - b)
            d.values += diff.size
            d.exact += int((diff == 0).sum())
            d.max_diff = max(d.max_diff, int(diff.max()))
        v.argmax_agree += cls == int(np.argmax(r[last].reshape(-1)))
        v.images += 1
    return v
