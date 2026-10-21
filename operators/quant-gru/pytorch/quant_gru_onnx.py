"""ONNX adapter for the operator-owned QuantGRU implementation.

Only layout conversion and runtime dispatch live here. Calibration, integer
execution, Po2 grids and encoding packing belong to QuantGRU.
"""

from __future__ import annotations

import copy
import importlib
import itertools
from pathlib import Path
import weakref

import numpy as np
import onnx
import torch

from quant_gru import QuantGRU, reorder_weights_pytorch_to_haste

DOMAIN = "rx.quant_gru"
OP_TYPE = "NativeGRU"
_contexts = weakref.WeakValueDictionary()
_identifiers = itertools.count(1)


def runtime_library() -> str:
    try:
        module = importlib.import_module("quant_gru_onnx_runtime")
    except ImportError as exc:
        raise RuntimeError(
            "ONNX GRU PTQ requires the quant_gru ONNX runtime extension; "
            "rebuild/install the updated quant_gru wheel"
        ) from exc
    return str(Path(module.__file__).resolve())


def _run(identifier, value, initial_h):
    try:
        adapter = _contexts[identifier]
    except KeyError as exc:
        raise RuntimeError("ONNX GRU session no longer owns its runtime context") from exc
    return adapter.run(value, initial_h)


class OnnxGRU:
    """A standard ONNX GRU backed by the existing native QuantGRU interface."""

    def __init__(self, node, initializers, *, device="cuda", method="minmax", percentile=99.99):
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        self.name = node.name
        if not self.name:
            raise ValueError("ONNX GRU PTQ requires a nonempty unique node name")
        prefix = f"GRU {self.name}: "
        direction = attrs.get("direction", b"forward").decode()
        if direction not in ("forward", "bidirectional"):
            raise NotImplementedError(prefix + f"unsupported direction={direction}")
        if attrs.get("linear_before_reset", 0) != 1:
            raise NotImplementedError(prefix + "native QuantGRU requires linear_before_reset=1")
        expected_activations = [b"Sigmoid", b"Tanh"] * (2 if direction == "bidirectional" else 1)
        if attrs.get("activations", expected_activations) != expected_activations:
            raise NotImplementedError(prefix + "custom activations are not supported")
        if any(key in attrs for key in ("clip", "activation_alpha", "activation_beta")):
            raise NotImplementedError(prefix + "activation clipping/parameters are not supported")
        inputs = list(node.input) + [""] * (6 - len(node.input))
        if not inputs[3]:
            raise NotImplementedError(prefix + "a constant bias initializer is currently required")
        if inputs[4]:
            raise NotImplementedError(prefix + "sequence_lens is not supported")
        self.layout = attrs.get("layout", 0)
        if self.layout not in (0, 1):
            raise ValueError(prefix + f"invalid layout={self.layout}")
        self.inputs, self.outputs = inputs, list(node.output)
        self.parameter_names = {name for name in inputs[1:4] if name}

        def read(name):
            if name not in initializers:
                raise NotImplementedError(prefix + f"constant initializer required: {name}")
            value = onnx.numpy_helper.to_array(initializers[name])
            if value.dtype != np.float32:
                raise TypeError(prefix + f"float32 weights required: {name}")
            if not np.isfinite(value).all():
                raise ValueError(prefix + f"nonfinite parameter: {name}")
            return value.copy()

        w, r = read(inputs[1]), read(inputs[2])
        directions = 2 if direction == "bidirectional" else 1
        hidden = attrs.get("hidden_size")
        if hidden is None or w.ndim != 3 or w.shape[:2] != (directions, 3 * hidden):
            raise ValueError(prefix + "invalid W shape or hidden_size")
        if r.shape != (directions, 3 * hidden, hidden):
            raise ValueError(prefix + "invalid R shape")
        bias = read(inputs[3]) if inputs[3] else np.zeros((directions, 6 * hidden), np.float32)
        if bias.shape != (directions, 6 * hidden):
            raise ValueError(prefix + "invalid B shape")
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError(prefix + "native QuantGRU calibration requires CUDA")
        self.module = QuantGRU(w.shape[2], hidden, batch_first=self.layout == 1,
                               bidirectional=directions == 2, use_quantization=True).to(device).eval()
        self.module.set_module_name(self.name)
        self.module.calibration_method = method
        self.module.percentile_value = percentile
        with torch.no_grad():
            for index in range(directions):
                suffix = "_reverse" if index else ""
                for target, array in (("weight_ih", w[index]), ("weight_hh", r[index]),
                                      ("bias_ih", bias[index, :3 * hidden]),
                                      ("bias_hh", bias[index, 3 * hidden:])):
                    # Swapping the first two gates is its own inverse.
                    converted = reorder_weights_pytorch_to_haste(torch.from_numpy(array))
                    getattr(self.module, f"{target}_l0{suffix}").copy_(converted)
        self.identifier = next(_identifiers)
        _contexts[self.identifier] = self

    def runtime_node(self, node):
        inputs = [node.input[0], self.inputs[5]]
        outputs = list(node.output)
        if len(outputs) == 1:
            outputs.append("")
        return onnx.helper.make_node(OP_TYPE, inputs, outputs, name=node.name,
                                     domain=DOMAIN, context_id=self.identifier)

    def configure(self, config_file):
        self.module.load_bitwidth_config(str(config_file), verbose=False)

    def run(self, value, initial_h=None):
        if value.dtype != np.float32 or value.ndim != 3:
            raise ValueError(f"GRU {self.name}: expected float32 rank-3 input")
        device = self.module.weight_ih_l0.device
        x = torch.from_numpy(np.asarray(value).copy()).to(device)
        h = None
        if initial_h is not None:
            if initial_h.dtype != np.float32:
                raise TypeError(f"GRU {self.name}: initial_h must be float32")
            h = torch.from_numpy(np.asarray(initial_h).copy()).to(device)
            if self.layout:
                h = h.transpose(0, 1).contiguous()
        with torch.cuda.device(device), torch.no_grad():
            output, state = self.module(x, h)
            if self.layout:
                output = output.reshape(output.shape[0], output.shape[1], self.module.num_directions,
                                        self.module.hidden_size)
                state = state.transpose(0, 1)
            else:
                output = output.reshape(output.shape[0], output.shape[1], self.module.num_directions,
                                        self.module.hidden_size).permute(0, 2, 1, 3)
            return output.contiguous().cpu().numpy(), state.contiguous().cpu().numpy()

    def export_encodings(self):
        if not self.module.use_quantization:
            return {"activation_encodings": {}, "param_encodings": {}, "excluded_layers": [self.name]}
        if not self.module.is_calibrated():
            raise RuntimeError(f"GRU {self.name}: internal quantizers are not calibrated")
        payload = self.module.export_quant_params_to_aimet_format({}, module_name=self.name)
        mapped = {}
        for suffix, original in zip(("weight_ih.weight", "weight_hh.weight", "bias"), self.inputs[1:4]):
            key = f"{self.name}.{suffix}"
            if original:
                mapped[original] = payload["param_encodings"][key]
        return {"activation_encodings": payload["activation_encodings"], "param_encodings": mapped,
                "excluded_layers": []}

    def load_encodings(self, payload, *, allow_overwrite=False):
        if self.name in payload.get("excluded_layers", []):
            self.module.use_quantization = False
            return
        canonical = copy.deepcopy(payload)
        for suffix, original in zip(("weight_ih.weight", "weight_hh.weight", "bias"), self.inputs[1:4]):
            key = f"{self.name}.{suffix}"
            if original:
                canonical["param_encodings"][key] = canonical["param_encodings"].pop(original)
        if not self.module.load_quant_params_from_aimet_format(canonical, module_name=self.name, verbose=False):
            raise ValueError(f"GRU {self.name}: missing internal encodings")
        self.module.use_quantization = True
        self.module.set_quant_params_locked(not allow_overwrite)
        try:
            restored = self.export_encodings()
        except ValueError as error:
            raise ValueError(f"GRU {self.name}: incomplete or inconsistent internal encodings: {error}") from error
        expected = {
            "activation_encodings": {self.name: payload["activation_encodings"][self.name]},
            "param_encodings": {name: payload["param_encodings"][name] for name in self.parameter_names},
            "excluded_layers": [],
        }
        def normalize_records(value):
            if isinstance(value, dict):
                if "scale" in value:
                    return self.module._convert_to_encoding_format(value)
                return {key: normalize_records(child) for key, child in value.items()}
            if isinstance(value, list):
                return [normalize_records(child) for child in value]
            return value

        expected = normalize_records(expected)
        entry = expected["activation_encodings"][self.name]
        for kind in ("input", "output"):
            port = entry[kind]
            if isinstance(port, list):
                entry[kind] = {str(index): record for index, record in enumerate(port)}
            elif isinstance(port, dict) and "scale" in port:
                entry[kind] = {"0": port}
        if restored != expected:
            raise ValueError(f"GRU {self.name}: incomplete or inconsistent internal encodings")
