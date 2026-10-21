"""ONNX execution adapter for operator-owned recurrent quantization."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx

from aimet_onnx.quantsim import QuantizationSimModel


def native_grus(sim):
    return getattr(sim, "_rx_native_grus", {})


def export_native_encodings(sim):
    payload = {"activation_encodings": {}, "param_encodings": {}, "excluded_layers": []}
    for adapter in native_grus(sim).values():
        part = adapter.export_encodings()
        for key in ("activation_encodings", "param_encodings"):
            overlap = payload[key].keys() & part[key].keys()
            if overlap:
                raise ValueError(f"Conflicting native recurrent encodings: {sorted(overlap)}")
            payload[key].update(part[key])
        payload["excluded_layers"].extend(part["excluded_layers"])
    return payload


def restore_native_encodings(sim, payload, *, allow_overwrite):
    adapters = native_grus(sim)
    if not adapters:
        if payload.get("native_recurrent_encodings"):
            raise ValueError("Encodings require native recurrent operators absent from the model")
        return
    saved = payload.get("native_recurrent_encodings")
    if saved is None:
        raise ValueError("Missing native GRU encodings; generic ONNX grids cannot restore GRU internals")
    actual = set(saved.get("activation_encodings", {})) | set(saved.get("excluded_layers", []))
    if actual != set(adapters):
        raise ValueError("Native GRU encoding nodes differ from the ONNX graph")
    enabled = set(saved.get("activation_encodings", {}))
    if enabled & set(saved.get("excluded_layers", [])):
        raise ValueError("Native GRU cannot have both encodings and an exclusion")
    parameters = {name for node in enabled for name in adapters[node].parameter_names}
    if set(saved.get("param_encodings", {})) != parameters:
        raise ValueError("Native GRU parameter encodings differ from the ONNX graph")
    from aimet_common.compiler_encodings import validate_compiler_encodings
    validate_compiler_encodings(saved)
    for adapter in adapters.values():
        adapter.load_encodings(saved, allow_overwrite=allow_overwrite)


class NativeGRUSim(QuantizationSimModel):
    """Keep generic QuantSim behavior while delegating GRU to its native package."""

    def __init__(self, model, **kwargs):
        try:
            from quant_gru_onnx import DOMAIN, OnnxGRU, runtime_library
        except ImportError as exc:
            raise RuntimeError("ONNX GRU PTQ requires the updated quant_gru package") from exc
        self._rx_native_library = runtime_library()
        self._rx_native_domain = DOMAIN
        providers = kwargs.get("providers") or ("CPUExecutionProvider",)
        device = None
        for provider in providers:
            name, options = provider if isinstance(provider, tuple) else (provider, {})
            if name == "CUDAExecutionProvider":
                device = f"cuda:{options.get('device_id', 0)}"
                break
        if device is None:
            raise RuntimeError("Native ONNX GRU calibration requires CUDAExecutionProvider")
        self._rx_original_onnx = copy.deepcopy(model)
        initializers = {value.name: value for value in model.graph.initializer}
        nodes = [node for node in model.graph.node if node.op_type == "GRU" and not node.domain]
        if len({node.name for node in nodes}) != len(nodes):
            raise ValueError("Native ONNX GRU nodes must have unique names")
        scheme = str(kwargs.get("quant_scheme", "minmax")).lower()
        method = "percentile" if "percentile" in scheme else "sqnr" if "enhanced" in scheme else "minmax"
        self._rx_native_grus = {
            node.name: OnnxGRU(node, initializers, device=device, method=method)
            for node in nodes
        }
        self._rx_native_parameters = set().union(*(a.parameter_names for a in self._rx_native_grus.values()))
        parameter_uses = [name for adapter in self._rx_native_grus.values() for name in adapter.inputs[1:4]]
        if len(set(parameter_uses)) != len(parameter_uses):
            raise NotImplementedError("Shared native GRU weight initializers are not supported")
        if self._rx_native_parameters & {item.name for item in model.graph.input}:
            raise NotImplementedError("Overridable native GRU weight inputs are not supported")
        self._rx_native_outputs = {name for a in self._rx_native_grus.values() for name in a.outputs if name}
        # Consumers such as Linear still need their own activation grid for
        # input precision and accumulator bias. Terminal GRU I/O is native-owned.
        self._rx_native_io = {
            name for name in self._rx_native_outputs
            if not any(name in node.input for node in model.graph.node)
        }
        for adapter in self._rx_native_grus.values():
            for name in (adapter.inputs[0], adapter.inputs[5]):
                consumers = [node for node in model.graph.node if name and name in node.input]
                if name and all(node.name in self._rx_native_grus for node in consumers):
                    self._rx_native_io.add(name)
        for node in model.graph.node:
            if node.name not in self._rx_native_grus and set(node.input) & self._rx_native_parameters:
                raise NotImplementedError("Native GRU weights shared with other operators are not supported")
        super().__init__(model, **kwargs)
        self.session._rx_native_grus = self._rx_native_grus

    def _prepare_runtime_model(self):
        self._ort_session_options.register_custom_ops_library(self._rx_native_library)
        for node in self.model.model.graph.node:
            if node.name in self._rx_native_grus and node.op_type == "GRU":
                node.CopyFrom(self._rx_native_grus[node.name].runtime_node(node))
        for value in list(self.model.model.graph.initializer):
            if value.name in self._rx_native_parameters:
                self.model.model.graph.initializer.remove(value)
        self.model.model.opset_import.append(onnx.helper.make_opsetid(self._rx_native_domain, 1))

    def _get_param_names(self):
        # Generic QuantSim must not create competing quantizers for native GRU.
        for name in self._rx_native_grus:
            self.connected_graph.get_all_ops()[name].parameters.clear()
        super()._get_param_names()

    def _is_tensor_quantizable(self, name):
        if name in self._rx_native_parameters or name in self._rx_native_io:
            return False
        return super()._is_tensor_quantizable(name)

    def _rebuild_session(self):
        super()._rebuild_session()
        # A caller may retain session after dropping sim. Keep callback contexts alive.
        self.session._rx_native_grus = self._rx_native_grus

    def set_percentile_value(self, percentile_value):
        super().set_percentile_value(percentile_value)
        for adapter in self._rx_native_grus.values():
            adapter.module.percentile_value = percentile_value

    def _export_encodings(self, encoding_file_path):
        super()._export_encodings(encoding_file_path)
        path = Path(encoding_file_path)
        payload = json.loads(path.read_text())
        payload["native_recurrent_encodings"] = export_native_encodings(self)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def export(self, path, filename_prefix, export_model=True):
        super().export(path, filename_prefix, export_model=False)
        if export_model:
            model = copy.deepcopy(self._rx_original_onnx)
            onnx.external_data_helper.convert_model_from_external_data(model)
            onnx.save_model(model, str(Path(path) / f"{filename_prefix}.onnx"))

    def _to_onnx_qdq(self, **kwargs):
        raise NotImplementedError("Native GRU requires clean ONNX plus sidecar encodings; use export_onnx_json")
