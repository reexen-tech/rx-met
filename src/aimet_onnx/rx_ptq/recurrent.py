"""ONNX execution adapter for operator-owned recurrent quantization."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import onnx

from .input_quantization import InputQuantizationSim


def native_grus(sim):
    return getattr(sim, "_rx_native_grus", {})


def native_recurrent(sim):
    return getattr(sim, "_rx_native_recurrent", native_grus(sim))


def export_native_encodings(sim):
    payload = {"activation_encodings": {}, "param_encodings": {}, "excluded_layers": []}
    for adapter in native_recurrent(sim).values():
        part = adapter.export_encodings()
        for key in ("activation_encodings", "param_encodings"):
            overlap = payload[key].keys() & part[key].keys()
            if overlap:
                raise ValueError(f"Conflicting native recurrent encodings: {sorted(overlap)}")
            payload[key].update(part[key])
        payload["excluded_layers"].extend(part["excluded_layers"])
    return payload


def restore_native_encodings(sim, payload, *, allow_overwrite):
    adapters = native_recurrent(sim)
    if not adapters:
        if payload.get("native_recurrent_encodings"):
            raise ValueError("Encodings require native recurrent operators absent from the model")
        return
    saved = payload.get("native_recurrent_encodings")
    if saved is None:
        raise ValueError("Missing native recurrent encodings; generic ONNX grids cannot restore recurrent internals")
    actual = set(saved.get("activation_encodings", {})) | set(saved.get("excluded_layers", []))
    if actual != set(adapters):
        raise ValueError("Native recurrent encoding nodes differ from the ONNX graph")
    enabled = set(saved.get("activation_encodings", {}))
    if enabled & set(saved.get("excluded_layers", [])):
        raise ValueError("Native recurrent node cannot have both encodings and an exclusion")
    parameters = {name for node in enabled for name in adapters[node].parameter_names}
    if set(saved.get("param_encodings", {})) != parameters:
        raise ValueError("Native recurrent parameter encodings differ from the ONNX graph")
    from aimet_common.compiler_encodings import validate_compiler_encodings
    validate_compiler_encodings(saved)
    for adapter in adapters.values():
        adapter.load_encodings(saved, allow_overwrite=allow_overwrite)


class NativeRecurrentSim(InputQuantizationSim):
    """Delegate recurrent arithmetic and grids to operator-owned adapters."""

    def __init__(self, model, **kwargs):
        providers = kwargs.get("providers") or ("CPUExecutionProvider",)
        device = "cpu"
        for provider in providers:
            name, options = provider if isinstance(provider, tuple) else (provider, {})
            if name == "CUDAExecutionProvider":
                device = f"cuda:{options.get('device_id', 0)}"
                break
        initializers = {value.name: value for value in model.graph.initializer}
        nodes = [node for node in model.graph.node if node.op_type in ("GRU", "LSTM") and not node.domain]
        if len({node.name for node in nodes}) != len(nodes):
            raise ValueError("Native ONNX recurrent nodes must have unique names")
        scheme = str(kwargs.get("quant_scheme", "minmax")).lower()
        method = "percentile" if "percentile" in scheme else "sqnr" if "enhanced" in scheme else "minmax"
        self._rx_native_recurrent = {}
        self._rx_native_runtimes = {}
        for node in nodes:
            package = f"quant_{node.op_type.lower()}_onnx"
            try:
                backend = importlib.import_module(package)
            except ImportError as exc:
                raise RuntimeError(f"ONNX {node.op_type} PTQ requires the updated {package} package") from exc
            if backend.DOMAIN not in self._rx_native_runtimes:
                self._rx_native_runtimes[backend.DOMAIN] = backend.runtime_library()
            adapter = getattr(backend, f"Onnx{node.op_type}")(node, initializers, device=device, method=method)
            self._rx_native_recurrent[node.name] = adapter
        # Retain the existing GRU inspection interface.
        self._rx_native_grus = {node.name: self._rx_native_recurrent[node.name]
                                for node in nodes if node.op_type == "GRU"}
        self._rx_native_parameters = set().union(*(a.parameter_names for a in self._rx_native_recurrent.values()))
        parameter_uses = [name for adapter in self._rx_native_recurrent.values()
                          for name in adapter.inputs[1:4] if name]
        if len(set(parameter_uses)) != len(parameter_uses):
            raise NotImplementedError("Shared native recurrent weight initializers are not supported")
        if self._rx_native_parameters & {item.name for item in model.graph.input}:
            raise NotImplementedError("Overridable native recurrent weight inputs are not supported")
        self._rx_native_outputs = {name for a in self._rx_native_recurrent.values() for name in a.outputs if name}
        # Consumers such as Linear still need their own activation grid for
        # input precision and accumulator bias. Terminal recurrent I/O is native-owned.
        self._rx_native_io = {
            name for name in self._rx_native_outputs
            if not any(name in node.input for node in model.graph.node)
        }
        for adapter in self._rx_native_recurrent.values():
            for name in (adapter.inputs[0], *adapter.inputs[5:]):
                consumers = [node for node in model.graph.node if name and name in node.input]
                if name and all(node.name in self._rx_native_recurrent for node in consumers):
                    self._rx_native_io.add(name)
        for node in model.graph.node:
            if node.name not in self._rx_native_recurrent and set(node.input) & self._rx_native_parameters:
                raise NotImplementedError("Native recurrent weights shared with other operators are not supported")
        super().__init__(model, **kwargs)
        self.session._rx_native_recurrent = self._rx_native_recurrent

    def _prepare_runtime_model(self):
        super()._prepare_runtime_model()
        for library in self._rx_native_runtimes.values():
            self._ort_session_options.register_custom_ops_library(library)
        for node in self.model.model.graph.node:
            if node.name in self._rx_native_recurrent and node.op_type in ("GRU", "LSTM"):
                node.CopyFrom(self._rx_native_recurrent[node.name].runtime_node(node))
        for value in list(self.model.model.graph.initializer):
            if value.name in self._rx_native_parameters:
                self.model.model.graph.initializer.remove(value)
        for domain in self._rx_native_runtimes:
            self.model.model.opset_import.append(onnx.helper.make_opsetid(domain, 1))

    def _get_param_names(self):
        # Generic QuantSim must not create competing quantizers for native operators.
        for name in self._rx_native_recurrent:
            self.connected_graph.get_all_ops()[name].parameters.clear()
        super()._get_param_names()

    def _is_tensor_quantizable(self, name):
        if name in self._rx_native_parameters or name in self._rx_native_io:
            return False
        return super()._is_tensor_quantizable(name)

    def _rebuild_session(self):
        super()._rebuild_session()
        # A caller may retain session after dropping sim. Keep callback contexts alive.
        self.session._rx_native_recurrent = self._rx_native_recurrent

    def set_percentile_value(self, percentile_value):
        super().set_percentile_value(percentile_value)
        for adapter in self._rx_native_recurrent.values():
            adapter.module.percentile_value = percentile_value

    def _export_encodings(self, encoding_file_path):
        super()._export_encodings(encoding_file_path)
        path = Path(encoding_file_path)
        payload = json.loads(path.read_text())
        payload["native_recurrent_encodings"] = export_native_encodings(self)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _to_onnx_qdq(self, **kwargs):
        raise NotImplementedError("Native recurrent operators require clean ONNX plus sidecar encodings; use export_onnx_json")


NativeGRUSim = NativeRecurrentSim
