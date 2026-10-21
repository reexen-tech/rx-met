"""Convert live AIMET ONNX quantizers to the compiler-native schema."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import onnx

from aimet_common.compiler_encodings import compact_encoding_records, normalize_compiler_encodings
from aimet_common.quantsim_config.compute_ops import is_grid_preserving_op
from .validation import validate_quantizer_state


@dataclass(frozen=True)
class _GraphIndex:
    producers: Dict[str, List[Tuple[int, int]]]
    consumers: Dict[str, List[Tuple[int, int]]]
    initializers: set[str]
    node_names: Dict[int, str]

    @classmethod
    def build(cls, model: onnx.ModelProto) -> "_GraphIndex":
        producers: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        consumers: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        node_names: Dict[int, str] = {}
        for index, node in enumerate(model.graph.node):
            node_names[index] = node.name or f"{node.op_type}_{index}"
            for output_index, tensor in enumerate(node.output):
                if tensor:
                    producers[tensor].append((index, output_index))
            for input_index, tensor in enumerate(node.input):
                if tensor:
                    consumers[tensor].append((index, input_index))
        return cls(
            producers=dict(producers),
            consumers=dict(consumers),
            initializers={item.name for item in model.graph.initializer},
            node_names=node_names,
        )

    def data_inputs(self, node: onnx.NodeProto) -> List[str]:
        return [
            tensor
            for tensor in node.input
            if tensor and tensor not in self.initializers
        ]


class CompilerEncodingConverter:
    """Build the node-I/O compiler schema from an ONNX QuantSim."""

    def __init__(self, sim: Any, original_model: onnx.ModelProto):
        self.sim = sim
        self.graph = _GraphIndex.build(original_model)
        self.original_model = original_model
        self.activation: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.params: Dict[str, Dict[str, Any]] = {}
        self.mapped: set[str] = set()
        self.unmapped: List[str] = []
        self.propagated: List[str] = []

    def _add_activation(
        self, node_index: int, io_name: str, io_index: int, encoding: Dict[str, Any]
    ) -> None:
        node_name = self.graph.node_names[node_index]
        if node_name in getattr(self.sim, "_rx_native_grus", {}):
            return  # The native operator supplies its complete port/internal schema.
        node_entry = self.activation.setdefault(node_name, {})
        io_entry = node_entry.setdefault(io_name, {})
        key = str(io_index)
        if key in io_entry and io_entry[key] != encoding:
            raise RuntimeError(
                f"同一 node-io 存在冲突 encoding: {node_name}.{io_name}[{key}]"
            )
        io_entry[key] = copy.deepcopy(encoding)

    def _map_activation(self, tensor_name: str, encoding: Dict[str, Any]) -> bool:
        mapped = False
        for node_index, input_index in self.graph.consumers.get(tensor_name, []):
            port = (self.graph.node_names[node_index], input_index)
            alias = getattr(self.sim, "_rx_input_quantizers", {}).get(port)
            if alias is None or not self.sim.qc_quantize_op_dict[alias].enabled:
                self._add_activation(node_index, "input", input_index, encoding)
            mapped = True
        for node_index, output_index in self.graph.producers.get(tensor_name, []):
            self._add_activation(node_index, "output", output_index, encoding)
            mapped = True
        return mapped

    def _encoding_from_quantizer(self, name: str, quantizer: Any) -> Dict[str, Any]:
        raw = quantizer.export_encodings("0.6.1")
        if raw is None:
            raise RuntimeError(f"enabled quantizer 没有 encoding: {name}")
        symmetric = bool(getattr(quantizer, "use_symmetric_encodings", False))
        unsigned = not symmetric or (
            bool(getattr(quantizer, "use_unsigned_symmetric", False))
            and all(entry.get("offset", 0) == 0 for entry in raw)
        )
        strict = bool(getattr(quantizer, "use_strict_symmetric", False))
        enc_type = quantizer._encoding_type().name
        entries = []
        for entry in raw:
            entry = dict(entry)
            entry["enc_type"] = enc_type
            if str(entry.get("dtype", "")).lower() == "int":
                bits = int(entry["bitwidth"])
                qmin = -(1 << (bits - 1)) + int(strict) if not unsigned else 0
                # TfEncoding uses an unsigned index grid. Translate its additive
                # offset to the effective compiler integer grid, preserving QDQ.
                offset = entry.pop("offset")
                if offset != int(offset):
                    raise ValueError(f"{name}: TfEncoding offset must be an integer")
                entry["zero_point"] = int(offset) - qmin
                entry["dtype"] = f"{'UINT' if unsigned else 'INT'}{bits}"
            entries.append(entry)
        return compact_encoding_records(entries, force_list=name in self.graph.initializers)

    def _propagate_grid_preserving(
        self, tensor_encodings: Dict[str, Dict[str, Any]]
    ) -> None:
        """Copy encodings through reshape/maxpool/pad so compute I/O is complete.

        Simulation still uses one quantizer per tensor. This only fills compiler
        sidecar slots for grid-preserving edges that AIMET left disabled.
        """
        changed = True
        while changed:
            changed = False
            for node in self.original_model.graph.node:
                if not is_grid_preserving_op(node.op_type):
                    continue
                data_inputs = self.graph.data_inputs(node)
                data_outputs = [tensor for tensor in node.output if tensor]
                if not data_inputs or not data_outputs:
                    continue
                source = next(
                    (
                        tensor_encodings[name]
                        for name in data_inputs
                        if name in tensor_encodings
                    ),
                    None,
                )
                if source is None:
                    source = next(
                        (
                            tensor_encodings[name]
                            for name in data_outputs
                            if name in tensor_encodings
                        ),
                        None,
                    )
                if source is None:
                    continue
                for name in data_inputs + data_outputs:
                    if name in tensor_encodings:
                        continue
                    tensor_encodings[name] = copy.deepcopy(source)
                    self.propagated.append(name)
                    changed = True

    def convert(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        validate_quantizer_state(self.sim)
        tensor_encodings: Dict[str, Dict[str, Any]] = {}
        port_encodings: Dict[str, Dict[str, Any]] = {}
        ports = getattr(self.sim, "_rx_input_quantizers", {})
        port_names = set(ports.values())
        enabled_activations: List[str] = []
        for name, quantizer in self.sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue
            encoding = self._encoding_from_quantizer(name, quantizer)
            if name in port_names:
                port_encodings[name] = encoding
                self.mapped.add(name)
                continue
            if name in self.graph.initializers:
                self.params[name] = encoding
                self.mapped.add(name)
                continue
            tensor_encodings[name] = encoding
            enabled_activations.append(name)
            self.mapped.add(name)

        from .recurrent import export_native_encodings, native_grus
        native_payload = export_native_encodings(self.sim)
        for name, adapter in native_grus(self.sim).items():
            native_activation = native_payload["activation_encodings"].get(name)
            if native_activation is not None:
                for output in adapter.outputs:
                    if output:
                        tensor_encodings.setdefault(output, native_activation["output"]["0"])
        self._propagate_grid_preserving(tensor_encodings)

        for name, encoding in tensor_encodings.items():
            if self._map_activation(name, encoding):
                continue
            if name in enabled_activations:
                self.unmapped.append(name)

        node_indices = {name: index for index, name in self.graph.node_names.items()}
        for (node_name, input_index), alias in ports.items():
            if alias in port_encodings:
                self._add_activation(node_indices[node_name], "input", input_index, port_encodings[alias])

        if self.unmapped:
            raise RuntimeError(
                "enabled quantizer 无法映射到原始 ONNX node-io/initializer: "
                + ", ".join(sorted(self.unmapped))
            )

        quantizer_args = dict(getattr(self.sim, "quant_args", {}) or {})
        self.activation.update(native_payload["activation_encodings"])
        self.params.update(native_payload["param_encodings"])
        payload = {
            "activation_encodings": self.activation,
            "excluded_layers": list(dict.fromkeys(
                list(getattr(self.sim, "_excluded_layer_names", []) or []) + native_payload["excluded_layers"]
            )),
            "param_encodings": self.params,
            "quantizer_args": quantizer_args,
            "version": "1.0.0",
            "schema_version": 3,
        }
        report = {
            "enabled_quantizers": len(self.mapped),
            "enabled_quantizer_names": sorted(self.mapped),
            "mapped_quantizers": len(self.mapped),
            "unmapped_quantizers": sorted(self.unmapped),
            "propagated_tensors": self.propagated,
            "activation_entries": len(self.activation),
            "param_entries": len(self.params),
            "initializer_count": len(self.graph.initializers),
            "native_recurrent_nodes": sorted(native_grus(self.sim)),
            "enabled_native_recurrent_nodes": sorted(
                name for name, adapter in native_grus(self.sim).items() if adapter.module.use_quantization
            ),
        }
        return normalize_compiler_encodings(payload, require_po2=True), report
