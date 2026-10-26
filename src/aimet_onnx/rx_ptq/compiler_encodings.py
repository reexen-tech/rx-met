"""Convert live AIMET ONNX quantizers to the compiler-native schema."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import onnx

from aimet_common.compiler_encodings import (
    COMPILER_SCHEMA_VERSION, compact_encoding_records, finalize_compiler_encodings,
)
from aimet_common.compiler_graph_validation import (
    common_native_recurrent_output_grid, validate_compiler_encoding_graph,
)
from aimet_common.quantization_grid import IntegerGrid, integer_bounds
from aimet_common.quantsim_config.compute_ops import is_grid_preserving_op
from aimet_onnx.quantization_roles import QuantizationInputRoles
from .validation import validate_quantizer_state

_FLOAT_TENSOR_TYPES = frozenset({
    onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16,
    onnx.TensorProto.DOUBLE, onnx.TensorProto.BFLOAT16,
})


@dataclass(frozen=True)
class _GraphIndex:
    producers: Dict[str, List[Tuple[int, int]]]
    consumers: Dict[str, List[Tuple[int, int]]]
    initializers: set[str]
    node_names: Dict[int, str]
    tensor_types: Dict[str, int]

    @classmethod
    def build(cls, model: onnx.ModelProto, activation_dtypes=None) -> "_GraphIndex":
        producers: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        consumers: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        node_names: Dict[int, str] = {}
        tensor_types = {name: onnx.helper.np_dtype_to_tensor_dtype(dtype)
                        for name, dtype in (activation_dtypes or {}).items()}
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
            if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type:
                tensor_types[value.name] = value.type.tensor_type.elem_type
        tensor_types.update({value.name: value.data_type for value in model.graph.initializer})
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
            tensor_types=tensor_types,
        )

    def data_inputs(self, node: onnx.NodeProto) -> List[str]:
        # Grid-preserving ops carry values on port 0. Other inputs are indices,
        # shapes, masks, ROI coordinates, or padding controls, even if floating.
        return [
            tensor
            for tensor in node.input[:1]
            if tensor not in self.initializers and self.is_floating(tensor)
        ]

    def data_outputs(self, node: onnx.NodeProto) -> List[str]:
        return [tensor for tensor in node.output if self.is_floating(tensor)]

    def is_floating(self, tensor: str) -> bool:
        return self.tensor_types.get(tensor) in _FLOAT_TENSOR_TYPES

    def validate_encoding_type(self, tensor: str) -> None:
        dtype = self.tensor_types.get(tensor)
        if dtype is not None and dtype not in _FLOAT_TENSOR_TYPES:
            raise ValueError(f"Quantization encoding on non-floating ONNX tensor: {tensor} "
                             f"({onnx.TensorProto.DataType.Name(dtype)})")


class CompilerEncodingConverter:
    """Build the node-I/O compiler schema from an ONNX QuantSim."""

    def __init__(self, sim: Any, original_model: onnx.ModelProto):
        self.sim = sim
        self.graph = _GraphIndex.build(original_model, getattr(sim, "activation_dtypes", {}))
        self.original_model = original_model
        self.input_roles = QuantizationInputRoles(original_model)
        self.activation: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.params: Dict[str, Dict[str, Any]] = {}
        self.mapped: set[str] = set()
        self.unmapped: List[str] = []
        self.propagated: List[str] = []
        self.mixed_native_outputs: set[str] = set()

    def _add_activation(
        self, node_index: int, io_name: str, io_index: int, encoding: Dict[str, Any]
    ) -> None:
        node_name = self.graph.node_names[node_index]
        if node_name in getattr(self.sim, "_rx_native_recurrent", {}):
            return  # The native operator supplies its complete port/internal schema.
        node = self.original_model.graph.node[node_index]
        if io_name == "input" and self.input_roles.is_control_port(node_name, io_index):
            raise ValueError(f"Quantization encoding on ONNX control port: {node_name}.input[{io_index}]")
        tensor = node.input[io_index] if io_name == "input" else node.output[io_index]
        self.graph.validate_encoding_type(tensor)
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
            if self.input_roles.is_control_port(*port):
                continue
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
        signed = getattr(quantizer, 'signed', True)
        strict = bool(getattr(quantizer, "use_strict_symmetric", False))
        enc_type = quantizer._encoding_type().name
        entries = []
        for entry in raw:
            entry = dict(entry)
            entry["enc_type"] = enc_type
            if str(entry.get("dtype", "")).lower() == "int":
                bits = int(entry["bitwidth"])
                qmin, qmax = integer_bounds(bits, signed, strict=strict and symmetric)
                # TfEncoding uses an unsigned index grid. Convert its additive
                # offset to a subtractive zero point in the compiler's code domain.
                offset = entry.pop("offset")
                if offset != int(offset):
                    raise ValueError(f"{name}: TfEncoding offset must be an integer")
                grid = IntegerGrid.from_offset(entry['scale'], offset, bits, signed,
                    index_domain=True, qmin=qmin, qmax=qmax)
                entry["zero_point"] = grid.zero_point
                entry["dtype"] = grid.dtype
                entry.pop('signed', None)
                if (qmin, qmax) != integer_bounds(bits, signed):
                    entry['qmin'], entry['qmax'] = qmin, qmax
            entries.append(entry)
        if symmetric and any(entry.get("zero_point", 0) != 0 for entry in entries):
            # Legacy unsigned-index calibration may rebase a positive-only
            # channel into a signed target grid. Describe that target honestly.
            for entry in entries:
                entry["is_symmetric"] = "False"
        return compact_encoding_records(entries, force_list=name in self.graph.initializers)

    def _track_mixed_native_outputs(self, tensor_encodings, ports):
        pending = list(self.mixed_native_outputs)
        visited = set()
        while pending:
            output = pending.pop()
            if output in visited:
                continue
            visited.add(output)
            for node_index, input_index in self.graph.consumers.get(output, []):
                consumer = self.original_model.graph.node[node_index]
                if consumer.op_type in ("Shape", "Size"):
                    continue
                if is_grid_preserving_op(consumer.op_type) and input_index == 0:
                    # A real output QDQ ends compound directional ownership.
                    # Its grid must never be propagated back onto the input.
                    for name in self.graph.data_outputs(consumer):
                        if name not in tensor_encodings and name not in self.mixed_native_outputs:
                            self.mixed_native_outputs.add(name)
                            pending.append(name)
                    continue
                if consumer.name in getattr(self.sim, "_rx_native_recurrent", {}) and input_index == 0:
                    continue
                alias = ports.get((consumer.name, input_index))
                if alias is None or not self.sim.qc_quantize_op_dict[alias].enabled:
                    raise ValueError(
                        f"{output} -> {consumer.name}.input[{input_index}]: different native direction grids "
                        "require an independent compute input quantizer"
                    )

    def _propagate_grid_preserving(
        self, tensor_encodings: Dict[str, Dict[str, Any]]
    ) -> None:
        """Copy encodings through reshape/maxpool/pad so compute I/O is complete.

        Only floating data ports participate; index, shape, mask, and other
        control ports retain their ONNX types. Live quantizers are unchanged.
        """
        changed = True
        while changed:
            changed = False
            for node in self.original_model.graph.node:
                if not is_grid_preserving_op(node.op_type):
                    continue
                if self.input_roles.is_control_port(node.name, 0):
                    continue
                data_inputs = self.graph.data_inputs(node)
                data_outputs = self.graph.data_outputs(node)
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
                    if name in tensor_encodings or name in self.mixed_native_outputs:
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
            self.graph.validate_encoding_type(name)
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

        from .recurrent import export_native_encodings, native_recurrent
        native_payload = export_native_encodings(self.sim)
        for name, adapter in native_recurrent(self.sim).items():
            native_activation = native_payload["activation_encodings"].get(name)
            if native_activation is not None:
                node = next(node for node in self.original_model.graph.node if node.name == name)
                for index, output in enumerate(adapter.outputs):
                    if output:
                        encoding = common_native_recurrent_output_grid(node, native_activation, index)
                        if encoding is not None:
                            tensor_encodings.setdefault(output, encoding)
                            continue
                        # A packed ONNX tensor can contain independently calibrated
                        # directions. Its forward summary is never its tensor grid.
                        self.mixed_native_outputs.add(output)
        self._track_mixed_native_outputs(tensor_encodings, ports)
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
            "schema_version": COMPILER_SCHEMA_VERSION,
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
            "native_recurrent_nodes": sorted(native_recurrent(self.sim)),
            "enabled_native_recurrent_nodes": sorted(
                name for name, adapter in native_recurrent(self.sim).items() if adapter.module.use_quantization
            ),
        }
        payload = finalize_compiler_encodings(payload, require_po2=True)
        validate_compiler_encoding_graph(self.original_model, payload)
        return payload, report
