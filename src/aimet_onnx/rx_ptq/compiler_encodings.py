"""Convert live AIMET ONNX quantizers to the compiler-native schema."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import onnx

from aimet_common.quantsim_config.compute_ops import is_grid_preserving_op


_VECTOR_FIELDS = {"scale", "n", "zero_point", "real_min", "real_max"}


def _po2_exponent(scale: float) -> int:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale 必须为正数且有限: {scale}")
    exponent = int(round(-math.log2(scale)))
    expected = 2.0 ** (-exponent)
    if not math.isclose(scale, expected, rel_tol=1e-5, abs_tol=1e-12):
        raise ValueError(f"scale 不是 Po2: {scale}")
    return exponent


def _map_recursive(value: Any, fn) -> Any:
    if isinstance(value, list):
        return [_map_recursive(item, fn) for item in value]
    return fn(value)


def _normalize_encoding(entry: Dict[str, Any]) -> Dict[str, Any]:
    bitwidth = int(entry["bitwidth"])
    dtype = str(entry.get("dtype", "int")).upper()
    if dtype == "INT":
        dtype = f"INT{bitwidth}"
    elif dtype == "FLOAT":
        dtype = f"FLOAT{bitwidth}"

    scale = entry.get("scale")
    if scale is None:
        raise KeyError(f"encoding 缺少 scale: {entry}")
    result: Dict[str, Any] = {
        "bitwidth": bitwidth,
        "dtype": dtype,
        "is_symmetric": str(entry.get("is_symmetric", "False")),
        "scale": scale,
        "n": _map_recursive(scale, lambda item: _po2_exponent(float(item))),
    }

    zero_point = entry.get("zero_point", entry.get("offset", 0))
    result["zero_point"] = zero_point
    result["real_min"] = entry.get("real_min", entry.get("min"))
    result["real_max"] = entry.get("real_max", entry.get("max"))
    if result["real_min"] is None or result["real_max"] is None:
        raise KeyError(f"encoding 缺少 real_min/real_max: {entry}")
    return result


def _compact_encodings(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = [_normalize_encoding(entry) for entry in entries]
    if not normalized:
        raise ValueError("不能压缩空 encoding 列表")
    if len(normalized) == 1:
        return normalized[0]

    keys = set().union(*(entry.keys() for entry in normalized))
    result: Dict[str, Any] = {}
    for key in sorted(keys):
        values = [entry.get(key) for entry in normalized]
        if key in _VECTOR_FIELDS or any(value != values[0] for value in values[1:]):
            result[key] = values
        else:
            result[key] = values[0]
    return result


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
        return _compact_encodings(raw)

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
        tensor_encodings: Dict[str, Dict[str, Any]] = {}
        enabled_activations: List[str] = []
        for name, quantizer in self.sim.qc_quantize_op_dict.items():
            if not quantizer.enabled:
                continue
            encoding = self._encoding_from_quantizer(name, quantizer)
            if name in self.graph.initializers:
                self.params[name] = encoding
                self.mapped.add(name)
                continue
            tensor_encodings[name] = encoding
            enabled_activations.append(name)
            self.mapped.add(name)

        self._propagate_grid_preserving(tensor_encodings)

        for name, encoding in tensor_encodings.items():
            if self._map_activation(name, encoding):
                continue
            if name in enabled_activations:
                self.unmapped.append(name)

        if self.unmapped:
            raise RuntimeError(
                "enabled quantizer 无法映射到原始 ONNX node-io/initializer: "
                + ", ".join(sorted(self.unmapped))
            )

        quantizer_args = dict(getattr(self.sim, "quant_args", {}) or {})
        payload = {
            "activation_encodings": self.activation,
            "excluded_layers": list(getattr(self.sim, "_excluded_layer_names", []) or []),
            "param_encodings": self.params,
            "quantizer_args": quantizer_args,
            "version": "1.0.0",
            "schema_version": 3,
        }
        report = {
            "enabled_quantizers": len(self.mapped),
            "mapped_quantizers": len(self.mapped),
            "unmapped_quantizers": sorted(self.unmapped),
            "propagated_tensors": self.propagated,
            "activation_entries": len(self.activation),
            "param_entries": len(self.params),
            "initializer_count": len(self.graph.initializers),
        }
        return payload, report
