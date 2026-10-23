"""Validate compiler encodings against the graph that the compiler receives.

Schema validation checks record syntax and arithmetic. This module checks tensor
ownership, actual ports and the directional grids of packed recurrent outputs.
It never repairs an encoding or derives an expected grid from a consumer record.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import onnx

from aimet_common.quantization_grid import integer_bounds
from aimet_common.quantsim_config.compute_ops import is_compute_op, is_grid_preserving_op


_GRU_INTERNAL = frozenset({
    "add_final_hidden", "mul_new_contribution", "mul_old_contribution", "mul_reset_hidden",
    "new_gate_input", "new_gate_output", "reset_gate_input", "reset_gate_output",
    "sub_one_minus_update", "update_gate_input", "update_gate_output",
    "weight_ih_linear", "weight_hh_linear",
})
_LSTM_INTERNAL = frozenset({
    "input", "output", "cell_state", "weight_ih_linear", "weight_hh_linear",
    "input_gate_input", "forget_gate_input", "cell_gate_input", "output_gate_input",
    "input_gate_output", "forget_gate_output", "cell_gate_output", "output_gate_output",
    "cell_tanh_output",
})


def _scalar_or_tuple(value):
    if isinstance(value, list):
        if len(value) == 1:
            return _scalar_or_tuple(value[0])
        return tuple(_scalar_or_tuple(item) for item in value)
    return value


def _grid(record):
    dtype, bits = record["dtype"], record["bitwidth"]
    if dtype.startswith("FLOAT"):
        return (dtype, bits)
    qmin, qmax = integer_bounds(bits, not dtype.startswith("UINT"))
    return (dtype, bits, _scalar_or_tuple(record["scale"]),
            _scalar_or_tuple(record["zero_point"]), record.get("qmin", qmin), record.get("qmax", qmax))


def native_recurrent_output_grids(node, entry, output_index):
    """Return the actual per-direction grids for an ONNX Y, Y_h or Y_c.

    Native ``output[0]`` is a forward compatibility summary, not a declaration
    that a bidirectional ONNX tensor has a single per-tensor grid.
    """
    key = ("cell_state" if output_index == 2 else "output") if node.op_type == "LSTM" else "add_final_hidden"
    forward = entry["internal_ops"][key]["output"][0]
    reverse = entry.get("internal_ops_reverse")
    return (forward, reverse[key]["output"][0]) if reverse is not None else (forward,)


def common_native_recurrent_output_grid(node, entry, output_index):
    """Return one scalar grid only when it represents every native direction."""
    records = native_recurrent_output_grids(node, entry, output_index)
    return records[0] if all(_grid(record) == _grid(records[0]) for record in records[1:]) else None


def _tensor_types(model):
    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except (onnx.shape_inference.InferenceError, ValueError) as error:
        raise ValueError(f"Cannot infer encoded ONNX tensor types: {error}") from error
    result = {value.name: value.type.tensor_type.elem_type
              for value in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)
              if value.type.HasField("tensor_type")}
    result.update({value.name: value.data_type for value in inferred.graph.initializer})
    return result


def _floating(dtype):
    if dtype in (None, onnx.TensorProto.UNDEFINED):
        return False
    name = onnx.TensorProto.DataType.Name(dtype)
    return name.startswith("FLOAT") or name in ("BFLOAT16", "DOUBLE")


def _port(entry, kind, index):
    return entry.get(kind, {}).get(str(index))


def _control_roles(model):
    transparent = {"Identity", "Cast", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten"}
    producers = {name: node for node in model.graph.node for name in node.output if name}
    consumers = defaultdict(set)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                consumers[name].add((node.name, index))
    bounds = {(node.name, index): name for node in model.graph.node
              if node.op_type == "Clip" and node.domain in ("", "ai.onnx")
              for index, name in enumerate(node.input) if index in (1, 2) and name}
    ports, control_only = set(bounds), set()
    graph_outputs = {value.name for value in model.graph.output}
    pending = list(bounds.values())
    while pending:
        name = pending.pop()
        if name in control_only or name in graph_outputs or not consumers[name] <= ports:
            continue
        control_only.add(name)
        producer = producers.get(name)
        if producer is not None and producer.domain in ("", "ai.onnx") and producer.op_type in transparent:
            ports.add((producer.name, 0))
            pending.append(producer.input[0])
    return ports, control_only


def _validate_native(node, entry, parameters, initializers):
    name = node.name
    kind = node.op_type
    if (entry.get(f"is_{kind}") is not True or
            entry.get("is_LSTM" if kind == "GRU" else "is_GRU") is True):
        raise ValueError(f"{name}: native encoding kind does not match ONNX {kind}")
    attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
    direction = attrs.get("direction", b"forward")
    if direction not in (b"forward", b"bidirectional"):
        raise ValueError(f"{name}: unsupported native direction {direction!r}")
    directions = 2 if direction == b"bidirectional" else 1
    has_reverse = "internal_ops_reverse" in entry
    if has_reverse != (directions == 2):
        raise ValueError(f"{name}: native internal_ops_reverse does not match ONNX direction")
    required = _GRU_INTERNAL if kind == "GRU" else _LSTM_INTERNAL
    for field in ("internal_ops", "internal_ops_reverse")[:directions]:
        internal = entry.get(field)
        if not isinstance(internal, dict) or set(internal) != required:
            missing = sorted(required - set(internal or {}))
            extra = sorted(set(internal or {}) - required)
            raise ValueError(f"{name}.{field}: incomplete native internal coverage (missing={missing}, extra={extra})")
        for operation, value in internal.items():
            if (not isinstance(value, dict) or set(value) != {"output"}
                    or not isinstance(value["output"], list) or len(value["output"]) != 1
                    or not isinstance(value["output"][0], dict)):
                raise ValueError(f"{name}.{field}.{operation}: expected one internal output record")
    for kind_io in ("input", "output"):
        if not isinstance(entry.get(kind_io), dict) or set(entry[kind_io]) != {"0"}:
            raise ValueError(f"{name}.{kind_io}: native compatibility summary requires only port 0")
    hidden_output = native_recurrent_output_grids(node, entry, 0)[0]
    if _grid(entry["output"]["0"]) != _grid(hidden_output):
        raise ValueError(f"{name}: output summary grid differs from forward internal output")
    if kind == "LSTM" and _grid(entry["input"]["0"]) != _grid(entry["internal_ops"]["input"]["output"][0]):
        raise ValueError(f"{name}: input summary grid differs from forward internal input")
    hidden = attrs.get("hidden_size")
    if type(hidden) is not int or hidden < 1:
        raise ValueError(f"{name}: native ONNX hidden_size must be positive")
    gates = 3 if kind == "GRU" else 4
    for index in (1, 2, 3):
        tensor = node.input[index] if index < len(node.input) else ""
        if not tensor and index == 3:
            continue
        if not tensor or tensor not in initializers or tensor not in parameters:
            raise ValueError(f"{name}.input[{index}]: missing packed native initializer/encoding {tensor!r}")
        rows = gates * hidden * (2 if index == 3 else 1)
        shape = list(initializers[tensor].dims)
        expected = [directions, rows] + ([hidden] if index == 2 else [])
        if index == 1:
            valid_shape = len(shape) == 3 and shape[:2] == expected and shape[2] > 0
        else:
            valid_shape = shape == expected
        if not valid_shape:
            raise ValueError(f"{name}: packed initializer {tensor} has invalid ONNX shape {shape}")
        for field in ("scale", "n", "zero_point", "real_min", "real_max"):
            value = parameters[tensor].get(field)
            if not isinstance(value, list) or len(value) != directions * rows or any(isinstance(item, list) for item in value):
                raise ValueError(f"{name}: packed parameter {tensor}.{field} requires {directions * rows} channel values")


def _validate_parameter_channels(model, parameters, initializers, reference_graph):
    consumers, raw_consumers = defaultdict(list), defaultdict(list)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                consumers[name].append((node, index))
    if reference_graph is not None:
        for node in reference_graph.graph.node:
            for index, name in enumerate(node.input):
                if name:
                    raw_consumers[name].append((node, index))
    for name, record in parameters.items():
        if record.get("dtype", "").startswith("FLOAT"):
            continue
        uses = consumers.get(name) or raw_consumers.get(name, [])
        native_uses = [(node, index) for node, index in uses
                       if node.op_type in ("GRU", "LSTM") and node.domain in ("", "ai.onnx") and index in (1, 2, 3)]
        if native_uses:
            if len(native_uses) != len(uses):
                raise ValueError(f"{name}: native packed parameter is also used in an incompatible ordinary role")
            if len(native_uses) != 1:
                raise ValueError(f"{name}: native packed initializer is shared across multiple parameter ports")
            continue  # Native packed rows are checked against each recurrent W/R/B.
        shape = tuple(initializers[name].dims)
        context = ", ".join(f"{node.name} ({node.op_type}).input[{index}]" for node, index in uses) or "no consumer"
        kind = record.get("enc_type")
        if kind == "PER_TENSOR":
            count = 1
        elif kind == "PER_CHANNEL":
            if len(shape) == 1:
                axes = {0}
            else:
                axes = set()
                for node, index in uses:
                    attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
                    axis = None
                    if node.domain in ("", "ai.onnx"):
                        if node.op_type == "Conv" and index == 1 and len(shape) >= 3:
                            axis = 0
                        elif node.op_type == "ConvTranspose" and index == 1 and len(shape) >= 3:
                            axis = 1
                        elif node.op_type == "Gemm" and index == 1 and len(shape) == 2:
                            axis = 0 if attrs.get("transB", 0) else 1
                        elif node.op_type == "MatMul" and index in (0, 1) and len(shape) >= 2:
                            axis = len(shape) - 1
                    if axis is None:
                        raise ValueError(f"{name}: unsupported PER_CHANNEL axis for initializer shape {shape} at "
                                         f"{node.name} ({node.op_type}).input[{index}]")
                    axes.add(axis)
                if not axes:
                    raise ValueError(f"{name}: unsupported PER_CHANNEL axis for initializer shape {shape} ({context})")
            if len(axes) != 1:
                raise ValueError(f"{name}: incompatible channel axes {sorted(axes)} for shared initializer ({context})")
            count = shape[next(iter(axes))]
        else:
            raise ValueError(f"{name}: unsupported ordinary parameter granularity {kind!r}; "
                             f"initializer shape {shape}, consumers: {context}")
        for field in ("scale", "n", "zero_point", "real_min", "real_max"):
            values = record.get(field)
            if (not isinstance(values, list) or len(values) != count
                    or any(isinstance(value, list) for value in values)):
                actual = len(values) if isinstance(values, list) else "scalar"
                raise ValueError(f"{name}.{field}: expected {count} {kind} channel values for initializer shape {shape} "
                                 f"at {context}, got {actual}")


def validate_compiler_encoding_graph(model, encodings, *, reference_graph=None,
                                     require_complete=False, scalar_inputs=()):
    """Reject declarations that do not describe the actual clean ONNX graph.

    ``reference_graph`` permits names eliminated by Torch simplification; real
    final nodes always take precedence. With that trusted export provenance,
    container names require real descendants; their virtual ports must also be
    checked against the live exporter's independently collected quantizers.
    Standalone graph validation rejects unverifiable virtual container ports.
    ``require_complete`` additionally checks generic compute input coverage.
    Numeric record validation remains the responsibility of the schema validator.
    """
    del scalar_inputs  # Scalar and tensor slots still must name real floating ONNX ports.
    activations, parameters = encodings.get("activation_encodings", {}), encodings.get("param_encodings", {})
    if not isinstance(activations, dict) or not isinstance(parameters, dict):
        raise ValueError("activation_encodings and param_encodings must be objects")
    counts = Counter(node.name for node in model.graph.node if node.name)
    if any(count != 1 for count in counts.values()):
        raise ValueError("ONNX node names must be unique for compiler encodings")
    nodes = {node.name: node for node in model.graph.node}
    raw_nodes = {node.name: node for node in reference_graph.graph.node} if reference_graph is not None else {}
    initializers = {value.name: value for value in model.graph.initializer}
    raw_initializers = {value.name: value for value in reference_graph.graph.initializer} if reference_graph is not None else {}
    types = _tensor_types(model)
    raw_types = _tensor_types(reference_graph) if reference_graph is not None else {}
    control_ports, control_only = _control_roles(model)
    raw_control_ports, raw_control_only = _control_roles(reference_graph) if reference_graph is not None else (set(), set())
    excluded = set(encodings.get("excluded_layers", []))
    for name, record in parameters.items():
        initializer = initializers.get(name)
        if initializer is None:
            initializer = raw_initializers.get(name)
        if initializer is None:
            raise ValueError(f"param_encodings.{name}: no corresponding ONNX initializer")
        if not _floating(initializer.data_type):
            raise ValueError(f"param_encodings.{name}: encoding on non-floating initializer")
        if name in (control_only if name in initializers else raw_control_only):
            raise ValueError(f"param_encodings.{name}: encoding on Clip-only control initializer")
    _validate_parameter_channels(model, parameters, raw_initializers | initializers, reference_graph)
    for name, entry in activations.items():
        node = nodes.get(name)
        node_types = types
        if node is None:
            node = raw_nodes.get(name)
            node_types = raw_types
        if node is None:
            if any(child.startswith(name + ".") or child.startswith(name + "#") for child in nodes.keys() | raw_nodes.keys()):
                if entry.get("is_GRU") or entry.get("is_LSTM"):
                    raise ValueError(f"{name}: native encoding requires one actual ONNX recurrent node")
                if reference_graph is None:
                    raise ValueError(f"{name}: cannot independently verify virtual container ports without export provenance")
                continue
            raise ValueError(f"activation_encodings.{name}: no corresponding ONNX node")
        native = entry.get("is_GRU") is True or entry.get("is_LSTM") is True
        if native:
            if node.op_type not in ("GRU", "LSTM") or node.domain not in ("", "ai.onnx"):
                raise ValueError(f"{name}: native encoding does not match a standard ONNX recurrent node")
            _validate_native(node, entry, parameters, raw_initializers | initializers)
            continue
        if node.op_type in ("GRU", "LSTM"):
            raise ValueError(f"{name}: recurrent ONNX node requires native packed encodings")
        for kind in ("input", "output"):
            ports = entry.get(kind, {})
            if not isinstance(ports, dict):
                raise ValueError(f"{name}.{kind}: expected port dictionary")
            tensors = node.input if kind == "input" else node.output
            for key in ports:
                if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
                    raise ValueError(f"{name}.{kind}: invalid port {key!r}")
                index = int(key)
                if index >= len(tensors) or not tensors[index]:
                    raise ValueError(f"{name}.{kind}[{index}]: nonexistent or empty ONNX port")
                if kind == "input" and (name, index) in (control_ports if name in nodes else raw_control_ports):
                    raise ValueError(f"{name}.{kind}[{index}]: encoding on Clip control port")
                if not _floating(node_types.get(tensors[index])):
                    raise ValueError(f"{name}.{kind}[{index}]: encoding on non-floating or unknown ONNX tensor {tensors[index]}")
    consumers = defaultdict(list)
    for node in model.graph.node:
        for index, tensor in enumerate(node.input):
            if tensor:
                consumers[tensor].append((node, index))
    mixed_outputs = []
    for node in model.graph.node:
        if node.op_type not in ("GRU", "LSTM") or node.domain not in ("", "ai.onnx"):
            continue
        entry = activations.get(node.name)
        if entry is None:
            if node.name not in excluded:
                raise ValueError(f"{node.name}: native recurrent node has neither encodings nor an exclusion")
            if any(tensor in parameters for tensor in node.input[1:4] if tensor):
                raise ValueError(f"{node.name}: excluded native recurrent node has packed parameter encodings")
            continue
        if node.name in excluded:
            raise ValueError(f"{node.name}: excluded native recurrent node has quantization encodings")
        for output_index, tensor in enumerate(node.output):
            if not tensor or common_native_recurrent_output_grid(node, entry, output_index) is not None:
                continue
            grids = native_recurrent_output_grids(node, entry, output_index)
            mixed_outputs.append((tensor, node.name, output_index, grids))
    visited = set()
    while mixed_outputs:
        tensor, origin, output_index, grids = mixed_outputs.pop()
        if tensor in visited:
            continue
        visited.add(tensor)
        for consumer, input_index in consumers[tensor]:
            if consumer.op_type in ("Shape", "Size"):
                continue
            consumer_entry = activations.get(consumer.name, {})
            independent_input = _port(consumer_entry, "input", input_index)
            if is_grid_preserving_op(consumer.op_type) and input_index == 0 and independent_input is None:
                # Data movement preserves the compound directional ownership.
                # Only an explicit output record starts a new tensor-wide QDQ
                # boundary; it does not assert a scalar grid on this input.
                mixed_outputs.extend((name, origin, output_index, grids)
                    for index, name in enumerate(consumer.output)
                    if name and _floating(types.get(name)) and _port(consumer_entry, "output", index) is None)
                continue
            if not is_grid_preserving_op(consumer.op_type) and is_compute_op(consumer.op_type) and independent_input is not None:
                continue
            raise ValueError(f"{origin}.output[{output_index}] -> {consumer.name}.input[{input_index}]: "
                f"different native direction grids (forward={_grid(grids[0])}, reverse={_grid(grids[1])}); "
                "a scalar propagated input grid cannot describe both directions; use an explicit compute input quantizer")
    if require_complete:
        scalar_constants = {name for name, value in initializers.items() if not value.dims}
        for node in model.graph.node:
            if node.op_type == "Constant" and node.domain in ("", "ai.onnx"):
                attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
                if ("value_float" in attrs or "value_int" in attrs or
                        ("value" in attrs and not attrs["value"].dims)):
                    scalar_constants.update(node.output)
        for node in model.graph.node:
            if not is_compute_op(node.op_type) or node.op_type in ("GRU", "LSTM") or node.name in excluded:
                continue
            entry = activations.get(node.name, {})
            for index, tensor in enumerate(node.input):
                if (_floating(types.get(tensor)) and tensor not in parameters
                        and (node.name, index) not in control_ports
                        and tensor not in scalar_constants
                        and _port(entry, "input", index) is None):
                    raise ValueError(f"{node.name}.input[{index}]: missing compiler quantization grid")
