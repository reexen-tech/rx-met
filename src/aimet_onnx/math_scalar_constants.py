"""Recognize mathematical literal ports in complete standard GELU formulas."""

from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def scalar_constant_values(model: onnx.ModelProto) -> dict[str, float]:
    """Read unambiguous, non-overridable static FP32 scalar values."""
    tensors = defaultdict(list)
    for tensor in model.graph.initializer:
        tensors[tensor.name].append(tensor)
    for node in model.graph.node:
        if (node.op_type != 'Constant' or node.domain not in ('', 'ai.onnx')
                or len(node.output) != 1 or len(node.attribute) != 1):
            continue
        attribute = node.attribute[0]
        if attribute.name == 'value' and attribute.type == onnx.AttributeProto.TENSOR:
            tensors[node.output[0]].append(attribute.t)
    graph_inputs = {value.name for value in model.graph.input}
    result = {}
    for name, candidates in tensors.items():
        if name in graph_inputs or len(candidates) != 1:
            continue
        tensor = candidates[0]
        if tensor.data_type != TensorProto.FLOAT or tensor.dims:
            continue
        try:
            value = float(numpy_helper.to_array(tensor))
        except (ValueError, OSError):
            continue
        if math.isfinite(value):
            result[name] = value
    return result


def identify_math_scalar_constants(model: onnx.ModelProto) -> dict[tuple[str, int], str]:
    """Map literal ports in complete FP32 erf/tanh GELU expansions to tensors.

    Only the canonical formulas and their two common final multiplication
    associations are recognized. Ports, rather than whole tensors, are exempt:
    the same constant may still require quantization at another consumer.
    Frozen trainable coefficients identical to these formulas cannot be
    distinguished from literals without additional source provenance.
    """
    literals = scalar_constant_values(model)
    names = Counter(node.name for node in model.graph.node)
    named_nodes = {node.name: node for node in model.graph.node}
    producers = defaultdict(list)
    consumers = defaultdict(set)
    for node in model.graph.node:
        for output in node.output:
            producers[output].append(node)
        for value in node.input:
            consumers[value].add(node.name)
    graph_outputs = {value.name for value in model.graph.output}

    def op(tensor, kind, inputs):
        matches = producers.get(tensor, ())
        if len(matches) != 1:
            return None
        node = matches[0]
        if (node.op_type != kind or node.domain not in ('', 'ai.onnx')
                or not node.name or names[node.name] != 1 or node.attribute
                or len(node.input) != inputs or len(node.output) != 1):
            return None
        return node

    def literal(tensor, value):
        return tensor in literals and literals[tensor] == float(np.float32(value))

    def with_literal(tensor, kind, value):
        node = op(tensor, kind, 2)
        if node is not None:
            for index in (0, 1):
                if literal(node.input[index], value):
                    return (node.input[1 - index], {(node.name, index): node.input[index]}, {node.name})
        return None

    def activation(tensor):
        erf = op(tensor, 'Erf', 1)
        if erf is not None:
            divide = op(erf.input[0], 'Div', 2)
            if divide is not None and literal(divide.input[1], math.sqrt(2.)):
                return (divide.input[0], {(divide.name, 1): divide.input[1]}, {divide.name, erf.name})
        tanh = op(tensor, 'Tanh', 1)
        if tanh is None:
            return None
        normal = with_literal(tanh.input[0], 'Mul', math.sqrt(2. / math.pi))
        inner = op(normal[0], 'Add', 2) if normal is not None else None
        if inner is None:
            return None
        for index in (0, 1):
            cubic_scale = with_literal(inner.input[index], 'Mul', .044715)
            cube = op(cubic_scale[0], 'Pow', 2) if cubic_scale is not None else None
            if (cube is not None and cube.input[0] == inner.input[1 - index]
                    and literal(cube.input[1], 3.)):
                ports = {**normal[1], **cubic_scale[1], (cube.name, 1): cube.input[1]}
                matched = normal[2] | cubic_scale[2] | {cube.name, inner.name, tanh.name}
                return cube.input[0], ports, matched
        return None

    def complete(root, source, shifted, half_ports, half_nodes):
        if shifted is None:
            return None
        function = activation(shifted[0])
        if function is None or function[0] != source:
            return None
        matched = half_nodes | shifted[2] | function[2] | {root.name}
        for name in matched - {root.name}:
            node = named_nodes[name]
            if any(output in graph_outputs or consumers[output] - matched for output in node.output):
                return None
        return {**half_ports, **shifted[1], **function[1]}

    protected = {}
    for node in model.graph.node:
        if not node.output or op(node.output[0], 'Mul', 2) is None:
            continue
        for index in (0, 1):
            half = with_literal(node.input[index], 'Mul', .5)
            shifted = with_literal(node.input[1 - index], 'Add', 1.)
            if half is not None:
                ports = complete(node, half[0], shifted, half[1], half[2])
                if ports is not None:
                    protected.update(ports)
        half = with_literal(node.output[0], 'Mul', .5)
        product = op(half[0], 'Mul', 2) if half is not None else None
        if product is not None:
            for index in (0, 1):
                shifted = with_literal(product.input[index], 'Add', 1.)
                ports = complete(node, product.input[1 - index], shifted,
                                 half[1], half[2] | {product.name})
                if ports is not None:
                    protected.update(ports)
    return protected
