"""Preserve source parameter names after ONNX folds their shape-only views."""
from __future__ import annotations

from collections import defaultdict
import copy
import inspect

import numpy as np
import onnx
import torch


# These operations preserve element order. Do not infer provenance through
# arithmetic or transpose: that would require transforming the encodings too.
_SHAPE_METHODS = {'view', 'reshape', 'flatten', 'squeeze', 'unsqueeze', 'contiguous', 'detach'}
_SHAPE_FUNCTIONS = {torch.reshape, torch.flatten, torch.squeeze, torch.unsqueeze}


def _parameter_inputs(model):
    """Read parameter-to-consumer edges from the prepared FX graph."""
    graph = getattr(model, 'graph', None)
    if not isinstance(graph, torch.fx.Graph):
        return {}
    parameters = dict(model.named_parameters())
    bindings = defaultdict(list)

    def source(value):
        while isinstance(value, torch.fx.Node):
            if value.op == 'get_attr':
                return value.target if value.target in parameters else None
            if ((value.op == 'call_method' and value.target in _SHAPE_METHODS)
                    or (value.op == 'call_function' and value.target in _SHAPE_FUNCTIONS)):
                value = value.args[0]
            else:
                return None
        return None

    for node in graph.nodes:
        if node.op != 'call_module':
            continue
        signature = inspect.signature(model.get_submodule(node.target).forward)
        bound = signature.bind(*node.args, **node.kwargs)
        bound.apply_defaults()
        values = []
        for key, value in bound.arguments.items():
            if signature.parameters[key].kind == inspect.Parameter.VAR_POSITIONAL:
                values.extend(value)
            elif signature.parameters[key].kind != inspect.Parameter.VAR_KEYWORD:
                values.append(value)
        for index, value in enumerate(values):
            name = source(value)
            if name is not None:
                bindings[name].append((node.target, index))
    return bindings


def restore_parameter_names(onnx_path: str, model, expected: dict) -> None:
    """Give folded initializers their encoding keys using proven FX input edges.

    A shared constant is copied per source parameter, so equal values with
    different quantization grids do not accidentally acquire a shared name.
    Unsupported or ambiguous mappings remain errors in artifact validation.
    """
    graph = onnx.load(onnx_path)
    initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    nodes = {node.name: node for node in graph.graph.node}
    occupied = {name for node in graph.graph.node for name in node.output}
    occupied.update(value.name for value in graph.graph.input)
    bindings = _parameter_inputs(model)
    changed = False
    for parameter in expected.get('param_encodings', {}):
        if parameter in initializers:
            continue
        consumers = []
        for name, index in bindings.get(parameter, []):
            node = nodes.get(name)
            if node is not None and index < len(node.input) and node.input[index] in initializers:
                consumers.append((node, index, initializers[node.input[index]]))
        if not consumers:
            continue
        candidate = consumers[0][2]
        # Shape-only paths to the same parameter must agree on the exported
        # representation before we can assign one public name to them.
        if any(tensor.data_type != candidate.data_type
               or list(tensor.dims) != list(candidate.dims)
               or not np.array_equal(onnx.numpy_helper.to_array(tensor),
                                     onnx.numpy_helper.to_array(candidate))
               for _, _, tensor in consumers[1:]):
            raise RuntimeError(f'Parameter {parameter}: incompatible ONNX views for one encoding key')
        if parameter in occupied:
            raise RuntimeError(f'Parameter {parameter}: ONNX tensor name is already occupied')
        restored = copy.deepcopy(candidate)
        restored.name = parameter
        graph.graph.initializer.append(restored)
        initializers[parameter] = restored
        for node, index, _ in consumers:
            node.input[index] = parameter
        changed = True

    if changed:
        used = {name for node in graph.graph.node for name in node.input}
        used.update(value.name for value in (*graph.graph.input, *graph.graph.output))
        retained = [tensor for tensor in graph.graph.initializer if tensor.name in used]
        del graph.graph.initializer[:]
        graph.graph.initializer.extend(retained)
        onnx.save(graph, onnx_path)
