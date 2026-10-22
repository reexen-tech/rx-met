"""Infer and persist ONNX tensor metadata without rewriting computations."""

import copy
from pathlib import Path

import onnx


def _graphs(graph, scope='graph'):
    yield scope, graph
    for index, node in enumerate(graph.node):
        for attribute in node.attribute:
            prefix = f'{scope}/{node.name or index}.{attribute.name}'
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g, prefix)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for number, child in enumerate(attribute.graphs):
                    yield from _graphs(child, f'{prefix}[{number}]')


def _inspect_shapes(model):
    errors = []
    count = symbolic = intermediate = 0
    for scope, graph in _graphs(model.graph):
        endpoints = {value.name: value for value in (*graph.input, *graph.output)}
        infos = {value.name: value for value in graph.value_info}
        infos.update(endpoints)
        initializers = {value.name for value in graph.initializer}
        produced = {name for node in graph.node for name in node.output if name}
        required = (produced | endpoints.keys()) - initializers
        intermediate += len(produced - endpoints.keys() - initializers)
        for name in sorted(required):
            count += 1
            location = f'{scope}:{name}'
            value = infos.get(name)
            if value is None or not value.type.HasField('tensor_type'):
                errors.append(f'{location}: missing tensor shape/dtype metadata')
                continue
            tensor = value.type.tensor_type
            if tensor.elem_type == onnx.TensorProto.UNDEFINED:
                errors.append(f'{location}: missing tensor dtype')
            if not tensor.HasField('shape'):
                errors.append(f'{location}: missing tensor shape')
                continue
            for index, dimension in enumerate(tensor.shape.dim):
                if dimension.HasField('dim_value') and dimension.dim_value >= 0:
                    continue
                if dimension.HasField('dim_param') and dimension.dim_param.strip():
                    symbolic += 1
                else:
                    errors.append(f'{location}: missing or invalid shape dimension {index}')
    return errors, {'tensor_count': count, 'intermediate_tensors': intermediate,
                    'symbolic_dimensions': symbolic}


def validate_onnx_shapes(model):
    """Check saved annotations as-is, including scalars and nested graph outputs."""
    errors, report = _inspect_shapes(model)
    if errors:
        details = '\n'.join(f'  - {error}' for error in errors[:20])
        raise RuntimeError(f'ONNX shape metadata validation failed:\n{details}')
    return report


def _copy_annotations(target, inferred):
    allowed = {value.name for value in (*target.input, *target.output, *target.initializer)}
    allowed.update(name for node in target.node for name in (*node.input, *node.output) if name)
    endpoints = {value.name for value in (*target.input, *target.output)}
    infos = {value.name: value for value in target.value_info}
    infos.update({value.name: value for value in inferred.value_info
                  if value.name in allowed and value.name not in endpoints})
    values = [copy.deepcopy(value) for value in infos.values()]
    target.ClearField('value_info')
    target.value_info.extend(values)
    inferred_endpoints = {value.name: value for value in (*inferred.input, *inferred.output)}
    for value in (*target.input, *target.output):
        if value.name in inferred_endpoints:
            value.type.CopyFrom(inferred_endpoints[value.name].type)
    inferred_nodes = {tuple(node.output): node for node in inferred.node}
    for node in target.node:
        other = inferred_nodes.get(tuple(node.output))
        if other is None:
            continue
        attributes = {attribute.name: attribute for attribute in other.attribute}
        for attribute in node.attribute:
            source = attributes.get(attribute.name)
            if source is None or source.type != attribute.type:
                continue
            if attribute.type == onnx.AttributeProto.GRAPH:
                _copy_annotations(attribute.g, source.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child, inferred_child in zip(attribute.graphs, source.graphs):
                    _copy_annotations(child, inferred_child)


def complete_onnx_shapes(model):
    """Use existing inference engines; copy back only tensor type annotations."""
    result = copy.deepcopy(model)
    missing = []
    try:
        inferred = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=True
        )
        missing, _ = _inspect_shapes(inferred)
        if missing:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
            inferred = SymbolicShapeInference.infer_shapes(
                inferred, auto_merge=True, guess_output_rank=False, verbose=0
            )
            if inferred is None:
                raise RuntimeError('Symbolic shape inference does not support this model')
        _copy_annotations(result.graph, inferred.graph)
        validate_onnx_shapes(result)
    except Exception as exc:
        details = '; '.join(missing[:5])
        raise RuntimeError(f'Unable to complete ONNX shape metadata: {details}; {exc}') from exc
    return result


def save_onnx_with_shapes(path):
    """Preserve existing tensor storage and external-data references when saving."""
    path = Path(path)
    stored = onnx.load(str(path), load_external_data=False)
    annotated = complete_onnx_shapes(onnx.load(str(path)))
    _copy_annotations(stored.graph, annotated.graph)
    validate_onnx_shapes(stored)
    onnx.save_model(stored, str(path))
    onnx.checker.check_model(str(path), full_check=True)
    return validate_onnx_shapes(onnx.load(str(path), load_external_data=False))
