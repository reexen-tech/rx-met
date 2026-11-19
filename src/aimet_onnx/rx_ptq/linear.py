"""Normalize static ONNX Linear expressions before creating quantizers."""

from __future__ import annotations

import copy
import json
import math

import onnx
from onnx import TensorProto, helper, numpy_helper


_METADATA_KEY = 'rx_met.linear_normalization'


def linear_groups(model: onnx.ModelProto) -> dict[str, dict]:
    """Return persisted aliases for Linear expressions replaced by Gemm."""
    entries = [entry for entry in model.metadata_props if entry.key == _METADATA_KEY]
    if not entries:
        return {}
    if len(entries) != 1:
        raise ValueError('Duplicate Linear normalization metadata')
    payload = json.loads(entries[0].value)
    if (not isinstance(payload, dict) or payload.get('schema_version') != 1
            or not isinstance(payload.get('groups'), list)):
        raise ValueError('Unsupported Linear normalization metadata')
    groups = {}
    for group in payload['groups']:
        _validate_group(model, group)
        name = group['name']
        if name in groups:
            raise ValueError(f'Duplicate normalized Linear name: {name}')
        groups[name] = group
    return groups


def _validate_group(model, group):
    fields = ('name', 'matmul', 'add', 'input', 'weight', 'bias', 'intermediate',
              'output', 'gemm_input', 'gemm_output')
    if (not isinstance(group, dict)
            or any(not isinstance(group.get(field), str) or not group[field] for field in fields)):
        raise ValueError('Incomplete Linear normalization metadata')
    for field in ('input_shape', 'output_shape'):
        shape = group.get(field)
        if (not isinstance(shape, list) or len(shape) < 2
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError(f'Invalid Linear normalization {field}')
    if (group['name'] != group['matmul']
            or group['input_shape'][:-1] != group['output_shape'][:-1]):
        raise ValueError('Inconsistent Linear normalization metadata')
    nodes = {name: [node for node in model.graph.node if node.name == name]
             for name in (group['name'], group.get('input_reshape'), group.get('output_reshape'))
             if isinstance(name, str)}
    gemms = nodes[group['name']]
    if (len(gemms) != 1 or gemms[0].op_type != 'Gemm' or gemms[0].domain not in ('', 'ai.onnx')
            or list(gemms[0].input) != [group['gemm_input'], group['weight'], group['bias']]
            or list(gemms[0].output) != [group['gemm_output']]):
        raise ValueError(f'Linear normalization metadata does not match Gemm {group["name"]}')
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in gemms[0].attribute}
    expected = {'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 0}
    if (len(attributes) != len(gemms[0].attribute)
            or any(name not in expected or value != expected[name] for name, value in attributes.items())):
        raise ValueError(f'Linear normalization requires an unscaled, untransposed Gemm: {group["name"]}')
    _static_initializer(model, group['weight'], TensorProto.FLOAT,
                        [group['input_shape'][-1], group['output_shape'][-1]])
    _static_initializer(model, group['bias'], TensorProto.FLOAT, [group['output_shape'][-1]])
    generated = group.get('generated_nodes')
    if len(group['input_shape']) == 2:
        if (generated != [] or group.get('input_reshape') is not None
                or group.get('output_reshape') is not None
                or group['gemm_input'] != group['input'] or group['gemm_output'] != group['output']):
            raise ValueError('Invalid two-dimensional Linear normalization metadata')
    else:
        names = [group.get('input_reshape'), group.get('output_reshape')]
        if any(not isinstance(name, str) or not name for name in names) or generated != names:
            raise ValueError('Invalid Linear normalization Reshape metadata')
        for name, source, target in ((names[0], group['input'], group['gemm_input']),
                                     (names[1], group['gemm_output'], group['output'])):
            matches = nodes[name]
            if (len(matches) != 1 or matches[0].op_type != 'Reshape'
                    or matches[0].domain not in ('', 'ai.onnx')
                    or len(matches[0].input) != 2 or matches[0].input[0] != source
                    or list(matches[0].output) != [target]):
                raise ValueError(f'Linear normalization metadata does not match Reshape {name}')
            expected_shape = ([math.prod(group['input_shape'][:-1]), group['input_shape'][-1]]
                              if name == names[0] else group['output_shape'])
            tensor = _static_initializer(model, matches[0].input[1], TensorProto.INT64, [len(expected_shape)])
            try:
                target_shape = numpy_helper.to_array(tensor).tolist()
            except (ValueError, OSError) as error:
                raise ValueError(f'Unreadable Linear normalization Reshape target: {name}') from error
            if target_shape != expected_shape:
                raise ValueError(f'Linear normalization Reshape target changed: {name}')


def _static_initializer(model, name, dtype, shape):
    tensors = [tensor for tensor in model.graph.initializer if tensor.name == name]
    if (len(tensors) != 1 or any(value.name == name for value in model.graph.input)
            or tensors[0].data_type != dtype or list(tensors[0].dims) != shape):
        raise ValueError(f'Linear normalization requires a static initializer with the original type/shape: {name}')
    return tensors[0]


def _constants(model):
    constants = {tensor.name: tensor for tensor in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Constant' and node.domain in ('', 'ai.onnx') and len(node.output) == 1:
            for attribute in node.attribute:
                if attribute.name == 'value' and attribute.type == onnx.AttributeProto.TENSOR:
                    constants[node.output[0]] = attribute.t
    return constants


def _tensor_info(model):
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except (onnx.shape_inference.InferenceError, ValueError):
        inferred = model
    result = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        tensor = value.type.tensor_type
        shape = None
        if tensor.HasField('shape'):
            dims = tensor.shape.dim
            if all(dim.HasField('dim_value') and dim.dim_value > 0 for dim in dims):
                shape = [dim.dim_value for dim in dims]
        result[value.name] = (tensor.elem_type, shape)
    return result


def _candidates(model, constants):
    producers = {output: node for node in model.graph.node for output in node.output}
    for add in model.graph.node:
        if add.op_type != 'Add' or len(add.input) != 2:
            continue
        for index, name in enumerate(add.input):
            matmul = producers.get(name)
            bias = add.input[1 - index]
            if matmul is not None and matmul.op_type == 'MatMul' and bias in constants:
                yield matmul, add, bias


def _unsupported_reason(model, matmul, add, bias, constants, infos):
    if matmul.domain not in ('', 'ai.onnx') or add.domain not in ('', 'ai.onnx'):
        return 'custom operator domain'
    opsets = {entry.domain: entry.version for entry in model.opset_import}
    if opsets.get(matmul.domain, 0) < 7 or opsets.get(add.domain, 0) < 7:
        return 'legacy operator broadcast semantics'
    if matmul.attribute or add.attribute:
        return 'nonstandard MatMul/Add attributes'
    if not matmul.name or not add.name:
        return 'unnamed operator'
    if len(matmul.input) != 2 or len(matmul.output) != 1 or len(add.output) != 1:
        return 'unsupported operator arity'
    names = [node.name for node in model.graph.node]
    if names.count(matmul.name) != 1 or names.count(add.name) != 1:
        return 'duplicate operator name'
    intermediate = matmul.output[0]
    if any(value.name == intermediate for value in model.graph.output):
        return 'MatMul output is a graph output'
    consumers = [node for node in model.graph.node if intermediate in node.input]
    if len(consumers) != 1:
        return 'MatMul output has multiple consumers'
    graph_inputs = {value.name for value in model.graph.input}
    initializers = {value.name for value in model.graph.initializer}
    weight = constants.get(matmul.input[1])
    if weight is None or matmul.input[1] in graph_inputs or bias in graph_inputs:
        return 'dynamic or overridable weight/bias'
    if matmul.input[1] not in initializers or bias not in initializers:
        return 'weight/bias must be initializers'
    bias_tensor = constants[bias]
    if weight.data_type != TensorProto.FLOAT or bias_tensor.data_type != TensorProto.FLOAT:
        return 'weight/bias must be FP32'
    if len(weight.dims) != 2 or len(bias_tensor.dims) != 1:
        return 'expected weight[K,N] and bias[N]'
    dtype, shape = infos.get(matmul.input[0], (None, None))
    if dtype != TensorProto.FLOAT or shape is None or len(shape) < 2:
        return 'expected a static FP32 input with rank >= 2'
    if any(dimension <= 0 for dimension in (*weight.dims, *bias_tensor.dims)):
        return 'zero-sized weight/bias'
    if shape[-1] != weight.dims[0] or weight.dims[1] != bias_tensor.dims[0]:
        return 'incompatible Linear dimensions'
    return None


def _unique_name(base, used):
    name, suffix = base, 1
    while name in used:
        name = f'{base}_{suffix}'
        suffix += 1
    used.add(name)
    return name


def normalize_linear_graph(model: onnx.ModelProto) -> onnx.ModelProto:
    """Copy the graph and replace safe MatMul + constant bias Add with Gemm."""
    result = copy.deepcopy(model)
    groups = linear_groups(result)
    constants = _constants(result)
    infos = _tensor_info(result)
    replacements, removed = {}, set()
    used = {node.name for node in result.graph.node}
    used.update(name for node in result.graph.node for name in (*node.input, *node.output))
    used.update(value.name for value in (*result.graph.input, *result.graph.output,
                                       *result.graph.value_info, *result.graph.initializer))
    for matmul, add, bias in _candidates(result, constants):
        if _unsupported_reason(result, matmul, add, bias, constants, infos):
            continue
        name = matmul.name
        output = add.output[0]
        input_shape = infos[matmul.input[0]][1]
        output_shape = [*input_shape[:-1], constants[bias].dims[0]]
        gemm_input, gemm_output = matmul.input[0], output
        input_reshape = output_reshape = None
        before, after = [], []
        if len(input_shape) > 2:
            gemm_input = _unique_name(name + '.linear_input', used)
            gemm_output = _unique_name(name + '.linear_output', used)
            input_reshape = _unique_name(name + '.linear_input_reshape', used)
            output_reshape = _unique_name(name + '.linear_output_reshape', used)
            input_target = _unique_name(name + '.linear_input_shape', used)
            output_target = _unique_name(name + '.linear_output_shape', used)
            flat_shape = [math.prod(input_shape[:-1]), input_shape[-1]]
            result.graph.initializer.extend([
                helper.make_tensor(input_target, TensorProto.INT64, [2], flat_shape),
                helper.make_tensor(output_target, TensorProto.INT64, [len(output_shape)], output_shape),
            ])
            before.append(helper.make_node('Reshape', [matmul.input[0], input_target],
                                           [gemm_input], name=input_reshape))
            after.append(helper.make_node('Reshape', [gemm_output, output_target],
                                          [output], name=output_reshape))
            result.graph.value_info.extend([
                helper.make_tensor_value_info(gemm_input, TensorProto.FLOAT, flat_shape),
                helper.make_tensor_value_info(gemm_output, TensorProto.FLOAT,
                                              [flat_shape[0], output_shape[-1]]),
            ])
        replacements[name] = [*before, helper.make_node(
            'Gemm', [gemm_input, matmul.input[1], bias], [gemm_output],
            name=name, alpha=1., beta=1., transB=0), *after]
        removed.add(add.name)
        groups[name] = {
            'name': name, 'matmul': name, 'add': add.name,
            'input': matmul.input[0], 'weight': matmul.input[1], 'bias': bias,
            'intermediate': matmul.output[0], 'output': output,
            'gemm_input': gemm_input, 'gemm_output': gemm_output,
            'input_shape': input_shape, 'output_shape': output_shape,
            'generated_nodes': [node.name for node in (*before, *after)],
            'input_reshape': input_reshape, 'output_reshape': output_reshape,
        }
    if not replacements:
        return result
    nodes = [replacement for node in result.graph.node if node.name not in removed
             for replacement in replacements.get(node.name, [node])]
    result.graph.ClearField('node')
    result.graph.node.extend(nodes)
    intermediate_names = {groups[name]['intermediate'] for name in replacements}
    values = [value for value in result.graph.value_info if value.name not in intermediate_names]
    result.graph.ClearField('value_info')
    result.graph.value_info.extend(values)
    entry = next((entry for entry in result.metadata_props if entry.key == _METADATA_KEY), None)
    if entry is None:
        entry = result.metadata_props.add(key=_METADATA_KEY)
    entry.value = json.dumps({'schema_version': 1, 'groups': list(groups.values())}, sort_keys=True)
    return result


def validate_legacy_linear_boundaries(model: onnx.ModelProto) -> None:
    """Reject unfused Linear targets whose split boundaries cannot be compiled."""
    constants = _constants(model)
    infos = _tensor_info(model)
    for matmul, add, bias in _candidates(model, constants):
        reason = _unsupported_reason(model, matmul, add, bias, constants, infos)
        raise ValueError(f'Unsupported MatMul + Add Linear boundary {matmul.name} -> '
                         f'{add.name}: {reason or "normalize the graph before calibration"}')
