"""Keep eval Conv-BatchNorm quantization in the original parameter domains."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def _constant(name, initializers, producers, graph_inputs, visited=()):
    if name in graph_inputs or name in visited:
        return None
    if name in initializers:
        tensor = initializers[name]
    else:
        node = producers.get(name)
        if node is None or node.domain not in ('', 'ai.onnx'):
            return None
        if node.op_type == 'Identity' and len(node.input) == 1:
            value = _constant(node.input[0], initializers, producers, graph_inputs, (*visited, name))
            return None if value is None else (value[0], (*value[1], name))
        if node.op_type != 'Constant':
            return None
        tensor = next((attribute.t for attribute in node.attribute
                       if attribute.name == 'value' and attribute.type == attribute.TENSOR), None)
        if tensor is None:
            return None
    if tensor.data_type != TensorProto.FLOAT:
        return None
    try:
        value = numpy_helper.to_array(tensor)
    except (ValueError, OSError):
        return None
    return value, (name,)


def conv_bn_groups(model):
    """Identify exclusive constant eval Conv -> BN pairs without rewriting them."""
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output if name}
    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    names = [node.name for node in model.graph.node]
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    groups = {}
    for bn in model.graph.node:
        if (bn.op_type != 'BatchNormalization' or bn.domain not in ('', 'ai.onnx')
                or len(bn.input) != 5 or len([name for name in bn.output if name]) != 1
                or not bn.output or not bn.output[0]
                or not bn.name or names.count(bn.name) != 1):
            continue
        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in bn.attribute}
        epsilon = attributes.get('epsilon', 1e-5)
        if attributes.get('training_mode', 0) != 0 or not math.isfinite(epsilon) or epsilon <= 0:
            continue
        conv = producers.get(bn.input[0])
        if (conv is None or conv.op_type != 'Conv' or conv.domain not in ('', 'ai.onnx')
                or not conv.name or names.count(conv.name) != 1 or len(conv.output) != 1
                or not 2 <= len(conv.input) <= 3 or conv.output[0] in graph_outputs
                or len(consumers.get(conv.output[0], ())) != 1):
            continue
        weight = _constant(conv.input[1], initializers, producers, graph_inputs)
        if (weight is None or not 3 <= weight[0].ndim <= 5
                or any(size <= 0 for size in weight[0].shape)):
            continue
        channels = weight[0].shape[0]
        bias_name = conv.input[2] if len(conv.input) == 3 and conv.input[2] else None
        conv_parameters = list(weight[1])
        if bias_name:
            bias = _constant(bias_name, initializers, producers, graph_inputs)
            if bias is None or bias[0].shape != (channels,):
                continue
            conv_parameters.extend(bias[1])
        parameters = [_constant(name, initializers, producers, graph_inputs) for name in bn.input[1:]]
        if any(value is None or value[0].shape != (channels,)
               or not np.isfinite(value[0]).all() for value in parameters):
            continue
        if (parameters[3][0] < 0).any():
            continue
        groups[conv.name] = {
            'name': conv.name, 'conv': conv.name, 'bn': bn.name,
            'input': conv.input[0], 'conv_output': conv.output[0], 'bn_output': bn.output[0],
            'conv_weight': conv.input[1], 'conv_bias': bias_name,
            'conv_param_tensors': sorted(set(conv_parameters)),
            'bn_params': list(bn.input[1:]),
            'bn_param_tensors': sorted({name for value in parameters for name in value[1]}),
            'channels': channels,
            'sourceflags': {'training_mode': 0, 'constant_stats': True, 'exclusive_output': True},
        }
    return groups


def _bool(value):
    return value is True or (isinstance(value, str) and value.lower() == 'true')


def conv_bn_output_intents(model, config_file=None):
    """Recover Torch's configured Conv output ownership, before BN consumes it."""
    groups = conv_bn_groups(model)
    if not groups:
        return {}
    from aimet_common.quantsim_config.quantsim_config import _get_config_file
    config = json.loads(Path(_get_config_file(config_file)).read_text(encoding='utf-8'))
    enabled = config.get('defaults', {}).get('ops', {}).get('is_output_quantized', False)
    enabled = config.get('op_type', {}).get('Conv', {}).get('is_output_quantized', enabled)
    intents = {name: _bool(enabled) for name in groups}
    nodes = {node.name: node for node in model.graph.node}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, set()).add(node.name)
    ignored = {group['bn'] for group in groups.values()}
    weights = {tensor.name: tensor for tensor in model.graph.initializer}

    def match(name, pattern):
        node = nodes[name]
        children = set().union(*(consumers.get(tensor, set()) for tensor in node.output))
        if name in ignored:
            return [path for child in children for path in match(child, pattern)]
        if not pattern or node.op_type != pattern[0]:
            return []
        if node.op_type == 'Conv' and len(node.input) > 1 and node.input[1] in weights:
            shape = weights[node.input[1]].dims
            group = next((attribute.i for attribute in node.attribute if attribute.name == 'group'), 1)
            if len(shape) == 5 or (group > 1 and len(shape) > 1 and shape[1] == 1):
                return []
        if len(pattern) == 1:
            return [[name]]
        if len(children) != 1:
            return []
        return [[name, *path] for path in match(next(iter(children)), pattern[1:])]

    for supergroup in config.get('supergroups', ()):
        pattern = supergroup.get('op_list', ())
        for name in nodes:
            for path in match(name, pattern):
                for producer in path[:-1]:
                    if producer in intents:
                        intents[producer] = False
    graph_outputs = {value.name for value in model.graph.output}
    output_flag = config.get('model_output', {}).get('is_output_quantized')
    if output_flag is not None:
        for name, group in groups.items():
            if group['bn_output'] in graph_outputs:
                intents[name] = _bool(output_flag)
    return intents


def _validate_parameter_uses(model, groups):
    tensors = {name for group in groups.values() for name in group['bn_param_tensors']}
    ports = {(group['bn'], index) for group in groups.values() for index in range(1, 5)}
    exposed = tensors & {value.name for value in model.graph.output}
    if exposed:
        raise ValueError('Conv-BN parameters also exposed as model outputs: ' + ', '.join(sorted(exposed)))
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name not in tensors or (node.name, index) in ports:
                continue
            if (node.op_type == 'Identity' and node.domain in ('', 'ai.onnx')
                    and all(output in tensors for output in node.output)):
                continue
            raise ValueError(f'Conv-BN parameter {name} is shared with an incompatible use: '
                             f'{node.name}.input[{index}]')


def apply_conv_bn_boundaries(sim, model, *, config_file=None):
    """Apply Conv-BN ownership once, then retain parameter and intermediate exclusions.

    Return whether a new runtime quantizer was inserted and needs session rebuild.
    Original convolution parameters remain in their own QDQ domains.
    """
    groups = conv_bn_groups(model)
    _validate_parameter_uses(model, groups)
    initialized = set(getattr(sim, '_rx_conv_bn_initialized', ()))
    intents = conv_bn_output_intents(model, config_file) if set(groups) - initialized else {}
    added = False
    for name, group in groups.items():
        if name not in initialized:
            output = sim.qc_quantize_op_dict.get(group['bn_output'])
            if output is None and intents[name]:
                sim._insert_quantizer(group['bn_output'], is_param=False)
                if group['bn_output'] not in sim.activation_names:
                    sim.activation_names.append(group['bn_output'])
                output = sim.qc_quantize_op_dict[group['bn_output']]
                added = True
            if output is not None:
                output.enabled = intents[name]
        names = [group['conv_output'], *group['bn_param_tensors']]
        alias = getattr(sim, '_rx_input_quantizers', {}).get((group['bn'], 0))
        if alias is not None:
            names.append(alias)
        for tensor in names:
            quantizer = sim.qc_quantize_op_dict.get(tensor)
            if quantizer is not None:
                quantizer.enabled = False
        initialized.add(name)
    sim._rx_conv_bn_groups = groups
    sim._rx_conv_bn_initialized = initialized
    return added
