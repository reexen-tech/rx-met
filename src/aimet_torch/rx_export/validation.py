"""Validate the final export against encodings collected independently from QuantSim.

Validate both QuantSim coverage and the final ONNX compute operands. Module
encodings alone do not prove that decomposed ONNX arithmetic is quantized.
Never invent intermediate grids to make an incomplete artifact pass.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
import inspect
import json
import math

import onnx
import torch

from aimet_common.quantsim_config.compute_ops import is_compute_op
from aimet_common.compiler_encodings import validate_compiler_encodings
from aimet_torch.native_recurrent import named_native_recurrent
from aimet_torch.v2.utils import flatten_nn_module_list
from .postprocess.encodings_ops import postprocess_encodings_inplace


def _fail(errors: list[str]) -> None:
    if errors:
        details = '\n'.join(f'  - {error}' for error in errors[:20])
        if len(errors) > 20:
            details += f'\n  ... {len(errors) - 20} more errors'
        raise RuntimeError(f'Export artifact validation failed:\n{details}')


@contextmanager
def observe_non_tensor_inputs(model):
    """Find input slots that never receive floating tensors in the export run.

    Inspect the unquantized copy so missing calibration cannot prevent this
    observation. Scalars such as Pow's exponent are not quantized by AIMET.
    An unobserved slot is never exempted from the calibration requirement.
    """
    floating_slots = set()
    scalar_slots = set()

    def has_float_tensor(value):
        if isinstance(value, torch.Tensor):
            return value.is_floating_point()
        if isinstance(value, (tuple, list)):
            return any(has_float_tensor(item) for item in value)
        if isinstance(value, dict):
            return any(has_float_tensor(item) for item in value.values())
        return False

    with ExitStack() as stack:
        for name, module in model.named_modules():
            signature = inspect.signature(module.forward)
            def observe(module, args, kwargs, name=name, signature=signature):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = []
                for param, value in bound.arguments.items():
                    if signature.parameters[param].kind == inspect.Parameter.VAR_POSITIONAL:
                        values.extend(value)
                    elif signature.parameters[param].kind != inspect.Parameter.VAR_KEYWORD:
                        values.append(value)
                for index, value in enumerate(values):
                    key = (name, index)
                    if has_float_tensor(value):
                        floating_slots.add(key)
                        scalar_slots.discard(key)
                    elif key not in floating_slots:
                        scalar_slots.add(key)
            handle = module.register_forward_pre_hook(observe, with_kwargs=True)
            stack.callback(handle.remove)
        yield scalar_slots


def collect_expected_encodings(model, non_tensor_inputs=()) -> dict:
    """Snapshot every enabled quantizer independently of exporter-generated files.

    A None/disabled quantizer needs no encoding. An enabled but uninitialized
    quantizer is an error, including one never exercised during calibration.
    Native recurrent operators own their internal grids and packing protocol.
    """
    expected = {'activation_encodings': {}, 'param_encodings': {}}
    errors = []
    native = list(named_native_recurrent(model))
    native_names = {name for name, _, _ in native}

    def read_quantizer(quantizer, path):
        if quantizer is None or not getattr(quantizer, 'enabled', True):
            return None
        if not quantizer.is_initialized():
            errors.append(f'{path}: enabled quantizer is not calibrated')
            return None
        encoding = quantizer.get_encodings()
        if encoding is None:
            errors.append(f'{path}: missing quantization parameters')
            return None
        entries = encoding.to_qnn_encoding_dict('0.6.1')
        if not entries:
            errors.append(f'{path}: empty quantization parameters')
            return None
        return entries

    for name, module in model.named_modules():
        if any(name == prefix or name.startswith(prefix + '.') for prefix in native_names):
            continue
        activation = {}
        for kind in ('input', 'output'):
            quantizers = flatten_nn_module_list(getattr(module, f'{kind}_quantizers', []))
            slots = {}
            for index, quantizer in enumerate(quantizers):
                if kind == 'input' and (name, index) in non_tensor_inputs:
                    continue
                values = read_quantizer(quantizer, f'{name}.{kind}[{index}]')
                if values is not None:
                    # AIMET's module-boundary activation schema stores one record.
                    if len(values) != 1:
                        errors.append(f'{name}.{kind}[{index}]: unsupported multi-record activation encoding')
                    slots[str(index)] = values[0]
            if slots:
                export = getattr(module, f'export_{kind}_encodings', None)
                if export is not None:
                    for index, values in enumerate(export('0.6.1')):
                        if values is not None:
                            slots[str(index)] = values[0]
                activation[kind] = slots
        if activation:
            expected['activation_encodings'][name] = activation
        for param, quantizer in getattr(module, 'param_quantizers', {}).items():
            key = f'{name}.{param}'
            values = read_quantizer(quantizer, key)
            if values is not None:
                expected['param_encodings'][key] = values

    _fail(errors)
    # Use the same public schema transformations as export, including optimized
    # GRU internal-op grouping, bidirectional packing and per-channel compaction.
    postprocess_encodings_inplace(expected, verbose=False)
    for name, module, _ in native:
        if not module.use_quantization:
            continue
        if not module.is_calibrated():
            errors.append(f'{name}: enabled recurrent quantizer is not calibrated')
            continue
        module.export_quant_params_to_aimet_format(expected, module_name=name)
    _fail(errors)
    return expected


def _compare(expected, actual, path: str, errors: list[str]) -> None:
    """Require every expected field, port and channel, with its calibrated value."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f'{path}: expected an encoding object')
            return
        for key, value in expected.items():
            if key not in actual:
                errors.append(f'{path}.{key}: missing quantization parameters')
            else:
                _compare(value, actual[key], f'{path}.{key}', errors)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            errors.append(f'{path}: encoding channel/entry count mismatch (expected {len(expected)})')
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f'{path}[{index}]', errors)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual)
                or not math.isclose(expected, actual, rel_tol=1e-6, abs_tol=0)):
            errors.append(f'{path}: encoding value differs from QuantSim ({actual!r}, expected {expected!r})')
    elif actual != expected:
        errors.append(f'{path}: encoding value differs from QuantSim ({actual!r}, expected {expected!r})')


def _check_scales(value, path: str, errors: list[str]) -> None:
    try:
        validate_compiler_encodings(value)
    except ValueError as exc:
        errors.append(f'{path}: {exc}')


def _io_encoding(entry, index):
    if isinstance(entry, list):
        return entry[index] if index < len(entry) else None
    if isinstance(entry, dict):
        return entry.get(str(index))
    return None


def _has_encoding(value):
    return (isinstance(value, dict) and 'bitwidth' in value and 'dtype' in value
            and (str(value['dtype']).upper().startswith('FLOAT') or 'scale' in value))


def _check_node_encodings(graph, encodings, scalar_inputs, reference_graph, errors):
    """Check actual compute operands even if QuantSim never created a quantizer.

    Ordinary keys must match final node names exactly. A module-level encoding
    does not cover unquantized arithmetic introduced by ONNX decomposition.
    Recurrent operators use their explicit packed schema instead of generic IO.
    """
    activations = encodings.get('activation_encodings', {})
    params = encodings.get('param_encodings', {})
    if not isinstance(activations, dict) or not isinstance(params, dict):
        return  # The model-based comparison reports malformed containers.
    graph = onnx.shape_inference.infer_shapes(graph, data_prop=True)
    nodes = list(graph.graph.node)
    initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    tensors = {value.name: value.type.tensor_type.elem_type
               for value in (*graph.graph.input, *graph.graph.output, *graph.graph.value_info)
               if value.type.HasField('tensor_type')}
    tensors.update({name: value.data_type for name, value in initializers.items()})
    scalar_constants = {name for name, value in initializers.items() if not value.dims}
    for node in nodes:
        if node.op_type == 'Constant':
            for attribute in node.attribute:
                if attribute.name == 'value':
                    tensors[node.output[0]] = attribute.t.data_type
                    if not attribute.t.dims:
                        scalar_constants.add(node.output[0])
                elif attribute.name in ('value_float', 'value_int'):
                    tensors[node.output[0]] = (onnx.TensorProto.FLOAT if attribute.name == 'value_float'
                                              else onnx.TensorProto.INT64)
                    scalar_constants.add(node.output[0])

    def floating(tensor):
        dtype = tensors.get(tensor)
        if dtype in (None, onnx.TensorProto.UNDEFINED):
            return None
        name = onnx.TensorProto.DataType.Name(dtype)
        return name.startswith('FLOAT') or name in ('DOUBLE', 'BFLOAT16')

    for node in nodes:
        if not is_compute_op(node.op_type):
            continue
        # Native recurrent operators pack parameters/state grids under one key.
        entry = activations.get(node.name)
        if node.op_type in ('GRU', 'LSTM'):
            if not isinstance(entry, dict) or not _has_encoding(_io_encoding(entry.get('input'), 0)):
                errors.append(f'ONNX {node.name} ({node.op_type}): missing encoding input[0]')
            for index in (1, 2, 3):
                if (index < len(node.input) and node.input[index]
                        and not _has_encoding(params.get(node.input[index]))):
                    errors.append(f'ONNX {node.name} ({node.op_type}) input[{index}]: '
                                  f'missing parameter encoding {node.input[index]}')
            continue
        required = []
        for index, tensor in enumerate(node.input):
            if not tensor:
                continue
            is_float = floating(tensor)
            if is_float is None:
                errors.append(f'ONNX {node.name} ({node.op_type}) input[{index}]: unknown tensor dtype')
            elif is_float:
                # A Python scalar operand need not have a tensor quantizer. The
                # ONNX exporter may move that scalar to a different input port.
                if tensor in scalar_constants:
                    continue
                required.append((index, tensor))
        if not required:
            continue  # Shape/index arithmetic has no floating data operands.
        if not node.name or not isinstance(entry, dict):
            errors.append(f'ONNX {node.name or "<unnamed>"} ({node.op_type}): '
                          'missing activation encoding key matching node name')
            continue
        for index, tensor in required:
            if _has_encoding(params.get(tensor)):
                continue  # Conv/Gemm weights and biases are parameter encodings.
            slot = index
            if len(required) == 1 and node.op_type in ('Add', 'Sub', 'Mul', 'Div', 'Pow'):
                # torch.add(2., x) has one tensor argument; legacy ONNX can swap
                # its commutative operands, whereas torch encodings retain slot 1.
                scalar_slots = {i for name, i in scalar_inputs if name == node.name}
                tensor_slots = set(range(len(node.input))) - scalar_slots
                if scalar_slots and len(tensor_slots) == 1:
                    slot = tensor_slots.pop()
            if not _has_encoding(_io_encoding(entry.get('input'), slot)):
                errors.append(f'ONNX {node.name} ({node.op_type}) input[{index}] '
                              f'({tensor}): missing encoding input[{slot}] or parameter encoding')

    # Container encodings and entries for nodes eliminated by simplification
    # are legitimate only when their provenance is present in the raw graph.
    final_names = {node.name for node in nodes}
    raw_names = {node.name for node in reference_graph.graph.node} if reference_graph else set()
    raw_params = {tensor.name for tensor in reference_graph.graph.initializer} if reference_graph else set()
    for name in activations:
        if name in final_names or name in raw_names:
            continue
        if any(node.startswith(name + '.') or node.startswith(name + '#')
               for node in final_names | raw_names):
            continue
        errors.append(f'activation_encodings.{name}: encoding key has no corresponding ONNX node')
    for name in params:
        if name not in initializers and name not in raw_params:
            errors.append(f'param_encodings.{name}: encoding key has no corresponding ONNX initializer')


def validate_exported_artifacts(onnx_path: str, encodings_path: str, expected: dict,
                                *, scalar_inputs=(), reference_graph=None) -> None:
    """Fail unless the final ONNX is valid and all required encodings survived.

    ``expected`` must come from collect_expected_encodings, never from an
    exporter-generated JSON file: that file may already have lost encodings.
    Also require exact encoding keys and floating-input coverage for final
    compute nodes. This does not validate numerical execution equivalence.
    """
    errors = []
    graph = None
    try:
        onnx.checker.check_model(onnx_path, full_check=True)
        graph = onnx.load(onnx_path)
        counts = Counter(node.name for node in graph.graph.node if node.name)
        for name, count in counts.items():
            if count > 1:
                errors.append(f'ONNX node {name}: duplicate name ({count})')
        # Recurrent entries are packed for a single, named ONNX operator.
        for name, entry in expected['activation_encodings'].items():
            for op_type in ('GRU', 'LSTM'):
                if entry.get(f'is_{op_type}'):
                    matches = sum(node.name == name and node.op_type == op_type
                                  for node in graph.graph.node)
                    if matches != 1:
                        errors.append(f'{name}: expected one ONNX {op_type}, found {matches}')
    except Exception as exc:
        errors.append(f'ONNX {onnx_path}: {exc}')
    try:
        with open(encodings_path, encoding='utf-8') as stream:
            actual = json.load(stream)
        _compare(expected, actual, 'encodings', errors)
        _check_scales(actual, 'encodings', errors)
        if graph is not None and isinstance(actual, dict):
            _check_node_encodings(graph, actual, scalar_inputs, reference_graph, errors)
    except (OSError, ValueError) as exc:
        errors.append(f'encodings {encodings_path}: {exc}')
    _fail(errors)
