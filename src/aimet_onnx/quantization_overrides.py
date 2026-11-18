"""Persist effective RX exclusions alongside raw quantizer encodings."""

from collections import Counter


OVERRIDES_KEY = 'rx_quantization_overrides'


def export_quantization_overrides(sim):
    disabled_ids = getattr(sim, '_rx_explicit_disabled_quantizers', set())
    disabled = sorted(name for name, quantizer in sim.qc_quantize_op_dict.items()
                      if id(quantizer) in disabled_ids and not quantizer.enabled)
    excluded = sorted(set(getattr(sim, '_excluded_layer_names', ())))
    if not disabled and not excluded:
        return None
    return {'schema_version': 1, 'disabled_quantizers': disabled, 'excluded_layers': excluded}


def _encoding_names(document):
    names = set()
    for section in ('activation_encodings', 'param_encodings', 'encodings'):
        entries = document.get(section, {})
        if isinstance(entries, dict):
            names.update(entries)
        elif isinstance(entries, list):
            names.update(entry['name'] for entry in entries
                         if isinstance(entry, dict) and isinstance(entry.get('name'), str))
    return names


def _source_model(sim):
    model = getattr(sim, '_rx_normalized_model', None)
    if model is None:
        model = getattr(sim, '_rx_original_onnx', None)
    return sim.model.model if model is None else model


def _owned_quantizer_names(sim, node):
    owned = set(node.output) | (set(node.input) & set(sim.param_names))
    group = getattr(sim, '_rx_linear_groups', {}).get(node.name)
    if group is not None:
        owned.add(group['output'])
    owned.update(alias for (owner, _), alias in getattr(sim, '_rx_input_quantizers', {}).items()
                 if owner == node.name)
    return owned


def validate_quantization_overrides(sim, document):
    """Validate exclusion declarations before any quantizer or native state loads."""
    state = document.get(OVERRIDES_KEY)
    if state is None:
        return None
    fields = {'schema_version', 'disabled_quantizers', 'excluded_layers'}
    if (not isinstance(state, dict) or set(state) != fields
            or type(state['schema_version']) is not int or state['schema_version'] != 1):
        raise ValueError('Invalid RX quantization override metadata')
    for field in ('disabled_quantizers', 'excluded_layers'):
        values = state[field]
        if (not isinstance(values, list) or any(not isinstance(name, str) or not name for name in values)
                or len(set(values)) != len(values)):
            raise ValueError(f'Invalid RX quantization override {field}')
    unknown = set(state['disabled_quantizers']) - sim.qc_quantize_op_dict.keys()
    if unknown:
        raise ValueError('RX overrides name unknown quantizers: ' + ', '.join(sorted(unknown)))
    encoded = _encoding_names(document)
    disabled_ids = {id(sim.qc_quantize_op_dict[name]) for name in state['disabled_quantizers']}
    encoded_ids = {id(sim.qc_quantize_op_dict[name]) for name in encoded
                   if name in sim.qc_quantize_op_dict}
    conflict = {name for name in encoded if name in sim.qc_quantize_op_dict
                and id(sim.qc_quantize_op_dict[name]) in disabled_ids}
    if conflict:
        raise ValueError('RX disabled or shared quantizers also have encodings: ' + ', '.join(sorted(conflict)))
    model = _source_model(sim)
    counts = Counter(node.name for node in model.graph.node)
    nodes = {node.name: node for node in model.graph.node}
    native = getattr(sim, '_rx_native_recurrent', {})
    for name in state['excluded_layers']:
        if counts[name] != 1 or name in native:
            raise ValueError(f'RX overrides name unknown or non-generic excluded layer: {name}')
        node = nodes[name]
        owned = _owned_quantizer_names(sim, node)
        owned_ids = {id(sim.qc_quantize_op_dict[tensor]) for tensor in owned
                     if tensor in sim.qc_quantize_op_dict}
        if owned & encoded or owned_ids & encoded_ids:
            raise ValueError(f'RX excluded layer {name} still has active owned encodings')
    return {field: list(state[field]) for field in ('disabled_quantizers', 'excluded_layers')}


def restore_quantization_overrides(sim, state):
    if state is None:
        disabled = getattr(sim, '_rx_explicit_disabled_quantizers', set())
        if disabled:
            inactive = {id(quantizer) for quantizer in sim.qc_quantize_op_dict.values()
                        if not quantizer.enabled}
            sim._rx_explicit_disabled_quantizers = disabled & inactive
        excluded = getattr(sim, '_excluded_layer_names', ())
        if excluded:
            nodes = {node.name: node for node in _source_model(sim).graph.node}
            sim._excluded_layer_names = [name for name in excluded if name not in nodes or not any(
                tensor in sim.qc_quantize_op_dict and sim.qc_quantize_op_dict[tensor].enabled
                for tensor in _owned_quantizer_names(sim, nodes[name]))]
        return
    for name in state['disabled_quantizers']:
        sim.qc_quantize_op_dict[name].enabled = False
    sim._rx_explicit_disabled_quantizers = {
        id(sim.qc_quantize_op_dict[name]) for name in state['disabled_quantizers']}
    sim._excluded_layer_names = list(state['excluded_layers'])
