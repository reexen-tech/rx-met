"""Framework-independent JSON2 validation, matching, and shared-grid planning."""

import fnmatch
import json
from pathlib import Path


_BITWIDTH_FIELDS = frozenset({
    'weight_bitwidth', 'bias_bitwidth', 'param_bitwidth', 'input_bitwidth', 'output_bitwidth',
})
_BOOL_FIELDS = frozenset({
    'weight_symmetric', 'bias_symmetric', 'param_symmetric', 'input_symmetric',
    'output_symmetric', 'disable_quantization', 'per_channel_quantization',
})


def is_config_comment(key):
    return key.startswith('_') or key in {'comment', 'examples', 'description'}


def _validate_bitwidth(value, location):
    if type(value) is not int or not 2 <= value <= 32:
        raise ValueError(f'{location}: bitwidth must be an integer between 2 and 32')


def load_mixed_precision_config(config_file):
    """Load JSON2 while rejecting misspelled fields and invalid setting types."""
    path = Path(config_file)
    if not path.is_file():
        raise FileNotFoundError(f'配置文件不存在: {config_file}')
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Mixed-precision configuration must be an object')
    defaults = config.get('default_bitwidth', {})
    if not isinstance(defaults, dict):
        raise ValueError('default_bitwidth must be an object')
    for key, value in defaults.items():
        if is_config_comment(key):
            continue
        if key not in {'weight', 'output'}:
            raise ValueError(f'Unknown default_bitwidth field: {key}')
        _validate_bitwidth(value, f'default_bitwidth.{key}')
    default_config = config.get('default_config', {})
    if not isinstance(default_config, dict):
        raise ValueError('default_config must be an object')
    if ('disable_quantization' in default_config
            and type(default_config['disable_quantization']) is not bool):
        raise ValueError('default_config.disable_quantization must be a boolean')
    for section in ('layer_type_config', 'layer_name_config'):
        layers = config.get(section, {})
        if not isinstance(layers, dict):
            raise ValueError(f'{section} must be an object')
        for name, settings in layers.items():
            if is_config_comment(name):
                continue
            if not isinstance(settings, dict):
                raise ValueError(f'{section}.{name} must be an object')
            for key, value in settings.items():
                location = f'{section}.{name}.{key}'
                if is_config_comment(key):
                    continue
                if key in _BITWIDTH_FIELDS:
                    _validate_bitwidth(value, location)
                elif key in _BOOL_FIELDS:
                    if type(value) is not bool:
                        raise ValueError(f'{location} must be a boolean')
                elif key == 'input_bitwidths':
                    if not isinstance(value, list):
                        raise ValueError(f'{location} must be a list')
                    for index, bitwidth in enumerate(value):
                        if bitwidth is not None:
                            _validate_bitwidth(bitwidth, f'{location}[{index}]')
                else:
                    raise ValueError(f'Unknown mixed-precision field: {location}')
    return config


def select_layer_config(name, layer_type, type_configs, name_configs):
    """Resolve exact name > type > wildcard, leaving defaults to the caller."""
    if name in name_configs:
        return name_configs[name], 'name', None
    if layer_type in type_configs:
        return type_configs[layer_type], 'type', None
    for pattern, settings in name_configs.items():
        if '*' in pattern and fnmatch.fnmatch(name, pattern):
            return settings, 'pattern', pattern
    return None, None, None


def default_layer_config(default_bitwidth):
    """Translate JSON2 defaults to the settings consumed by both adapters."""
    return {
        'weight_bitwidth': default_bitwidth.get('weight'),
        'input_bitwidth': default_bitwidth.get('output'),
        'output_bitwidth': default_bitwidth.get('output'),
    }


class QuantizerConfigPlan:
    """Collect compatible settings before either adapter mutates quantizers."""

    def __init__(self):
        self._entries = {}

    def request(self, quantizer, field, value, source, *, fallback=False):
        if quantizer is None or value is None:
            return
        _, settings = self._entries.setdefault(id(quantizer), (quantizer, {}))
        previous = settings.get(field)
        if previous is not None:
            old_value, old_source, old_fallback = previous
            if not old_fallback and fallback:
                return
            if old_value != value and old_fallback == fallback:
                raise ValueError(
                    f'Conflicting shared quantizer {field}: {old_source}={old_value} '
                    f'vs {source}={value}'
                )
        settings[field] = (value, source, fallback)

    def entries(self):
        for quantizer, settings in self._entries.values():
            yield quantizer, {field: request[0] for field, request in settings.items()}
