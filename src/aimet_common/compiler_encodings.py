"""Framework-independent compiler encoding normalization and validation.

Compiler schema 4 uses ``real = scale * (q - zero_point)`` for every record.
Legacy ordinary records used an additive zero_point; native recurrent records
already used the standard subtractive convention. AIMET source offsets remain
additive and are converted only at the compiler interface.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Any, Iterable

logger = logging.getLogger(__name__)
COMPILER_SCHEMA_VERSION = 4
_VECTOR_FIELDS = {'scale', 'n', 'offset', 'zero_point', 'min', 'max', 'real_min', 'real_max'}


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _map_values(value, fn):
    return [_map_values(item, fn) for item in value] if isinstance(value, list) else fn(value)


def _values(value):
    if isinstance(value, list):
        for item in value:
            yield from _values(item)
    else:
        yield value


def _shape(value):
    if not isinstance(value, list):
        return ()
    shapes = [_shape(item) for item in value]
    if not shapes or any(shape != shapes[0] for shape in shapes):
        raise ValueError('empty or ragged channel array')
    return (len(value), *shapes[0])


def _native_owners(document):
    return {name for name, entry in document.get('activation_encodings', {}).items()
            if isinstance(entry, dict) and (entry.get('is_GRU') is True or entry.get('is_LSTM') is True)}


def _native_parameter(name, owners):
    return any(not owner or name == owner or name.startswith(owner + '.') for owner in owners)


def _records(document):
    def visit(value, path, native=False):
        if isinstance(value, dict):
            native |= value.get('is_GRU') is True or value.get('is_LSTM') is True
            if ('bitwidth' in value and 'dtype' in value) or (
                'scale' in value and any(key in value for key in ('offset', 'zero_point', 'min', 'real_min'))
            ):
                yield path, value, native
            else:
                for key, child in value.items():
                    yield from visit(child, f'{path}.{key}', native)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from visit(child, f'{path}[{index}]', native)
    yield from visit(document.get('activation_encodings', {}), 'activation_encodings')
    owners = _native_owners(document)
    for name, value in document.get('param_encodings', {}).items():
        yield from visit(value, f'param_encodings.{name}', _native_parameter(name, owners))


def _schema_version(document):
    version = document.get('schema_version', 3)
    if type(version) is not int or version not in (1, 2, 3, COMPILER_SCHEMA_VERSION):
        raise ValueError(f'schema_version: unsupported compiler encoding version {version!r}')
    return version


def _standardize_zero_points(document):
    version = _schema_version(document)
    updates = []
    for path, entry, native in _records(document):
        def point(value):
            if not _finite_number(value) or value != int(value):
                raise ValueError(f'{path}.zero_point: expected finite integers')
            return value

        zero_point = None
        if 'zero_point' in entry:
            zero_point = _map_values(entry['zero_point'], point)
            if version < COMPILER_SCHEMA_VERSION and not native:
                zero_point = _map_values(zero_point, lambda value: -value)
        if 'offset' in entry:
            from_offset = _map_values(entry['offset'], lambda value: -point(value))
            if zero_point is not None and zero_point != from_offset:
                raise ValueError(f'{path}: offset and zero_point describe different grids')
            zero_point = from_offset
        if zero_point is not None:
            updates.append((entry, zero_point))
    changed = document.get('schema_version') != COMPILER_SCHEMA_VERSION
    for entry, zero_point in updates:
        changed |= 'offset' in entry or entry.get('zero_point') != zero_point
        entry.pop('offset', None)
        entry['zero_point'] = zero_point
    document['schema_version'] = COMPILER_SCHEMA_VERSION
    return changed


def compact_encoding_records(entries: Iterable[dict[str, Any]], *, force_list: bool = False) -> dict[str, Any]:
    """Keep channel fields as arrays, even for a single parameter channel."""
    entries = list(entries)
    if not entries:
        raise ValueError('cannot compact empty encoding records')
    if len(entries) == 1 and not force_list:
        return copy.deepcopy(entries[0])
    result = {}
    for key in sorted(set().union(*(entry.keys() for entry in entries))):
        values = [entry.get(key) for entry in entries]
        result[key] = copy.deepcopy(values if key in _VECTOR_FIELDS or any(
            value != values[0] for value in values[1:]) else values[0])
    return result


def compact_per_channel_param_encodings(enc: dict, verbose: bool = True) -> tuple[dict, bool]:
    changed = False
    parameters = enc.get('param_encodings', {})
    owners = _native_owners(enc)
    if isinstance(parameters, dict):
        for name, value in list(parameters.items()):
            if _native_parameter(name, owners):
                continue
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                parameters[name] = compact_encoding_records(value, force_list=True)
                changed = True
    if changed and verbose:
        logger.info('Compacted parameter encoding channel arrays')
    return enc, changed


def flatten_activation_io_index_dict(enc: dict, verbose: bool = True) -> tuple[dict, bool]:
    changed = False
    def visit(value):
        nonlocal changed
        if not isinstance(value, dict):
            return
        if value.get('is_GRU') is True or value.get('is_LSTM') is True:
            return  # Native recurrent operators own their packed port schema.
        for key in ('input', 'output'):
            if key in value:
                child = value[key]
                if isinstance(child, list):
                    value[key] = {str(index): item for index, item in enumerate(child)}
                    changed = True
                elif isinstance(child, dict) and child and not all(
                        isinstance(index, str) and index.isdigit() for index in child):
                    value[key] = {'0': child}
                    changed = True
        for field in ('internal_ops', 'internal_ops_reverse'):
            internal = value.get(field, {})
            if isinstance(internal, dict):
                for child in internal.values():
                    visit(child)
    activations = enc.get('activation_encodings', {})
    if isinstance(activations, dict):
        for value in activations.values():
            visit(value)
    if changed and verbose:
        logger.info('Normalized activation input/output index dictionaries')
    return enc, changed


def _exponent(scale, path, require_po2, verbose=False):
    if not _finite_number(scale) or scale <= 0:
        raise ValueError(f'{path}.scale: scale must be positive and finite')
    value = -math.log2(scale)
    exponent = int(round(value))
    if abs(value - exponent) > 1e-6:
        if require_po2:
            raise ValueError(f'{path}.scale: scale is not Po2: {scale}')
        if verbose:
            logger.warning('%s: affine scale=%s has rounded exponent n=%s', path, scale, exponent)
    return exponent


def normalize_quant_fields_add_n(
    enc: dict, verbose: bool = True, *, require_po2: bool = False,
) -> tuple[dict, bool]:
    changed = _standardize_zero_points(enc)
    for path, entry, native in _records(enc):
        if not native and 'scale' in entry:
            exponent = _map_values(entry['scale'], lambda scale: _exponent(scale, path, require_po2, verbose))
            if 'n' not in entry:
                entry['n'] = exponent
                changed = True
        for old, new in (('min', 'real_min'), ('max', 'real_max')):
            if old in entry:
                value = entry.pop(old)
                if new not in entry:
                    entry[new] = value
                changed = True
        if 'is_symmetric' in entry:
            value = entry['is_symmetric']
            if str(value).lower() not in ('true', 'false'):
                raise ValueError(f'{path}.is_symmetric: expected a boolean')
            normalized = str(str(value).lower() == 'true')
            changed |= value != normalized
            entry['is_symmetric'] = normalized
    return enc, changed


def normalize_dtype_with_bitwidth(enc: dict, verbose: bool = True) -> tuple[dict, bool]:
    changed = False
    version = _schema_version(enc)
    for path, entry, native in _records(enc):
        if native:
            continue
        if 'dtype' not in entry or 'bitwidth' not in entry:
            continue
        try:
            bits = int(entry['bitwidth'])
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f'{path}.bitwidth: expected an integer') from None
        if isinstance(entry['bitwidth'], bool) or str(entry['bitwidth']) not in (str(bits), f'{bits}.0'):
            raise ValueError(f'{path}.bitwidth: expected an integer')
        changed |= entry['bitwidth'] != bits
        entry['bitwidth'] = bits
        dtype = str(entry['dtype']).upper()
        base = dtype.rstrip('0123456789')
        if base in ('INT', 'UINT', 'FLOAT'):
            if dtype == 'INT' and all(key in entry for key in ('scale', 'zero_point', 'real_min')):
                scales = list(_values(entry['scale']))
                zero_points = list(_values(entry['zero_point']))
                if version < COMPILER_SCHEMA_VERSION:
                    zero_points = [-value for value in zero_points]
                minima = list(_values(entry['real_min']))
                length = max(len(scales), len(zero_points), len(minima))
                if (all(len(values) in (1, length) for values in (scales, zero_points, minima))
                        and all(_finite_number(value) for value in (*scales, *zero_points, *minima))
                        and all(scale > 0 for scale in scales)):
                    bounds = [minima[i % len(minima)] / scales[i % len(scales)] + zero_points[i % len(zero_points)]
                              for i in range(length)]
                    dtype = 'UINT' if all(bound >= -1e-5 for bound in bounds) else 'INT'
                    base = dtype
            dtype = base + str(bits)
        changed |= dtype != entry['dtype']
        entry['dtype'] = dtype
    return enc, changed


def validate_compiler_encodings(enc: dict, *, require_po2: bool = False) -> None:
    """Validate numbers, channel shapes and representable compiler integer grids."""
    version = _schema_version(enc)
    for path, entry, native in _records(enc):
        try:
            if 'enc_type' in entry and (not isinstance(entry['enc_type'], str) or
                    entry['enc_type'] not in {'PER_TENSOR', 'PER_CHANNEL', 'PER_GATE', 'PER_BLOCK'}):
                raise ValueError('enc_type: unsupported quantization granularity')
            bits = entry['bitwidth']
            if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 64:
                raise ValueError('bitwidth: expected an integer between 1 and 64')
            dtype = entry['dtype']
            if dtype == f'FLOAT{bits}':
                continue
            if dtype not in (f'INT{bits}', f'UINT{bits}'):
                raise ValueError(f'dtype: unsupported compiler dtype {dtype!r}')
            scale = entry['scale']
            shape = _shape(scale)
            scales = list(_values(scale))
            exponents = [_exponent(value, path, require_po2) for value in scales]
            fields = {}
            for field in ('n', 'zero_point', 'real_min', 'real_max'):
                if field == 'n' and field not in entry and (native or not require_po2):
                    continue
                value = entry[field]
                if isinstance(value, list) and _shape(value) != shape:
                    raise ValueError(f'{field}: channel shape differs from scale')
                values = list(_values(value))
                if any(not _finite_number(item) for item in values):
                    raise ValueError(f'{field}: expected finite numbers')
                if field in ('n', 'zero_point') and any(item != int(item) for item in values):
                    raise ValueError(f'{field}: expected integers')
                fields[field] = values * len(scales) if len(values) == 1 else values
            if 'n' in fields and fields['n'] != exponents:
                raise ValueError('n: exponent does not match scale')
            if any(low > high for low, high in zip(fields['real_min'], fields['real_max'])):
                raise ValueError('range: real_min exceeds real_max')
            signed = dtype.startswith('INT')
            qmin = -(1 << (bits - 1)) if signed else 0
            qmax = (1 << (bits - int(signed))) - 1
            tolerance = max(1e-5, max(abs(qmin), abs(qmax)) * 1e-6)
            for index, step in enumerate(scales):
                low, high = fields['real_min'][index], fields['real_max'][index]
                zero_point = fields['zero_point'][index]
                if version < COMPILER_SCHEMA_VERSION and not native:
                    zero_point = -zero_point
                if low > high or low / step + zero_point < qmin - tolerance or high / step + zero_point > qmax + tolerance:
                    raise ValueError('range: scale/zero_point/real bounds do not fit dtype')
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            raise ValueError(f'{path}: {exc}') from exc


def normalize_compiler_encodings(enc: dict, *, require_po2: bool = False, verbose: bool = False) -> dict:
    """Normalize compiler records and migrate legacy zero points without changing grids."""
    result = copy.deepcopy(enc)
    compact_per_channel_param_encodings(result, verbose)
    flatten_activation_io_index_dict(result, verbose)
    normalize_quant_fields_add_n(result, verbose, require_po2=require_po2)
    normalize_dtype_with_bitwidth(result, verbose)
    validate_compiler_encodings(result, require_po2=require_po2)
    return result
