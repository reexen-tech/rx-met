"""Validate compiler schema 4 on disk without migrating or repairing records."""
import copy
import math
import unittest

from aimet_common import compiler_encodings as shared
from aimet_common.quantization_grid import IntegerGrid


def record(**updates):
    value = dict(bitwidth=8, dtype='INT8', is_symmetric='True',
                 enc_type='PER_TENSOR', scale=.125, n=3, zero_point=0,
                 real_min=-16., real_max=15.875)
    value.update(updates)
    return value


def document(value=None):
    return dict(schema_version=4,
                activation_encodings={'op': {'input': {'0': value or record()}}},
                param_encodings={})


def parameter(**updates):
    value = record()
    value.update({key: [value[key]] for key in
                  ('scale', 'n', 'zero_point', 'real_min', 'real_max')})
    value.update(updates)
    return dict(schema_version=4, activation_encodings={}, param_encodings={'weight': value})


class StrictCompilerEncodingTest(unittest.TestCase):
    def validate(self, value, **kwargs):
        return shared.validate_compiler_encodings(value, strict=True, **kwargs)

    def test_complete_current_format_and_non_po2_scales(self):
        self.validate(document(), require_po2=True)
        self.validate(parameter(), require_po2=True)
        self.validate(document(record(scale=.1, real_min=-12.8, real_max=12.7)))
        self.validate(document(record(bitwidth=16, dtype='FLOAT16')))
        with self.assertRaisesRegex(ValueError, 'Po2'):
            self.validate(document(record(scale=.1, real_min=-12.8, real_max=12.7)),
                          require_po2=True)

    def test_schema4_required_without_repairing_input(self):
        for version in (None, 3, '4', True):
            value = document()
            if version is None:
                value.pop('schema_version')
            else:
                value['schema_version'] = version
            before = copy.deepcopy(value)
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'schema_version'):
                self.validate(value)
            self.assertEqual(value, before)

    def test_all_integer_fields_required_even_if_record_discovery_would_skip_it(self):
        for key in record():
            value = record()
            value.pop(key)
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, f'op.*{key}'):
                self.validate(document(value))
        value = record()
        value.pop('scale')
        value.pop('bitwidth')
        with self.assertRaisesRegex(ValueError, 'op'):
            self.validate(document(value))

    def test_strict_validation_does_not_migrate_legacy_fields(self):
        for field in ('offset', 'offset_domain', 'signed', 'min', 'max'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.validate(document(record(**{field: 0})))

    def test_indexed_ports_must_use_canonical_nonnegative_numbers(self):
        for ports in ([record()], {'01': record()}, {'-1': record()}, {'x': record()},
                      {0: record()}, {'0': []}):
            value = document()
            value['activation_encodings']['op']['input'] = ports
            with self.subTest(ports=ports), self.assertRaisesRegex(ValueError, 'op.*input'):
                self.validate(value)

    def test_native_internal_records_keep_single_output_list(self):
        value = document()
        node = value['activation_encodings']['op']
        node.update(is_LSTM=True, internal_ops={'gate': {'output': [record()]}},
                    internal_ops_reverse={'gate': {'output': [record()]}})
        self.validate(value)
        for output in (record(), {'0': record()}, [], [record(), record()]):
            changed = copy.deepcopy(value)
            changed['activation_encodings']['op']['internal_ops']['gate']['output'] = output
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'internal_ops.*output'):
                self.validate(changed)
        changed = copy.deepcopy(value)
        changed['activation_encodings']['op']['internal_ops_reverse']['gate']['output'][0].pop('n')
        with self.assertRaisesRegex(ValueError, 'internal_ops_reverse.*n'):
            self.validate(changed)

    def test_channel_fields_have_identical_shapes_and_parameter_arrays(self):
        for field in ('scale', 'n', 'zero_point', 'real_min', 'real_max'):
            value = parameter()
            value['param_encodings']['weight'][field] = record()[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f'weight.*{field}'):
                self.validate(value)
        value = parameter(scale=[.125, .25], n=[3, 2], zero_point=[0, 0],
                          real_min=[-16, -32], real_max=[15.875, 31.75], enc_type='PER_CHANNEL')
        self.validate(value)
        value['param_encodings']['weight']['n'] = [[3, 2]]
        with self.assertRaisesRegex(ValueError, 'weight.*n'):
            self.validate(value)

    def test_per_tensor_cannot_declare_different_channel_grids(self):
        for updates in (
                dict(scale=[.125, .25], n=[3, 2], zero_point=[0, 0],
                     real_min=[-16, -32], real_max=[15.875, 31.75]),
                dict(is_symmetric='False', scale=[.125, .125], n=[3, 3], zero_point=[0, 1],
                     real_min=[-16, -16.125], real_max=[15.875, 15.75])):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'PER_TENSOR'):
                self.validate(parameter(**updates))

    def test_signedness_is_independent_of_symmetry_and_symmetric_point_is_zero(self):
        for signed in (True, False):
            for symmetric in (True, False):
                grid = IntegerGrid(.125, 0 if symmetric else 7, 8, signed)
                self.validate(document(record(dtype=grid.dtype, is_symmetric=str(symmetric),
                    zero_point=grid.zero_point, real_min=float(grid.real_min), real_max=float(grid.real_max))))
            grid = IntegerGrid(.125, 7, 8, signed)
            with self.assertRaisesRegex(ValueError, 'symmetric.*zero_point'):
                self.validate(document(record(dtype=grid.dtype, zero_point=7,
                    real_min=float(grid.real_min), real_max=float(grid.real_max))))
        for symmetry in (True, 'true', 'banana', None):
            with self.subTest(symmetry=symmetry), self.assertRaisesRegex(ValueError, 'is_symmetric'):
                self.validate(document(record(is_symmetric=symmetry)))

    def test_dtype_bitwidth_exponent_and_bound_types(self):
        for updates, field in ((dict(dtype='INT16'), 'dtype'), (dict(bitwidth=True), 'bitwidth'),
                               (dict(n=3.0), 'n'), (dict(n=True), 'n'),
                               (dict(qmin=-127), 'real_min'), (dict(qmax=126), 'real_max'),
                               (dict(qmin=-129), 'range'), (dict(qmax=True), 'range')):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, field):
                self.validate(document(record(**updates)))
        self.validate(document(record(qmin=-127, real_min=-15.875)))

    def test_integer_endpoint_equality_rejects_one_and_256_int32_codes(self):
        step = 2. ** -30
        grid = IntegerGrid(step, 0, 32, True)
        value = record(bitwidth=32, dtype='INT32', scale=step, n=30,
                       real_min=float(grid.real_min), real_max=float(grid.real_max))
        self.validate(document(value), require_po2=True)
        for delta in (1, 256, -256):
            changed = dict(value, real_max=value['real_max'] + delta * step)
            with self.subTest(delta=delta), self.assertRaisesRegex(ValueError, 'real_max.*grid'):
                self.validate(document(changed))
        for updates in (dict(real_min=-1, real_max=1), dict(real_min=0, real_max=0)):
            with self.assertRaisesRegex(ValueError, 'real_min.*grid'):
                self.validate(document(record(**updates)))

    def test_endpoint_check_allows_only_double_roundoff(self):
        value = record(scale=.1, real_min=-12.8, real_max=math.nextafter(12.7, math.inf))
        self.validate(document(value))
        value['real_max'] = 12.70000001
        with self.assertRaisesRegex(ValueError, 'real_max'):
            self.validate(document(value))

    def test_po2_validation_rejects_nearby_affine_scales(self):
        for scale in (.125 * (1 + 1.e-8), math.nextafter(.125, math.inf)):
            grid = IntegerGrid(scale, 0, 8, True)
            value = document(record(scale=scale, real_min=float(grid.real_min),
                                    real_max=float(grid.real_max)))
            self.validate(value)
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'Po2'):
                self.validate(value, require_po2=True)

    def test_invalid_int64_zero_point_is_rejected_without_float_rounding(self):
        for signed, point in ((True, 1 << 63), (False, 1 << 64), (True, -(1 << 63) - 1)):
            with self.subTest(signed=signed, point=point), self.assertRaisesRegex(ValueError, 'zero_point'):
                IntegerGrid(.125, point, 64, signed)
        for signed, point in ((True, (1 << 63) - 2), (False, (1 << 64) - 2)):
            grid = IntegerGrid(.125, point, 64, signed)
            self.assertEqual(float(grid.real_max), .125)

    def test_finalize_repairs_live_endpoints_but_preserves_the_grid_and_input(self):
        step = 2. ** -30
        value = document(record(bitwidth=32, dtype='INT32', scale=step, n=30,
                                real_min=-2., real_max=2.))
        value['param_encodings']['weight'] = [copy.deepcopy(value['activation_encodings']['op']['input']['0'])]
        before = copy.deepcopy(value)
        final = shared.finalize_compiler_encodings(value, require_po2=True)
        self.assertEqual(value, before)
        self.validate(final, require_po2=True)
        self.assertEqual(final['activation_encodings']['op']['input']['0']['real_max'], 2. - step)
        self.assertEqual(final['param_encodings']['weight']['real_max'], [2. - step])
        for key in ('scale', 'zero_point', 'dtype', 'bitwidth'):
            self.assertEqual(final['activation_encodings']['op']['input']['0'][key],
                             before['activation_encodings']['op']['input']['0'][key])

    def test_legacy_validation_remains_compatible(self):
        value = document()
        value.pop('schema_version')
        item = value['activation_encodings']['op']['input']['0']
        item.pop('enc_type')
        item.pop('n')
        shared.validate_compiler_encodings(value)


if __name__ == '__main__':
    unittest.main()
