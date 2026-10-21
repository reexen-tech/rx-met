"""Compiler encoding rules must not depend on either quantization framework."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'compiler_encodings', ROOT / 'src/aimet_common/compiler_encodings.py')
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)


def encoding(**updates):
    result = dict(bitwidth=8, dtype='int', is_symmetric='True', enc_type='PER_TENSOR', scale=0.125,
                  offset=0, min=-16.0, max=15.875)
    result.update(updates)
    return result


def document(entry):
    return {'activation_encodings': {'conv': {'input': [entry]}},
            'param_encodings': {'conv.weight': [copy.deepcopy(entry)]}}


class CompilerEncodingTest(unittest.TestCase):
    def test_normalization_preserves_input_and_torch_parameter_arrays(self):
        original = document(encoding())
        before = copy.deepcopy(original)
        result = shared.normalize_compiler_encodings(original, require_po2=True)
        self.assertEqual(original, before)
        activation = result['activation_encodings']['conv']['input']['0']
        self.assertEqual(activation, dict(bitwidth=8, dtype='INT8', is_symmetric='True',
                                         enc_type='PER_TENSOR',
                                         scale=0.125, n=3, zero_point=0,
                                         real_min=-16.0, real_max=15.875))
        parameter = result['param_encodings']['conv.weight']
        self.assertEqual(parameter['scale'], [0.125])
        self.assertEqual(parameter['zero_point'], [0])
        self.assertEqual(parameter['n'], [3])
        self.assertEqual(shared.normalize_compiler_encodings(result, require_po2=True), result)

    def test_signed_symmetric_8_16_32_bit_encodings(self):
        for bits in (8, 16, 32):
            with self.subTest(bitwidth=bits):
                entry = encoding(bitwidth=bits, min=-(1 << (bits - 1)) * 0.125,
                                 max=((1 << (bits - 1)) - 1) * 0.125)
                result = shared.normalize_compiler_encodings(document(entry), require_po2=True)
                parameter = result['param_encodings']['conv.weight']
                self.assertEqual(parameter['dtype'], f'INT{bits}')
                self.assertEqual(parameter['zero_point'], [0])

    def test_asymmetric_offset_remains_additive_and_unsigned(self):
        entry = encoding(is_symmetric='False', offset=-7, min=-0.875, max=31.0)
        result = shared.normalize_compiler_encodings(document(entry), require_po2=True)
        parameter = result['param_encodings']['conv.weight']
        self.assertEqual(parameter['dtype'], 'UINT8')
        self.assertEqual(parameter['zero_point'], [-7])
        for q in (0, 7, 255):
            self.assertEqual((q + parameter['zero_point'][0]) * parameter['scale'][0],
                             (q - 7) * 0.125)

    def test_unsigned_symmetric_encoding(self):
        entry = encoding(dtype='uint', min=0.0, max=31.875)
        result = shared.normalize_compiler_encodings(document(entry), require_po2=True)
        self.assertEqual(result['param_encodings']['conv.weight']['dtype'], 'UINT8')
        self.assertEqual(result['param_encodings']['conv.weight']['zero_point'], [0])

    def test_channel_vectors_keep_matching_lengths(self):
        entries = [encoding(), encoding(scale=0.25, min=-32.0, max=31.75)]
        result = shared.normalize_compiler_encodings(
            {'param_encodings': {'weight': entries}}, require_po2=True)
        self.assertEqual(result['param_encodings']['weight']['n'], [3, 2])
        broken = copy.deepcopy(result)
        broken['param_encodings']['weight']['zero_point'].pop()
        with self.assertRaisesRegex(ValueError, 'weight.*zero_point'):
            shared.validate_compiler_encodings(broken, require_po2=True)

    def test_preserves_nested_recurrent_grids(self):
        entry = dict(bitwidth=8, dtype='INT8', is_symmetric='True',
                     scale=[[0.125, 0.25], [0.5, 1.0]], zero_point=[[0, 0], [0, 0]],
                     real_min=[[-16, -32], [-64, -128]],
                     real_max=[[15.875, 31.75], [63.5, 127]])
        result = shared.normalize_compiler_encodings(
            {'activation_encodings': {'gru': {'internal_ops': {'input': {'output': entry}}}}},
            require_po2=True)
        grid = result['activation_encodings']['gru']['internal_ops']['input']['output']['0']
        self.assertEqual(grid['n'], [[3, 2], [1, 0]])

    def test_float_encoding_needs_no_integer_grid(self):
        result = shared.normalize_compiler_encodings(document({'bitwidth': 16, 'dtype': 'float'}))
        self.assertEqual(result['param_encodings']['conv.weight']['dtype'], 'FLOAT16')

    def test_native_recurrent_list_protocol_is_preserved(self):
        for kind in ('GRU', 'LSTM'):
            with self.subTest(kind=kind):
                grid = dict(bitwidth=8, dtype='INT8', is_symmetric='False', scale=0.125,
                            zero_point=7, real_min=(-128 - 7) * 0.125, real_max=(127 - 7) * 0.125)
                source = {'activation_encodings': {
                    'recurrent': {f'is_{kind}': True, 'input': [grid], 'output': [grid]}},
                    'param_encodings': {'recurrent.weight_ih.weight': copy.deepcopy(grid)}}
                result = shared.normalize_compiler_encodings(source)
                self.assertEqual(result, source)

    def test_native_recurrent_indexed_ports_and_internal_lists_are_preserved(self):
        for kind in ('GRU', 'LSTM'):
            with self.subTest(kind=kind):
                grid = dict(bitwidth=8, dtype='INT8', is_symmetric='False', enc_type='PER_TENSOR',
                            scale=0.125, n=3, zero_point=7, real_min=-16.875, real_max=15.0)
                source = {'activation_encodings': {'recurrent': {
                    f'is_{kind}': True, 'input': {'0': copy.deepcopy(grid)},
                    'output': {'0': copy.deepcopy(grid)},
                    'internal_ops': {'gate': {'output': [copy.deepcopy(grid)]}},
                    'internal_ops_reverse': {'gate': {'output': [copy.deepcopy(grid)]}}}},
                    'param_encodings': {'recurrent.weight_ih.weight': copy.deepcopy(grid)}}
                self.assertEqual(shared.normalize_compiler_encodings(source, require_po2=True), source)

    def test_invalid_encoding_type_and_affine_exponent_are_rejected(self):
        for value in ('unknown', ['PER_TENSOR'], None):
            with self.subTest(enc_type=value), self.assertRaisesRegex(ValueError, 'conv.*enc_type'):
                shared.normalize_compiler_encodings(document(encoding(enc_type=value)))
        with self.assertRaisesRegex(ValueError, 'conv.*n'):
            shared.normalize_compiler_encodings(document(encoding(scale=0.1, n=4, min=-12.8, max=12.7)))

    def test_affine_scale_is_supported_but_po2_mode_rejects_it(self):
        source = document(encoding(scale=0.1, min=-12.8, max=12.7))
        result = shared.normalize_compiler_encodings(source)
        self.assertEqual(result['param_encodings']['conv.weight']['n'], [3])
        with self.assertRaisesRegex(ValueError, 'Po2'):
            shared.normalize_compiler_encodings(source, require_po2=True)

    def test_invalid_grid_is_rejected_with_its_location(self):
        for changes, field in (
            ({'scale': 0}, 'scale'), ({'scale': float('nan')}, 'scale'),
            ({'scale': float('inf')}, 'scale'), ({'zero_point': 0.5}, 'zero_point'),
            ({'zero_point': float('inf')}, 'zero_point'), ({'real_min': None}, 'real_min'),
            ({'dtype': 'INT8', 'offset': -128}, 'range'),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, f'conv.*{field}'):
                shared.normalize_compiler_encodings(document(encoding(**changes)))

    def test_po2_mode_checks_existing_exponent(self):
        with self.assertRaisesRegex(ValueError, 'conv.*n'):
            shared.normalize_compiler_encodings(document(encoding(n=4)), require_po2=True)


if __name__ == '__main__':
    unittest.main()
