"""Standard subtractive zero points and legacy compiler-schema migration."""
import copy
import unittest

from tests.support import compiler_encodings as fixtures

shared = fixtures.shared


def grid(zero_point=7):
    return dict(bitwidth=8, dtype='UINT8', is_symmetric='False', enc_type='PER_TENSOR',
                scale=.125, n=3, zero_point=zero_point,
                real_min=-zero_point * .125, real_max=(255 - zero_point) * .125)


class StandardZeroPointTest(unittest.TestCase):
    def test_torch_additive_offset_becomes_standard_zero_point(self):
        entry = fixtures.encoding(is_symmetric='False', offset=-7, min=-.875, max=31.)
        result = shared.normalize_compiler_encodings(fixtures.document(entry), require_po2=True)
        self.assertEqual(result['schema_version'], 4)
        record = result['activation_encodings']['conv']['input']['0']
        self.assertEqual(record['zero_point'], 7)
        self.assertEqual(record['dtype'], 'UINT8')
        for code in range(256):
            self.assertEqual((code - record['zero_point']) * record['scale'], (code - 7) * .125)
        self.assertEqual(shared.normalize_compiler_encodings(result, require_po2=True), result)

    def test_legacy_mixed_document_converts_only_ordinary_zero_points(self):
        for version in (None, 1, 2, 3):
            with self.subTest(version=version):
                ordinary = grid()
                ordinary['zero_point'] = -7
                recurrent = grid()
                source = {'activation_encodings': {
                    'conv': {'input': {'0': ordinary}},
                    'gru': {'is_GRU': True, 'input': {'0': copy.deepcopy(recurrent)}},
                    'lstm': {'is_LSTM': True, 'output': {'0': copy.deepcopy(recurrent)}}},
                    'param_encodings': {'gru.weight': copy.deepcopy(recurrent)}}
                if version is not None:
                    source['schema_version'] = version
                before = copy.deepcopy(source)
                result = shared.normalize_compiler_encodings(source, require_po2=True)
                self.assertEqual(source, before)
                self.assertEqual(result['schema_version'], 4)
                self.assertEqual(result['activation_encodings']['conv']['input']['0'], grid())
                self.assertEqual(result['activation_encodings']['gru']['input']['0'], recurrent)
                self.assertEqual(result['activation_encodings']['lstm']['output']['0'], recurrent)
                self.assertEqual(result['param_encodings']['gru.weight'], recurrent)
                self.assertEqual(shared.normalize_compiler_encodings(result), result)

    def test_signed_and_unsigned_nonzero_points_keep_every_integer_code(self):
        for bits in (8, 16, 32):
            for signed in (True, False):
                with self.subTest(bits=bits, signed=signed):
                    low = -(1 << (bits - 1)) if signed else 0
                    high = (1 << (bits - int(signed))) - 1
                    point = -7 if signed else 7
                    source = fixtures.encoding(bitwidth=bits, dtype=f'{"INT" if signed else "UINT"}{bits}',
                        is_symmetric='False', offset=-point,
                        min=(low - point) * .125, max=(high - point) * .125)
                    result = shared.normalize_compiler_encodings(fixtures.document(source), require_po2=True)
                    entry = result['activation_encodings']['conv']['input']['0']
                    self.assertEqual(entry['zero_point'], point)
                    for code in (low, point, 0, high):
                        self.assertEqual((code - entry['zero_point']) * entry['scale'],
                                         (code + source['offset']) * source['scale'])

    def test_new_version_checks_standard_ranges_for_ordinary_and_native_records(self):
        for owner in ('ordinary', 'GRU', 'LSTM'):
            with self.subTest(owner=owner):
                entry = {'input': {'0': grid()}}
                if owner != 'ordinary':
                    entry['is_' + owner] = True
                source = {'schema_version': 4, 'activation_encodings': {'op': entry}}
                shared.validate_compiler_encodings(source)
                entry['input']['0']['zero_point'] = -7
                with self.assertRaisesRegex(ValueError, 'op.*range'):
                    shared.validate_compiler_encodings(source)

    def test_unknown_version_and_conflicting_offset_are_rejected(self):
        for version in (True, 0, 5, '4'):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'schema_version'):
                shared.normalize_compiler_encodings({'schema_version': version})
        source = {'schema_version': 4, 'activation_encodings': {'op': {'input': {'0': grid()}}}}
        source['activation_encodings']['op']['input']['0']['offset'] = 7
        with self.assertRaisesRegex(ValueError, 'op.*offset.*zero_point'):
            shared.normalize_compiler_encodings(source)


if __name__ == '__main__':
    unittest.main()
