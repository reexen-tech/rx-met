"""Assertions for exported public recurrent encoding documents."""

import math


def assert_public_encoding_document(test, document, schema_version=4):
    """Compiler encodings never contain standalone LSTM checkpoint metadata."""
    forbidden = {'quant_lstm_encodings', 'model_info', 'execution_metadata'}
    def visit(value, root=False):
        if isinstance(value, dict):
            test.assertFalse(forbidden.intersection(value))
            if str(value.get('dtype', '')).startswith(('INT', 'UINT')):
                test.assertTrue({'bitwidth', 'dtype', 'is_symmetric', 'enc_type',
                                 'scale', 'n', 'zero_point', 'real_min', 'real_max'}.issubset(value))
            if not root:
                test.assertNotIn('schema_version', value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(document, root=True)
    test.assertEqual(document['schema_version'], schema_version)


def assert_recurrent_encoding_bounds(test, record):
    bits = record['bitwidth']
    unsigned = record['dtype'].startswith('UINT')
    qmin = 0 if unsigned else -(1 << (bits - 1))
    qmax = (1 << (bits if unsigned else bits - 1)) - 1
    def vector(key):
        value = record[key]
        return value if isinstance(value, list) else [value]
    fields = [vector(key) for key in ('scale', 'zero_point', 'real_min', 'real_max')]
    test.assertEqual(len({len(values) for values in fields}), 1)
    for scale, zp, lo, hi in zip(*fields):
        test.assertTrue(math.isclose(lo, (qmin - zp) * scale, rel_tol=1e-6, abs_tol=1e-9))
        test.assertTrue(math.isclose(hi, (qmax - zp) * scale, rel_tol=1e-6, abs_tol=1e-9))
    expected_n = [round(-math.log2(scale)) for scale in vector('scale')]
    test.assertEqual(vector('n'), expected_n)


def assert_recurrent_encoding_schema(test, layer):
    """One schema check applied to real QuantGRU and QuantLSTM exports."""
    fields = {'dtype', 'bitwidth', 'is_symmetric', 'enc_type',
              'scale', 'n', 'zero_point', 'real_min', 'real_max'}
    records = []
    for kind in ('input', 'output'):
        test.assertIsInstance(layer[kind], dict)
        test.assertEqual(set(layer[kind]), {'0'})
        records.extend(layer[kind].values())
    for kind in ('internal_ops', 'internal_ops_reverse'):
        for operation in layer.get(kind, {}).values():
            test.assertIn('output', operation)
            test.assertIsInstance(operation['output'], list)
            test.assertEqual(len(operation['output']), 1)
            records.extend(operation['output'])
    for record in records:
        assert_recurrent_encoding_bounds(test, record)
        test.assertEqual(set(record), fields)
        test.assertIsInstance(record['bitwidth'], int)
        test.assertIn(record['is_symmetric'], ('True', 'False'))
        test.assertEqual(record['enc_type'], 'PER_TENSOR')
        for field in ('scale', 'zero_point', 'real_min', 'real_max'):
            test.assertIsInstance(record[field], (int, float))
