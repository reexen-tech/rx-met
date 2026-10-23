"""The comparison report must expose missing native LSTM quantization records."""

import copy
import unittest

from verify_kws_ptq_roundtrip import compare_encodings


def document(kind='LSTM'):
    record = {'bitwidth': 8, 'dtype': 'INT8', 'is_symmetric': 'True',
              'enc_type': 'PER_TENSOR', 'scale': .125, 'n': 3,
              'zero_point': 0, 'real_min': -16., 'real_max': 15.875}
    return {'activation_encodings': {'rnn': {f'is_{kind}': True,
                'input': {'0': copy.deepcopy(record)},
                'internal_ops': {'cell_state': {'output': [copy.deepcopy(record)]}}}},
            'param_encodings': {'rnn.weight': copy.deepcopy(record)}}


class KwsRoundtripComparisonTest(unittest.TestCase):
    def test_identical_gru_and_lstm_records_are_both_checked(self):
        for kind in ('GRU', 'LSTM'):
            with self.subTest(kind=kind):
                source = document(kind)
                report = compare_encodings(source, copy.deepcopy(source))
                self.assertTrue(report['json_equal'])
                self.assertTrue(report[f'{kind.lower()}_equal']['rnn']['activation'])
                self.assertEqual(report['recurrent_equal']['rnn']['kind'], kind)

    def test_missing_lstm_internals_and_parameters_are_reported(self):
        source = document()
        generic = copy.deepcopy(source)
        generic['activation_encodings']['rnn'].pop('is_LSTM')
        generic['activation_encodings']['rnn'].pop('internal_ops')
        generic['param_encodings'].clear()
        report = compare_encodings(source, generic)
        self.assertFalse(report['json_equal'])
        self.assertFalse(report['lstm_equal']['rnn']['activation'])
        self.assertFalse(report['lstm_equal']['rnn']['parameters']['rnn.weight'])
        self.assertIn(['rnn', 'internal_ops', 'cell_state', 'output', '0'],
                      report['activation_encodings']['removed_records'])

    def test_granularity_and_po2_exponent_changes_are_visible(self):
        source = document()
        changed = copy.deepcopy(source)
        changed['param_encodings']['rnn.weight'].update(enc_type='PER_CHANNEL', n=4)
        report = compare_encodings(source, changed)
        self.assertEqual(report['param_encodings']['changed_records'][0]['fields'], ['enc_type', 'n'])


if __name__ == '__main__':
    unittest.main()
