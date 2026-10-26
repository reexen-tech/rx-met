"""The comparison report must expose missing native LSTM quantization records."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from kws_roundtrip_artifacts import load_final_compiler_encodings, resolve_artifact_paths


def document(kind='LSTM'):
    record = {'bitwidth': 8, 'dtype': 'INT8', 'is_symmetric': 'True',
              'enc_type': 'PER_TENSOR', 'scale': .125, 'n': 3,
              'zero_point': 0, 'real_min': -16., 'real_max': 15.875}
    return {'activation_encodings': {'rnn': {f'is_{kind}': True,
                'input': {'0': copy.deepcopy(record)},
                'internal_ops': {'cell_state': {'output': [copy.deepcopy(record)]}}}},
            'param_encodings': {'rnn.weight': copy.deepcopy(record)}}


class KwsRoundtripComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from verify_kws_ptq_roundtrip import compare_encodings
        cls.compare_encodings = staticmethod(compare_encodings)

    def test_identical_gru_and_lstm_records_are_both_checked(self):
        for kind in ('GRU', 'LSTM'):
            with self.subTest(kind=kind):
                source = document(kind)
                report = self.compare_encodings(source, copy.deepcopy(source))
                self.assertTrue(report['json_equal'])
                self.assertTrue(report[f'{kind.lower()}_equal']['rnn']['activation'])
                self.assertEqual(report['recurrent_equal']['rnn']['kind'], kind)

    def test_missing_lstm_internals_and_parameters_are_reported(self):
        source = document()
        generic = copy.deepcopy(source)
        generic['activation_encodings']['rnn'].pop('is_LSTM')
        generic['activation_encodings']['rnn'].pop('internal_ops')
        generic['param_encodings'].clear()
        report = self.compare_encodings(source, generic)
        self.assertFalse(report['json_equal'])
        self.assertFalse(report['lstm_equal']['rnn']['activation'])
        self.assertFalse(report['lstm_equal']['rnn']['parameters']['rnn.weight'])
        self.assertIn(['rnn', 'internal_ops', 'cell_state', 'output', '0'],
                      report['activation_encodings']['removed_records'])

    def test_granularity_and_po2_exponent_changes_are_visible(self):
        source = document()
        changed = copy.deepcopy(source)
        changed['param_encodings']['rnn.weight'].update(enc_type='PER_CHANNEL', n=4)
        report = self.compare_encodings(source, changed)
        self.assertEqual(report['param_encodings']['changed_records'][0]['fields'], ['enc_type', 'n'])


class KwsRoundtripArtifactPathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = {}
        for stage, frontend, suffix in (('torch_ptq', 'torch', '.json'),
                                        ('onnx_ptq', 'onnx', '.json')):
            exports = self.root / stage / 'one-run' / 'exports'
            exports.mkdir(parents=True)
            model = exports / 'att_mh_rnn_kws.onnx'
            model.touch()
            encoding = model.with_suffix(suffix)
            encoding.write_text(json.dumps({'schema_version': 4, 'activation_encodings': {},
                                            'param_encodings': {}}))
            self.paths[f'{frontend}_onnx'] = model
            self.paths[f'{frontend}_encodings'] = encoding
        self.metadata = self.paths['onnx_onnx'].with_suffix('.meta.json')
        self.metadata.write_text('{}')
        self.raw = self.paths['torch_onnx'].with_suffix('.encodings')
        self.raw.write_text(json.dumps({'version': '1.0.0', 'activation_encodings': [],
                                        'param_encodings': []}))
        self.write_report()

    def write_report(self, extra=None):
        artifacts = {key: '/validation/run/' + str(path.relative_to(self.root))
                     for key, path in self.paths.items()}
        artifacts.update(extra or {})
        (self.root / 'comparison.json').write_text(json.dumps({'artifacts': artifacts}))

    def test_returned_json_wins_when_old_glob_selects_raw_aimet(self):
        legacy_path, = (self.root / 'torch_ptq').glob('*/exports/att_mh_rnn_kws.encodings')
        self.assertNotIn('schema_version', json.loads(legacy_path.read_text()))
        actual = resolve_artifact_paths(self.root)
        self.assertEqual(actual['torch_encodings'], self.paths['torch_encodings'])
        self.assertEqual(actual['onnx_encodings'], self.paths['onnx_encodings'])
        self.assertEqual(actual['onnx_metadata'], self.metadata)
        for frontend in ('torch', 'onnx'):
            self.assertEqual(load_final_compiler_encodings(actual[f'{frontend}_encodings'])['schema_version'], 4)

    def test_historical_final_encodings_report_can_be_relocated(self):
        for frontend in ('torch', 'onnx'):
            historical = self.paths[f'{frontend}_encodings'].with_suffix('.encodings')
            historical.write_text(self.paths[f'{frontend}_encodings'].read_text())
            self.paths[f'{frontend}_encodings'] = historical
        self.write_report()
        actual = resolve_artifact_paths(self.root)
        for frontend in ('torch', 'onnx'):
            historical = self.paths[f'{frontend}_encodings']
            self.assertEqual(actual[f'{frontend}_encodings'], historical)
            self.assertEqual(load_final_compiler_encodings(historical)['schema_version'], 4)

    def test_explicit_metadata_and_raw_paths_are_preserved(self):
        metadata = self.metadata.with_name('custom-metadata.json')
        metadata.write_text('{}')
        raw = self.metadata.with_name('att_mh_rnn_kws.aimet.encodings')
        raw.write_text('{}')
        self.paths.update(onnx_metadata=metadata, onnx_aimet_encodings=raw)
        self.write_report()
        actual = resolve_artifact_paths(self.root)
        self.assertEqual(actual['onnx_metadata'], metadata)
        self.assertEqual(actual['onnx_aimet_encodings'], raw)

    def test_missing_recorded_file_does_not_select_another_run(self):
        self.paths['torch_encodings'].unlink()
        other = self.root / 'torch_ptq' / 'other-run' / 'exports' / 'att_mh_rnn_kws.json'
        other.parent.mkdir(parents=True)
        other.write_text('{}')
        with self.assertRaises(FileNotFoundError):
            resolve_artifact_paths(self.root)

    def test_raw_aimet_is_rejected_even_when_report_selects_it(self):
        self.paths['torch_encodings'] = self.raw
        self.write_report()
        actual = resolve_artifact_paths(self.root)
        with self.assertRaisesRegex(ValueError, 'final compiler schema 4'):
            load_final_compiler_encodings(actual['torch_encodings'])

    def test_historical_numeric_defects_remain_available_for_audit(self):
        historical = document()
        historical['schema_version'] = 4
        historical['param_encodings']['rnn.weight']['real_max'] += .125
        self.paths['torch_encodings'].write_text(json.dumps(historical))
        # Source-format identification must not repair or hide the old endpoint bug.
        self.assertEqual(load_final_compiler_encodings(self.paths['torch_encodings']), historical)


if __name__ == '__main__':
    unittest.main()
