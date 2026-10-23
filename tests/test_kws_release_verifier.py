"""Run the release verifier's Python block without training or a container."""
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aimet_common.compiler_encodings import validate_compiler_encodings


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/release/verify_kws_example.sh'


class KwsReleaseVerifierTest(unittest.TestCase):
    def verify(self, kind, *, schema=4):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / kind / 'previous'
            old.mkdir(parents=True)
            (old / 'run.json').write_text(json.dumps({'status': 'success'}))
            checked = []
            nodes = [SimpleNamespace(name='rnn.' + str(index), op_type=kind.upper())
                     for index in range(2)]

            def record(bits=8):
                return {'bitwidth': bits, 'dtype': 'INT' + str(bits), 'is_symmetric': 'True',
                        'enc_type': 'PER_TENSOR', 'scale': 2 ** (1 - bits), 'n': bits - 1,
                        'zero_point': 0, 'real_min': -1, 'real_max': 1 - 2 ** (1 - bits)}

            def example(*args, **kwargs):
                execution = root / kind / 'current'
                exports = execution / 'exports'
                exports.mkdir(parents=True)
                (execution / 'run.json').write_text(json.dumps({'status': 'success'}))
                checkpoint = execution / 'checkpoints/att_mh_rnn_kws_qat.pth'
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b'checkpoint')
                (exports / 'att_mh_rnn_kws.onnx').write_bytes(b'onnx')
                layer = {'is_' + kind.upper(): True,
                         'input': {'0': record()}, 'output': {'0': record()},
                         'internal_ops': {'cell_state': {'output': [record(16)]}}}
                document = {'schema_version': schema, 'activation_encodings':
                            {node.name: copy.deepcopy(layer) for node in nodes}, 'param_encodings': {}}
                (exports / 'att_mh_rnn_kws.json').write_text(json.dumps(document))
                (exports / 'att_mh_rnn_kws.encodings').write_text('{"raw_aimet": true}')

            def validate(model_path, encoding_path):
                self.assertEqual(model_path.parent, root / kind / 'current/exports')
                self.assertEqual(encoding_path.name, 'att_mh_rnn_kws.json')
                validate_compiler_encodings(json.loads(encoding_path.read_text()),
                                            require_po2=True, strict=True)
                checked.append(encoding_path)

            fake_pathlib = SimpleNamespace(Path=lambda value: root if value == '/output' else Path(value))
            fake_onnx = SimpleNamespace(load=lambda path: SimpleNamespace(graph=SimpleNamespace(node=nodes)),
                                        checker=SimpleNamespace(check_model=lambda model: None))
            fake_artifacts = SimpleNamespace(validate_compiler_artifacts=validate)
            block = SCRIPT.read_text().split("<<'PYTHON'", 1)[1].split('\n', 1)[1].split('\nPYTHON', 1)[0]
            with patch.dict(sys.modules, {'pathlib': fake_pathlib, 'onnx': fake_onnx,
                                         'aimet_common.compiler_artifacts': fake_artifacts}), \
                 patch.object(runpy, 'run_path', side_effect=example), \
                 patch.object(sys, 'argv', ['-', kind]):
                exec(compile(block, str(SCRIPT), 'exec'), {})
            self.assertEqual(len(checked), 1)

    def test_both_recurrent_types_read_final_json_from_the_current_run(self):
        for kind in ('gru', 'lstm'):
            with self.subTest(kind=kind):
                self.verify(kind)

    def test_legacy_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            self.verify('gru', schema=3)


if __name__ == '__main__':
    unittest.main()
