"""Public GPU environment identity remains independent of product and CPU releases."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.support.paths import REPO_ROOT as ROOT
spec = importlib.util.spec_from_file_location('shared_environment', ROOT / 'scripts/environment/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class SharedEnvironmentTest(unittest.TestCase):
    def test_default_variants_exclude_cpu_and_preserve_matrix_order(self):
        matrix = {'variants': {'cu126': {}, 'cpu': {}, 'cu118': {}}}
        self.assertEqual(common.default_variants(matrix), ['cu126', 'cu118'])

    def test_environment_month_is_fixed_by_configuration(self):
        self.assertEqual(common.config()['environment_version'], '202610')
        for role in ('build', 'runtime'):
            for variant in common.config()['variants']:
                self.assertEqual(common.image_ref(role, variant), f'rx-quant-env-{role}:202610-{variant}')
        with patch.object(common, 'config', return_value={'environment_version': '200001'}):
            self.assertEqual(common.image_ref('runtime', 'cpu'), 'rx-quant-env-runtime:200001-cpu')

    def test_common_cli_exports_and_loads_each_selected_role(self):
        for action in ('export', 'load'):
            for selection, expected in (('build', ['build']), ('all', ['build', 'runtime'])):
                operation = Mock()
                with patch.object(common, 'export', operation), \
                        patch.dict('sys.modules', {'environment.runtime_archives': SimpleNamespace(load_environment=operation)}), \
                        patch('sys.argv', ['common.py', action, '--directory', '/tmp/environment',
                                          '--role', selection, 'cpu']):
                    common.main()
                self.assertEqual([call.args[2] for call in operation.call_args_list], expected)

    def test_cli_cpu_environment_requires_explicit_selection(self):
        matrix = common.config()
        defaults = [variant for variant in matrix['variants'] if variant != 'cpu']
        selections = [([], defaults), (['cpu'], ['cpu']),
                      (['cu126', '--gpu', 'cpu'], ['cu126', 'cpu'])]
        for action in ('build', 'verify', 'export', 'load', 'download'):
            for selection, expected in selections:
                with self.subTest(action=action, selection=selection), contextlib.ExitStack() as stack:
                    operations = {name: Mock() for name in ('build', 'verify', 'export', 'load', 'download')}
                    for name in ('build', 'verify', 'export', 'download'):
                        stack.enter_context(patch.object(common, name, operations[name]))
                    stack.enter_context(patch.dict('sys.modules', {
                        'environment.runtime_archives': SimpleNamespace(load_environment=operations['load'])}))
                    stack.enter_context(patch('sys.argv', ['common.py', action, '--directory', '/tmp/runtime',
                                                          '--output-dir', '/tmp/runtime', *selection]))
                    common.main()
                    self.assertEqual([call.args[0] for call in operations[action].call_args_list], expected)
                    if action == 'build':
                        self.assertEqual([call.args[0] for call in operations['verify'].call_args_list], expected)

    def test_cli_rejects_unknown_environment_variants(self):
        with patch('sys.argv', ['common.py', 'build', 'unknown']), \
                contextlib.redirect_stderr(io.StringIO()), patch.object(common, 'build') as build:
            with self.assertRaises(SystemExit) as error:
                common.main()
        self.assertEqual(error.exception.code, 2)
        build.assert_not_called()

    def test_product_and_cpu_versions_do_not_change_public_gpu_identity(self):
        original = json.loads((ROOT / 'docker/variants.json').read_text())
        modified = json.loads(json.dumps(original))
        modified.update(product_version='9.9.9', cpu_environment_version='future-cpu')
        modified['cpu_common_packages'] = {'example-cpu-only': '1.0.0'}
        # The selected public fields, not repository-specific matrix fields, are hashed.
        with patch.object(Path, 'read_text', return_value=json.dumps(original)):
            before = common.config()
        with patch.object(Path, 'read_text', return_value=json.dumps(modified)):
            after = common.config()
        self.assertEqual(before, after)
        self.assertNotIn('cpu_variant', before)

    def test_existing_version_tags_reuse_images_without_source_digest_or_records(self):
        with patch.object(common.subprocess, 'run') as run, \
                patch.object(common, 'inspect') as inspect, \
                patch.object(common, 'config_digest', side_effect=AssertionError('source digest must not gate reuse')), \
                patch.object(common.shutil, 'copytree', side_effect=AssertionError('no rebuild needed')):
            run.return_value.returncode = 0
            common.build('cu126')
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual([call.args[0] for call in inspect.call_args_list],
                         [common.image_ref(role, 'cu126') for role in ('build', 'runtime')])


    def test_same_tag_with_wrong_environment_label_is_rejected(self):
        labels = {common.LABEL + key: value for key, value in {
            'environment.name': common.NAME, 'environment.version': common.config()['environment_version'],
            'environment.role': 'runtime', 'environment.config-sha256': 'another-repository-build', 'cuda.variant': 'cu126'}.items()}
        info = {'Id': 'pinned', 'Os': 'linux', 'Architecture': 'amd64', 'Config': {'Labels': labels}}
        with patch.object(common.subprocess, 'check_output', return_value=json.dumps([info]).encode()):
            self.assertEqual(common.inspect('a-tag', 'runtime', 'cu126')['Id'], 'pinned')
        labels[common.LABEL + 'environment.role'] = 'build'
        with patch.object(common.subprocess, 'check_output', return_value=json.dumps([info]).encode()):
            with self.assertRaises(RuntimeError): common.inspect('a-tag', 'runtime', 'cu126')

    def test_prepared_headers_match_every_variant(self):
        matrix = common.config()
        for variant, spec in matrix['variants'].items():
            package = 'onnxruntime' if variant == 'cpu' else 'onnxruntime-gpu'
            version = spec['onnxruntime']['packages'][package]
            if spec['onnxruntime']['source_sha256'] != 'vendored':
                self.assertTrue((ROOT / 'docker/environment/third_party/onnxruntime-prepared' / version / 'SHA256SUMS').is_file())
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'headers'
                common.subprocess.run([str(ROOT / 'scripts/internal/prepare_onnxruntime_headers.sh'), version,
                                       str(output), spec['onnxruntime']['source_sha256']], check=True, capture_output=True)
                api = version.split('.')[1]
                self.assertIn(f'#define ORT_API_VERSION {api}', (output / 'include/onnxruntime_c_api.h').read_text())
