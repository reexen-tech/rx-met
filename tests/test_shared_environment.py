"""Public GPU environment identity remains independent of product and CPU releases."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_environment', ROOT / 'scripts/environment/common.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class SharedEnvironmentTest(unittest.TestCase):
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

    def test_changed_config_or_missing_variant_rejects_published_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'images.json'
            document = {'environment_name': common.NAME, 'environment_version': common.config()['environment_version'],
                        'config_sha256': common.config_digest(), 'variants': {'cu126': {'runtime_image_id': 'pinned'}}}
            path.write_text(json.dumps(document))
            self.assertEqual(common.load_record('cu126', path)['runtime_image_id'], 'pinned')
            with self.assertRaises(KeyError): common.load_record('cu118', path)
            document['config_sha256'] = 'different'
            path.write_text(json.dumps(document))
            with self.assertRaises(RuntimeError): common.load_record('cu126', path)

    def test_same_tag_with_wrong_environment_label_is_rejected(self):
        labels = {common.LABEL + key: value for key, value in {
            'environment.name': common.NAME, 'environment.version': common.config()['environment_version'],
            'environment.role': 'runtime', 'environment.config-sha256': common.config_digest(), 'cuda.variant': 'cu126'}.items()}
        info = {'Id': 'pinned', 'Os': 'linux', 'Architecture': 'amd64', 'Config': {'Labels': labels}}
        with patch.object(common.subprocess, 'check_output', return_value=json.dumps([info]).encode()):
            self.assertEqual(common.inspect('a-tag', 'runtime', 'cu126')['Id'], 'pinned')
        labels[common.LABEL + 'environment.role'] = 'build'
        with patch.object(common.subprocess, 'check_output', return_value=json.dumps([info]).encode()):
            with self.assertRaises(RuntimeError): common.inspect('a-tag', 'runtime', 'cu126')

    def test_prepared_headers_match_every_gpu_variant(self):
        matrix = common.config()
        for variant, spec in matrix['variants'].items():
            version = spec['onnxruntime']['packages']['onnxruntime-gpu']
            if spec['onnxruntime']['source_sha256'] != 'vendored':
                self.assertTrue((ROOT / 'native/third_party/onnxruntime-prepared' / version / 'SHA256SUMS').is_file())
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'headers'
                common.subprocess.run([str(ROOT / 'scripts/internal/prepare_onnxruntime_headers.sh'), version,
                                       str(output), spec['onnxruntime']['source_sha256']], check=True, capture_output=True)
                api = version.split('.')[1]
                self.assertIn(f'#define ORT_API_VERSION {api}', (output / 'include/onnxruntime_c_api.h').read_text())
