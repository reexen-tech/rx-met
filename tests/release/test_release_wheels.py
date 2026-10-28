"""Validate wheel names and metadata before assembling release bundles."""
from email.message import EmailMessage
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

from tests.support.paths import REPO_ROOT as ROOT
SPEC = importlib.util.spec_from_file_location('release_artifacts', ROOT / 'scripts/release/artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)

VERSION = '1.1.0+cpu'
RESOURCES = {
    'rx-met': 'aimet_onnx/rx_ptq/config/quick_start_full_quant.json',
    'quant-gru': 'lib/libgru_quant_shared.so',
    'quant-lstm': 'quant_lstm_resources/lstm_quant_default_v1.json',
}


class ReleaseWheelsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-wheels-test-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for name in RESOURCES:
            self.write_wheel(name)

    def write_wheel(self, name, *, metadata_name=None, filename=None, version=VERSION,
                    tag='cp310', include_resource=True):
        distribution = name.replace('-', '_')
        path = self.directory / (filename or f'{distribution}-{version}-{tag}-{tag}-linux_x86_64.whl')
        metadata = EmailMessage()
        metadata['Metadata-Version'] = '2.1'
        metadata['Name'] = metadata_name or name
        metadata['Version'] = version
        with zipfile.ZipFile(path, 'w') as wheel:
            wheel.writestr(f'{distribution}-{version}.dist-info/METADATA', metadata.as_bytes())
            if include_resource:
                wheel.writestr(RESOURCES[name], b'wheel resource fixture')
        return path

    def inspect(self):
        return artifacts.inspect_wheels(self.directory, VERSION, '3.10')

    def test_accepts_hyphenated_distribution_names_and_underscored_wheel_filenames(self):
        wheels = self.inspect()
        self.assertEqual([(item['name'], item['filename']) for item in wheels], [
            ('quant-gru', 'quant_gru-1.1.0+cpu-cp310-cp310-linux_x86_64.whl'),
            ('quant-lstm', 'quant_lstm-1.1.0+cpu-cp310-cp310-linux_x86_64.whl'),
            ('rx-met', 'rx_met-1.1.0+cpu-cp310-cp310-linux_x86_64.whl'),
        ])

    def test_rejects_underscored_distribution_metadata(self):
        self.write_wheel('quant-gru', metadata_name='quant_gru')
        with self.assertRaisesRegex(RuntimeError, 'wheel'):
            self.inspect()

    def test_rejects_nonstandard_wheel_distribution_prefixes(self):
        for name, prefix in [('rx-met', 'rx-met'), ('quant-gru', 'quant-gru'),
                             ('quant-lstm', 'quant-lstm'), ('quant-gru', 'unrelated')]:
            with self.subTest(name=name, prefix=prefix):
                path = self.write_wheel(name)
                renamed = path.with_name(f'{prefix}-{VERSION}-cp310-cp310-linux_x86_64.whl')
                path.rename(renamed)
                try:
                    with self.assertRaisesRegex(RuntimeError, 'wheel'):
                        self.inspect()
                finally:
                    renamed.rename(path)

    def test_rejects_wrong_metadata_version(self):
        path = self.write_wheel('quant-gru')
        self.write_wheel('quant-gru', filename=path.name, version='9.9.9')
        with self.assertRaisesRegex(RuntimeError, 'wheel'):
            self.inspect()

    def test_rejects_wrong_python_abi(self):
        self.write_wheel('quant-gru').unlink()
        self.write_wheel('quant-gru', tag='cp311')
        with self.assertRaisesRegex(RuntimeError, 'wheel ABI'):
            self.inspect()

    def test_rejects_missing_package_resources(self):
        self.write_wheel('quant-lstm', include_resource=False)
        with self.assertRaisesRegex(RuntimeError, 'wheel'):
            self.inspect()

    def test_rejects_missing_project_wheel(self):
        self.write_wheel('rx-met').unlink()
        with self.assertRaisesRegex(RuntimeError, 'wheel'):
            self.inspect()

    def test_rejects_unexpected_project_wheel(self):
        self.write_wheel('quant-gru', metadata_name='unrelated')
        with self.assertRaisesRegex(RuntimeError, 'wheel'):
            self.inspect()


if __name__ == '__main__':
    unittest.main()
