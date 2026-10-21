"""Exercise publication boundaries and reusable environment attachments."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('delivery_artifacts', ROOT / 'scripts/release/artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)
delivery = artifacts.delivery


class ReleaseDeliveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='release-delivery-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.product = ROOT.name
        self.destination = self.root / (self.product + '-v1.1.0')
        self.environment = self.root / 'rx-quant-env-v1'

    def runtime(self, variant='cu126'):
        return {'schema_version': 1, 'environment_name': 'rx-quant-env',
                'environment_version': 'v1', 'config_sha256': 'configuration',
                'variant': variant, 'image_id': 'sha256:pinned-' + variant,
                'filename': f'rx-quant-env-runtime-v1-{variant}-linux-amd64.tar.gz'}

    def bundle(self, directory, variant='cu126'):
        name = f'{self.product}-v1.1.0-{variant}-linux-amd64'
        (directory / (name + '.tar.gz')).write_bytes(b'application-' + variant.encode())
        return {'bundle_name': name, 'variant': variant, 'runtime': self.runtime(variant)}

    def archive(self, variant='cu126'):
        record = self.runtime(variant)
        cache = self.root / 'cache'
        cache.mkdir(exist_ok=True)
        archive = cache / record['filename']
        archive.write_bytes(variant.encode())
        record.update(sha256=artifacts.sha256(archive), size_bytes=archive.stat().st_size)
        return archive, record

    def assert_checksums(self, directory):
        result = subprocess.run(['sha256sum', '--check', '--strict', 'SHA256SUMS'],
                                cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_app_only_release_has_no_environment_links_or_directory(self):
        with delivery.stage_release(self.destination, replace=False) as staged:
            bundle = self.bundle(staged)
            delivery.write_release_index(staged, self.product, '1.1.0', 'notes', [bundle], with_env=False)
        self.assertFalse(self.environment.exists())
        self.assertFalse(any(path.is_dir() for path in self.destination.iterdir()))
        readme = (self.destination / 'README.md').read_text()
        self.assertIn('未附带，需另行准备', readme)
        self.assertNotIn('[已附带]', readme)
        document = json.loads((self.destination / 'release.json').read_text())
        entry, = document['bundles']
        self.assertFalse(entry['environment_included'])
        self.assertEqual(entry['runtime']['image_id'], 'sha256:pinned-cu126')
        self.assertNotIn('sha256', entry['runtime'])
        self.assert_checksums(self.destination)

    def test_environment_and_product_have_independent_checksums(self):
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        with delivery.stage_release(self.destination, replace=False) as staged:
            bundle = self.bundle(staged)
            delivery.write_release_index(staged, self.product, '1.1.0', 'notes', [bundle], with_env=True)
        self.assertIn('../rx-quant-env-v1/' + archive.name, (self.destination / 'README.md').read_text())
        self.assertNotIn(record['sha256'], (self.destination / 'SHA256SUMS').read_text())
        self.assert_checksums(self.destination)
        self.assert_checksums(self.environment)

    def test_existing_product_directory_requires_replace(self):
        self.destination.mkdir()
        (self.destination / 'old').write_text('preserve')
        with self.assertRaisesRegex(RuntimeError, '--replace'):
            with delivery.stage_release(self.destination, replace=False):
                self.fail('existing release accepted')
        self.assertEqual((self.destination / 'old').read_text(), 'preserve')

    def test_replace_removes_unselected_variants_and_preserves_shared_environment(self):
        self.destination.mkdir()
        (self.destination / 'old-cu118.tar.gz').write_text('previous')
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        environment_before = {path.name: path.read_bytes() for path in self.environment.iterdir()}
        with delivery.stage_release(self.destination, replace=True) as staged:
            bundle = self.bundle(staged)
            delivery.write_release_index(staged, self.product, '1.1.0', 'notes', [bundle], with_env=False)
        self.assertFalse((self.destination / 'old-cu118.tar.gz').exists())
        self.assertEqual(environment_before, {path.name: path.read_bytes() for path in self.environment.iterdir()})

    def test_later_variant_failure_retains_entire_previous_release(self):
        self.destination.mkdir()
        (self.destination / 'previous').write_text('preserve')
        with self.assertRaisesRegex(RuntimeError, 'second variant'):
            with delivery.stage_release(self.destination, replace=True) as staged:
                self.bundle(staged, 'cu118')
                raise RuntimeError('second variant failed')
        self.assertEqual([path.name for path in self.destination.iterdir()], ['previous'])
        self.assertFalse(list(self.root.glob('.' + self.destination.name + '-*')))

    def test_publish_failure_restores_previous_release(self):
        self.destination.mkdir()
        (self.destination / 'previous').write_text('preserve')
        original = Path.replace
        patcher = None
        try:
            with self.assertRaisesRegex(OSError, 'injected rename failure'):
                with delivery.stage_release(self.destination, replace=True) as staged:
                    self.bundle(staged)
                    def replace(path, target):
                        if path == staged:
                            raise OSError('injected rename failure')
                        return original(path, target)
                    patcher = patch.object(Path, 'replace', new=replace)
                    patcher.start()
        finally:
            if patcher is not None:
                patcher.stop()
        self.assertEqual((self.destination / 'previous').read_text(), 'preserve')

    def test_environment_is_reused_and_variants_are_added(self):
        archive, record = self.archive('cu118')
        delivery.publish_environment(archive, record, self.environment)
        previous = (self.environment / archive.name).stat()
        delivery.publish_environment(archive, record, self.environment)
        self.assertEqual((self.environment / archive.name).stat().st_mtime_ns, previous.st_mtime_ns)
        other, other_record = self.archive('cu126')
        delivery.publish_environment(other, other_record, self.environment)
        document = json.loads((self.environment / 'manifest.json').read_text())
        self.assertEqual(set(document['variants']), {'cu118', 'cu126'})
        self.assertEqual((self.environment / archive.name).read_bytes(), b'cu118')
        self.assert_checksums(self.environment)

    def test_conflicting_environment_identity_is_rejected(self):
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        before = (self.environment / 'manifest.json').read_bytes()
        for key in ('environment_name', 'environment_version', 'config_sha256', 'image_id'):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, '冲突|其他环境'):
                delivery.publish_environment(archive, {**record, key: 'different'}, self.environment)
        self.assertEqual((self.environment / 'manifest.json').read_bytes(), before)

    def test_corrupt_existing_archive_is_not_overwritten(self):
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        target = self.environment / archive.name
        target.unlink()
        target.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, '归档冲突'):
            delivery.publish_environment(archive, record, self.environment)
        self.assertEqual(target.read_bytes(), b'corrupt')

    def test_old_flag_is_rejected_before_docker_or_output(self):
        with patch('sys.argv', ['artifacts.py', 'build', '--with-images']), \
                patch.object(artifacts, 'run') as run, \
                patch('sys.stderr'), self.assertRaises(SystemExit) as raised:
            artifacts.main()
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_adding_variant_does_not_accept_corruption_of_existing_variant(self):
        archive, record = self.archive('cu118')
        delivery.publish_environment(archive, record, self.environment)
        target = self.environment / archive.name
        target.unlink()
        target.write_bytes(b'corrupt old variant')
        other, other_record = self.archive('cu126')
        with self.assertRaisesRegex(RuntimeError, '归档冲突'):
            delivery.publish_environment(other, other_record, self.environment)
        self.assertFalse((self.environment / other.name).exists())

    def test_cli_replaces_only_selected_product_files_without_touching_environment(self):
        (self.root / 'VERSION').write_text('1.1.0\n')
        output = self.root / '.release/export' / (self.product + '-v1.1.0')
        output.mkdir(parents=True)
        (output / 'unselected.tar.gz').write_bytes(b'old variant')
        environment = output.parent / 'rx-quant-env-v1'
        environment.mkdir()
        (environment / 'keep').write_bytes(b'shared environment')
        targets = ['cu126', 'cpu' if hasattr(artifacts, 'cpu') else 'cu118']
        published = {'runtime_image_id': 'image', 'build_image_id': 'image'}
        def runtime_archive(variant, *args, **kwargs):
            record = self.runtime(variant)
            if variant == 'cpu':
                record.update(environment_name='rx-lm-quant-env', environment_version='deps-v2')
            return self.root / record['filename'], record
        def build(*command, **kwargs):
            output_path = next(arg.split('dest=', 1)[1] for arg in command
                               if isinstance(arg, str) and '.output=type=local,dest=' in arg)
            Path(output_path).mkdir(parents=True)
        def assemble(args, matrix, variant, runtime, build, record, *rest, **kwargs):
            manifest = self.bundle(args.output_dir, variant)
            manifest['runtime'] = record
            return manifest
        with ExitStack() as stack:
            for obj, name, options in [
                (artifacts, 'ROOT', {'new': self.root}),
                (artifacts.dependencies, 'check_generated_artifacts', {}),
                (artifacts.common, 'config', {'return_value': {'environment_version': 'v1'}}),
                (artifacts.common, 'load_record', {'return_value': published}),
                (artifacts.environment_archives, 'ensure_image', {'return_value': {'Id': 'image'}}),
                (artifacts.environment_archives, 'runtime_archive', {'side_effect': runtime_archive}),
                (artifacts, 'release_changelog', {'return_value': 'notes'}),
                (artifacts, 'runtime_compatibility', {'return_value': {}}),
                (artifacts, 'source_digest', {'return_value': 'source'}),
                (artifacts, 'image_info', {'return_value': {'Id': 'image'}}),
                (artifacts, 'run', {'side_effect': build}),
                (artifacts, 'assemble', {'side_effect': assemble}),
            ]:
                stack.enter_context(patch.object(obj, name, **options))
            if hasattr(artifacts, 'cpu'):
                stack.enter_context(patch.object(artifacts.cpu, 'config', return_value={'environment_version': 'deps-v2'}))
                stack.enter_context(patch.object(artifacts.cpu, 'load_record', return_value=published))
            stack.enter_context(patch('sys.argv', ['artifacts.py', 'build', *targets, '--replace']))
            artifacts.main()
        entries = json.loads((output / 'release.json').read_text())['bundles']
        self.assertEqual([entry['variant'] for entry in entries], targets)
        self.assertEqual(entries[1]['environment_directory'],
                         '../rx-lm-quant-env-deps-v2' if targets[1] == 'cpu' else '../rx-quant-env-v1')
        self.assertFalse(any(entry['environment_included'] for entry in entries))
        self.assertFalse((output / 'unselected.tar.gz').exists())
        self.assertEqual((environment / 'keep').read_bytes(), b'shared environment')
        self.assertEqual([path.name for path in environment.iterdir()], ['keep'])
        self.assert_checksums(output)


if __name__ == '__main__':
    unittest.main()
