"""Check build-time installation isolation and artifact publication boundaries."""
from argparse import Namespace
from contextlib import ExitStack
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import tarfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_build_artifacts', ROOT / 'scripts/release/artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


class ReleaseBuildTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-build-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'export'
        self.output.mkdir()
        self.name = 'rx-met-v1.1.0-cu118-linux-amd64'
        self.args = Namespace(output_dir=self.output, replace=True, runtime_base_url='', install_prefix=None)
        self.prefixes = []
        self.fail_install = False
        self.revision = b'first-revision'
        self.wheel_dir = self.root / 'wheels'
        self.wheel_dir.mkdir()
        (self.wheel_dir / 'example.whl').write_bytes(b'wheel fixture')
        for relative in (
            'examples/README.md', 'packaging/release-bundle/README.md.in',
            'scripts/release/verify_install.py', 'scripts/release/verify_runtime_archive.py',
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture\n')
        (self.root / 'packaging/release-bundle/install.sh.in').write_text('#!/bin/bash\n@BOOTSTRAP@\n')
        for target, options in (
            ('ROOT', {'new': self.root}),
            ('inspect_wheels', {'return_value': [{'filename': 'example.whl', 'unpacked_size_bytes': 1}]}),
            ('source_digest', {'return_value': 'source-digest'}),
            ('delivery_digest', {'return_value': 'delivery-digest'}),
            ('run', {'side_effect': self.run_command}),
        ):
            patcher = patch.object(artifacts, target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(artifacts.subprocess, 'check_output', side_effect=self.git_output)
        patcher.start()
        self.addCleanup(patcher.stop)

    def git_output(self, command, **kwargs):
        return self.revision if command[1] == 'rev-parse' else b''

    def run_command(self, *command, **kwargs):
        if command[0] == 'bash':
            return
        prefix = Path(command[command.index('--prefix') + 1])
        self.prefixes.append(prefix)
        destination = prefix / self.name
        manifest = Path(command[0]).parent / 'manifest.json'
        if destination.exists() and (destination / 'manifest.json').read_bytes() != manifest.read_bytes():
            raise subprocess.CalledProcessError(1, command, stderr='existing installation differs')
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'manifest.json').write_bytes(manifest.read_bytes())
        if self.fail_install:
            raise subprocess.CalledProcessError(1, command, stderr='injected verification failure')

    def assemble(self):
        record = dict(schema_version=1, environment_name='rx-quant-env', environment_version='v1',
                      variant='cu118', config_sha256='config', image_id='runtime', filename='runtime.tar.gz')
        artifacts.assemble(self.args, {'environment_version': 'v1'}, 'cu118', {'Id': 'runtime'},
                           {'Id': 'build'}, record, self.root / 'runtime.tar.gz', self.wheel_dir,
                           {'python_version': '3.10'}, version='1.1.0', changelog='# Changelog\n')

    def bundle_manifest(self):
        with tarfile.open(self.output / (self.name + '.tar.gz')) as archive:
            return archive.extractfile(self.name + '/manifest.json').read()

    def test_replacing_changed_bundle_uses_fresh_installation_and_cleans_up(self):
        previous = self.root / '.release/installed' / self.name
        previous.mkdir(parents=True)
        (previous / 'manifest.json').write_text('old installation')
        self.assemble()
        self.revision = b'second-revision'
        self.assemble()
        self.assertEqual(len(set(self.prefixes)), 2)
        self.assertTrue(all(not prefix.exists() for prefix in self.prefixes))
        self.assertEqual((previous / 'manifest.json').read_text(), 'old installation')
        self.assertIn('second-revision', self.bundle_manifest().decode())
        self.assertTrue((self.output / (self.name + '.tar.gz')).is_file())
        self.assertFalse(any(path.is_dir() for path in self.output.iterdir()))
        self.assertFalse(list((self.root / '.release/staging').glob('.bundle-*')))

    def test_failed_temporary_verification_preserves_previous_artifacts(self):
        self.assemble()
        manifest = self.bundle_manifest()
        packed = (self.output / (self.name + '.tar.gz')).read_bytes()
        self.revision = b'failed-revision'
        self.fail_install = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.assemble()
        self.assertFalse(self.prefixes[-1].exists())
        self.assertEqual(self.bundle_manifest(), manifest)
        self.assertEqual((self.output / (self.name + '.tar.gz')).read_bytes(), packed)
        self.assertFalse(any(path.is_dir() for path in self.output.iterdir()))
        self.assertFalse(list((self.root / '.release/staging').glob('.bundle-*')))

    def test_explicit_install_prefix_is_retained_and_not_overwritten(self):
        self.args.install_prefix = self.root / 'retained installation'
        self.assemble()
        installed = self.args.install_prefix / self.name / 'manifest.json'
        manifest = installed.read_bytes()
        self.assertEqual(self.prefixes, [self.args.install_prefix])
        self.revision = b'changed-revision'
        with self.assertRaises(subprocess.CalledProcessError):
            self.assemble()
        self.assertEqual(installed.read_bytes(), manifest)
        self.assertFalse(any(path.is_dir() for path in self.output.iterdir()))
        self.assertFalse(list((self.root / '.release/staging').glob('.bundle-*')))

    def assert_existing_partial_is_preserved(self, *, symlink, fail_archive):
        partial = self.output / (self.name + '.tar.gz.partial')
        retained = self.root / 'partial symlink target' if symlink else partial
        previous = b'existing partial content must remain untouched'
        retained.write_bytes(previous)
        if symlink:
            partial.symlink_to(retained)
        if fail_archive:
            with patch.object(tarfile.TarFile, 'add', side_effect=RuntimeError('injected archive failure')), \
                    self.assertRaisesRegex(RuntimeError, 'injected archive failure'):
                self.assemble()
        else:
            self.assemble()
        self.assertEqual(retained.read_bytes(), previous)
        if symlink:
            self.assertTrue(partial.is_symlink())
            self.assertEqual(partial.readlink(), retained)
        self.assertFalse(list(self.output.glob('.archive-*')))
        self.assertFalse(list((self.root / '.release/staging').glob('.bundle-*')))
        self.assertTrue(all(not prefix.exists() for prefix in self.prefixes))

    def test_successful_archive_preserves_existing_partial_file(self):
        self.assert_existing_partial_is_preserved(symlink=False, fail_archive=False)

    def test_successful_archive_preserves_existing_partial_symlink_and_target(self):
        self.assert_existing_partial_is_preserved(symlink=True, fail_archive=False)

    def test_failed_archive_preserves_existing_partial_file(self):
        self.assert_existing_partial_is_preserved(symlink=False, fail_archive=True)

    def test_failed_archive_preserves_existing_partial_symlink_and_target(self):
        self.assert_existing_partial_is_preserved(symlink=True, fail_archive=True)


class ReleaseBuildPreflightTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-preflight-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'VERSION').write_text('1.1.0\n')
        self.output = self.root / '.release/export/rx-met-v1.1.0'
        matrix = {'environment_version': 'v1', 'variants': {'cpu': {}, 'cu118': {}, 'cu126': {}, 'cu130': {}}}
        for obj, target, options in (
            (artifacts, 'ROOT', {'new': self.root}),
            (artifacts.dependencies, 'load_matrix', {'return_value': matrix}),
            (artifacts.dependencies, 'check_generated_artifacts', {}),
            (artifacts, 'release_changelog', {'return_value': '# Changelog\n'}),
        ):
            patcher = patch.object(obj, target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(artifacts.environment_archives, 'ensure_image',
                               side_effect=RuntimeError('Docker work started'))
        self.ensure_image = patcher.start()
        self.addCleanup(patcher.stop)

    def existing(self, variant, *, archive=False):
        path = self.output / f'rx-met-v1.1.0-{variant}-linux-amd64'
        if archive:
            path = path.with_name(path.name + '.tar.gz')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'existing archive')
        else:
            path.mkdir(parents=True)
        return path

    def assert_conflict_before_docker(self, path, *options):
        with patch('sys.argv', ['artifacts.py', 'build', *options]), \
                self.assertRaisesRegex(RuntimeError, '--replace') as raised:
            artifacts.main()
        self.assertIn(str(path), str(raised.exception))
        self.ensure_image.assert_not_called()
        self.assertFalse((self.root / '.release/wheels').exists())
        self.assertFalse((self.root / '.release/runtime-images').exists())

    def test_existing_selected_archive_is_rejected_before_docker(self):
        path = self.existing('cu118', archive=True)
        self.assert_conflict_before_docker(path, 'cu118')

    def test_existing_cpu_archive_is_rejected_before_docker(self):
        path = self.existing('cpu', archive=True)
        self.assert_conflict_before_docker(path, 'cpu')

    def test_later_archive_in_custom_output_blocks_all_selected_variants(self):
        self.output = self.root / 'custom output'
        path = self.existing('cu126', archive=True)
        self.assert_conflict_before_docker(path, 'cu118', 'cu126', '--with-env',
                                          '--output-dir', str(self.output))
        self.assertEqual(path.read_bytes(), b'existing archive')
        self.assertEqual(list(self.output.iterdir()), [path])

    def test_default_targets_are_all_checked_before_starting(self):
        path = self.existing('cu130', archive=True)
        self.assert_conflict_before_docker(path)

    def test_all_conflicting_archives_are_reported_before_docker(self):
        paths = [self.existing(variant, archive=True) for variant in ('cu118', 'cu130')]
        with patch('sys.argv', ['artifacts.py', 'build']), \
                self.assertRaisesRegex(RuntimeError, '--replace') as raised:
            artifacts.main()
        for path in paths:
            self.assertIn(str(path), str(raised.exception))
        self.ensure_image.assert_not_called()

    def test_replace_allows_existing_outputs_to_reach_environment_validation(self):
        path = self.existing('cu118', archive=True)
        with patch('sys.argv', ['artifacts.py', 'build', 'cu118', '--replace']), \
                self.assertRaisesRegex(RuntimeError, 'Docker work started'):
            artifacts.main()
        self.ensure_image.assert_called_once()
        self.assertEqual(path.read_bytes(), b'existing archive')

    def assert_reaches_environment_validation(self, *options):
        with patch('sys.argv', ['artifacts.py', 'build', *options]), \
                self.assertRaisesRegex(RuntimeError, 'Docker work started'):
            artifacts.main()
        self.ensure_image.assert_called_once()

    def test_existing_extracted_bundle_does_not_block_selected_archive(self):
        path = self.existing('cu118')
        self.assert_reaches_environment_validation('cu118')
        self.assertTrue(path.is_dir())

    def test_unselected_archive_does_not_block_selected_variant(self):
        path = self.existing('cpu', archive=True)
        self.assert_reaches_environment_validation('cu118')
        self.assertEqual(path.read_bytes(), b'existing archive')

    def test_cpu_archive_does_not_block_default_cuda_variants(self):
        path = self.existing('cpu', archive=True)
        self.assert_reaches_environment_validation()
        self.assertEqual(path.read_bytes(), b'existing archive')

    def test_existing_readme_does_not_block_build(self):
        self.output.mkdir(parents=True)
        path = self.output / 'README.md'
        path.write_text('retained delivery note\n')
        self.assert_reaches_environment_validation('cu118')
        self.assertEqual(path.read_text(), 'retained delivery note\n')

    def test_checksum_without_selected_archive_does_not_block_build(self):
        self.output.mkdir(parents=True)
        path = self.output / 'rx-met-v1.1.0-cu118-linux-amd64.tar.gz.sha256'
        path.write_text('orphan checksum\n')
        self.assert_reaches_environment_validation('cu118')
        self.assertEqual(path.read_text(), 'orphan checksum\n')

    def test_selected_archive_directory_is_rejected_even_with_replace(self):
        path = self.output / 'rx-met-v1.1.0-cu118-linux-amd64.tar.gz'
        path.mkdir(parents=True)
        with patch('sys.argv', ['artifacts.py', 'build', 'cu118', '--replace']), \
                self.assertRaises(RuntimeError) as raised:
            artifacts.main()
        self.assertIn(str(path), str(raised.exception))
        self.ensure_image.assert_not_called()

    def test_selected_archive_symlink_is_rejected_even_with_replace(self):
        self.output.mkdir(parents=True)
        target = self.root / 'existing archive'
        target.write_bytes(b'keep target')
        path = self.output / 'rx-met-v1.1.0-cu118-linux-amd64.tar.gz'
        path.symlink_to(target)
        with patch('sys.argv', ['artifacts.py', 'build', 'cu118', '--replace']), \
                self.assertRaises(RuntimeError) as raised:
            artifacts.main()
        self.assertIn(str(path), str(raised.exception))
        self.ensure_image.assert_not_called()
        self.assertEqual(target.read_bytes(), b'keep target')

    def assemble_fixture(self, args, matrix, variant, *unused, **kwargs):
        name = f'rx-met-v{kwargs["version"]}-{variant}-linux-amd64.tar.gz'
        (args.output_dir / name).write_bytes(f'{variant} application archive'.encode())
        return {}

    def assert_selected_targets(self, options, expected):
        self.ensure_image.side_effect = None
        self.ensure_image.return_value = {'Id': 'image'}
        record = {'environment_name': 'rx-quant-env', 'environment_version': 'v1'}
        with ExitStack() as stack:
            mocks = {}
            for obj, name, values in (
                (artifacts, 'runtime_compatibility', {'return_value': {}}),
                (artifacts, 'source_digest', {'return_value': 'source'}),
                (artifacts, 'image_info', {'return_value': {'Id': 'image'}}),
                (artifacts, 'run', {}),
                (artifacts, 'write_json', {}),
                (artifacts, 'assemble', {'side_effect': self.assemble_fixture}),
                (artifacts.environment_archives, 'image_archive',
                 {'return_value': (self.root / 'runtime.tar.gz', record)}),
                (artifacts.delivery, 'publish_environment', {}),
            ):
                mocks[name] = stack.enter_context(patch.object(obj, name, **values))
            stack.enter_context(patch('sys.argv', ['artifacts.py', 'build', *options]))
            artifacts.main()
        self.assertEqual([call.args[2] for call in mocks['assemble'].call_args_list], expected)
        self.assertEqual([call.args[1] for call in self.ensure_image.call_args_list],
                         [variant for variant in expected for _ in ('runtime', 'build')])
        self.assertEqual([call.args[1] for call in mocks['image_archive'].call_args_list], expected)
        self.assertTrue(all(call.kwargs['export'] == ('--with-env' in options)
                            for call in mocks['image_archive'].call_args_list))
        self.assertEqual(mocks['publish_environment'].call_count,
                         len(expected) if '--with-env' in options else 0)
        names = [f'rx-met-v1.1.0-{variant}-linux-amd64.tar.gz' for variant in expected]
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         sorted(name for archive in names for name in (archive, archive + '.sha256')))
        for name in names:
            archive = self.output / name
            expected_checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual((self.output / (name + '.sha256')).read_text(),
                             f'{expected_checksum}  {name}\n')

    def test_default_build_excludes_cpu(self):
        self.assert_selected_targets([], ['cu118', 'cu126', 'cu130'])

    def test_default_build_with_environment_excludes_cpu(self):
        self.assert_selected_targets(['--with-env'], ['cu118', 'cu126', 'cu130'])

    def test_explicit_cpu_build_selects_only_cpu(self):
        self.assert_selected_targets(['cpu'], ['cpu'])

    def test_explicit_cpu_build_includes_requested_environment(self):
        self.assert_selected_targets(['--with-env', 'cpu'], ['cpu'])

    def test_explicit_mixed_build_preserves_requested_targets(self):
        self.assert_selected_targets(['cu126', 'cpu', '--with-env'], ['cu126', 'cpu'])


if __name__ == '__main__':
    unittest.main()
