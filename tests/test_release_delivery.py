"""Exercise publication boundaries and reusable environment attachments."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
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
        self.environment = self.root / 'rx-quant-env-202610'

    def runtime(self, variant='cu126'):
        return {'schema_version': 1, 'environment_name': 'rx-quant-env',
                'environment_version': '202610', 'config_sha256': 'configuration',
                'variant': variant, 'role': 'runtime', 'image_id': 'sha256:pinned-' + variant,
                'filename': f'rx-quant-env-runtime-202610-{variant}-linux-amd64.docker.tar.gz'}

    def bundle_filename(self, variant='cu126'):
        return f'{self.product}-v1.1.0-{variant}-linux-amd64.tar.gz'

    def bundle(self, directory, variant='cu126'):
        filename = self.bundle_filename(variant)
        (directory / filename).write_bytes(b'application-' + variant.encode())
        return {'bundle_name': filename.removesuffix('.tar.gz'),
                'variant': variant, 'runtime': self.runtime(variant)}

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

    def assert_product_checksums(self, directory, variants):
        for variant in variants:
            filename = self.bundle_filename(variant)
            checksum = directory / (filename + '.sha256')
            self.assertEqual(checksum.read_text(),
                             f'{artifacts.sha256(directory / filename)}  {filename}\n')
            result = subprocess.run(['sha256sum', '--check', '--strict', checksum.name],
                                    cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for filename in ('README.md', 'CHANGELOG.md', 'release.json', 'SHA256SUMS'):
            self.assertFalse((directory / filename).exists(), filename)

    def snapshot(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob('*') if path.is_file()}

    def probe_lock(self, directory):
        return subprocess.run(['flock', '--nonblock', str(delivery.lock_path(directory)), 'true'],
                              capture_output=True, timeout=5).returncode

    def test_product_and_environment_locks_are_outside_export(self):
        export = self.root / '.release/export'
        product = export / self.destination.name
        environment = export / self.environment.name
        with delivery.stage_release(product, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        archive, record = self.archive()
        delivery.publish_environment(archive, record, environment)
        self.assertEqual(set(export.iterdir()), {product, environment})
        self.assertFalse(list(export.rglob('*.lock')))
        self.assertEqual(set((self.root / '.release/locks').iterdir()),
                         {delivery.lock_path(product), delivery.lock_path(environment)})

    def test_custom_output_lock_paths_use_the_canonical_target(self):
        parent = self.root / 'custom output'
        parent.mkdir()
        alias = self.root / 'output alias'
        alias.symlink_to(parent, target_is_directory=True)
        target = parent / self.destination.name
        self.assertEqual(delivery.lock_path(target).parent, parent / '.release/locks')
        self.assertEqual(delivery.lock_path(target), delivery.lock_path(alias / target.name))
        self.assertNotEqual(delivery.lock_path(target),
                            delivery.lock_path(parent / 'other output' / target.name))
        nested = self.root / '.release/export/nested' / target.name
        self.assertEqual(delivery.lock_path(nested).parent, self.root / '.release/locks')
        custom_export = parent / 'export/nested' / target.name
        self.assertEqual(delivery.lock_path(custom_export).parent, parent / '.release/locks')

    def test_product_lock_is_exclusive_and_released_after_publication(self):
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
            self.assertTrue(delivery.lock_path(self.destination).is_file())
            self.assertEqual(self.probe_lock(self.destination), 1)
        self.assertEqual(self.probe_lock(self.destination), 0)
        self.assertTrue(delivery.lock_path(self.destination).is_file())

    def test_environment_lock_is_exclusive_during_manifest_writes(self):
        archive, record = self.archive()
        write_json = delivery.common.write_json
        def checked_write(path, document):
            self.assertEqual(self.probe_lock(self.environment), 1)
            return write_json(path, document)
        with patch.object(delivery.common, 'write_json', side_effect=checked_write):
            delivery.publish_environment(archive, record, self.environment)
        self.assertEqual(self.probe_lock(self.environment), 0)

    def test_app_only_release_contains_only_archive_and_its_checksum(self):
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        self.assertFalse(self.environment.exists())
        self.assertEqual({path.name for path in self.destination.iterdir()},
                         {self.bundle_filename(), self.bundle_filename() + '.sha256'})
        self.assert_product_checksums(self.destination, ['cu126'])

    def test_environment_and_product_have_independent_checksums(self):
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        self.assertNotIn(record['sha256'],
                         (self.destination / (self.bundle_filename() + '.sha256')).read_text())
        self.assert_product_checksums(self.destination, ['cu126'])
        self.assert_checksums(self.environment)

    def test_existing_directory_accepts_new_archive_and_preserves_other_contents(self):
        self.destination.mkdir()
        (self.destination / 'operator-notes.txt').write_text('preserve')
        unpacked = self.destination / 'unpacked-cpu'
        unpacked.mkdir()
        (unpacked / 'manifest.json').write_text('preserve unpacked bundle')
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        self.assertEqual((self.destination / 'operator-notes.txt').read_text(), 'preserve')
        self.assertEqual((unpacked / 'manifest.json').read_text(), 'preserve unpacked bundle')
        self.assert_product_checksums(self.destination, ['cu126'])

    def test_existing_selected_archive_requires_replace_even_in_a_batch(self):
        self.destination.mkdir()
        self.bundle(self.destination, 'cpu')
        before = self.snapshot(self.destination)
        archives = [self.bundle_filename('cu126'), self.bundle_filename('cpu')]
        with self.assertRaisesRegex(RuntimeError, '--replace'):
            delivery.check_destination(self.destination, archives, False)
        with self.assertRaisesRegex(RuntimeError, '--replace'):
            with delivery.stage_release(self.destination, archives, replace=False):
                self.fail('conflicting archive accepted')
        self.assertEqual(self.snapshot(self.destination), before)

    def test_orphan_checksum_does_not_require_replace(self):
        self.destination.mkdir()
        checksum = self.destination / (self.bundle_filename() + '.sha256')
        checksum.write_text('orphan checksum')
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        self.assert_product_checksums(self.destination, ['cu126'])

    def test_replace_updates_only_selected_archive_and_preserves_shared_environment(self):
        self.destination.mkdir()
        self.bundle(self.destination, 'cpu')
        unselected = self.destination / self.bundle_filename('cpu')
        unselected_checksum = unselected.with_name(unselected.name + '.sha256')
        unselected_checksum.write_text(f'{artifacts.sha256(unselected)}  {unselected.name}\n')
        (self.destination / self.bundle_filename()).write_bytes(b'old selected archive')
        (self.destination / (self.bundle_filename() + '.sha256')).write_text('old selected checksum')
        unselected_before = (unselected.read_bytes(), unselected_checksum.read_bytes())
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        environment_before = self.snapshot(self.environment)
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=True) as staged:
            self.bundle(staged)
        self.assertEqual((unselected.read_bytes(), unselected_checksum.read_bytes()), unselected_before)
        self.assertEqual((self.destination / self.bundle_filename()).read_bytes(), b'application-cu126')
        self.assertEqual(environment_before, self.snapshot(self.environment))
        self.assert_product_checksums(self.destination, ['cpu', 'cu126'])

    def test_successful_append_removes_legacy_aggregate_files(self):
        self.destination.mkdir()
        for name in ('README.md', 'CHANGELOG.md', 'release.json', 'SHA256SUMS'):
            (self.destination / name).write_text('obsolete aggregate')
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        self.assert_product_checksums(self.destination, ['cu126'])

    def test_legacy_cpu_release_gains_archive_checksum_when_appending_cuda(self):
        self.destination.mkdir()
        self.bundle(self.destination, 'cpu')
        cpu_archive = self.destination / self.bundle_filename('cpu')
        previous_cpu = cpu_archive.read_bytes()
        for name in ('README.md', 'CHANGELOG.md', 'release.json'):
            (self.destination / name).write_text('obsolete aggregate')
        (self.destination / 'SHA256SUMS').write_text(
            f'{artifacts.sha256(cpu_archive)}  {cpu_archive.name}\n')
        with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
            self.bundle(staged)
        self.assertEqual(cpu_archive.read_bytes(), previous_cpu)
        self.assert_product_checksums(self.destination, ['cpu', 'cu126'])

    def test_corrupt_legacy_unselected_archive_blocks_migration_without_changes(self):
        self.destination.mkdir()
        self.bundle(self.destination, 'cpu')
        cpu_archive = self.destination / self.bundle_filename('cpu')
        (self.destination / 'SHA256SUMS').write_text(
            f'{artifacts.sha256(cpu_archive)}  {cpu_archive.name}\n')
        cpu_archive.write_bytes(b'corrupt legacy CPU archive')
        (self.destination / 'README.md').write_text('legacy readme')
        before = self.snapshot(self.destination)
        with self.assertRaises(RuntimeError):
            with delivery.stage_release(self.destination, [self.bundle_filename()], replace=False) as staged:
                self.bundle(staged)
        self.assertEqual(self.snapshot(self.destination), before)
        self.assertFalse(list(self.root.glob('.' + self.destination.name + '-*')))
        self.assertEqual(self.probe_lock(self.destination), 0)

    def test_later_variant_failure_retains_entire_previous_release(self):
        self.destination.mkdir()
        (self.destination / 'previous').write_text('preserve')
        (self.destination / 'README.md').write_text('legacy metadata survives failed publication')
        self.bundle(self.destination, 'cpu')
        (self.destination / self.bundle_filename('cu118')).write_bytes(b'old selected archive')
        (self.destination / (self.bundle_filename('cu118') + '.sha256')).write_text('old checksum')
        before = self.snapshot(self.destination)
        with self.assertRaisesRegex(RuntimeError, 'second variant'):
            with delivery.stage_release(self.destination,
                                        [self.bundle_filename('cu118'), self.bundle_filename('cu126')],
                                        replace=True) as staged:
                self.bundle(staged, 'cu118')
                raise RuntimeError('second variant failed')
        self.assertEqual(self.snapshot(self.destination), before)
        self.assertFalse(list(self.root.glob('.' + self.destination.name + '-*')))
        self.assertEqual(self.probe_lock(self.destination), 0)

    def test_missing_selected_archive_prevents_incomplete_publication(self):
        self.destination.mkdir()
        self.bundle(self.destination, 'cpu')
        before = self.snapshot(self.destination)
        with self.assertRaises(RuntimeError):
            with delivery.stage_release(self.destination,
                                        [self.bundle_filename('cu118'), self.bundle_filename('cu126')],
                                        replace=False) as staged:
                self.bundle(staged, 'cu118')
        self.assertEqual(self.snapshot(self.destination), before)
        self.assertFalse(list(self.root.glob('.' + self.destination.name + '-*')))
        self.assertEqual(self.probe_lock(self.destination), 0)

    def test_publish_failure_restores_previous_release(self):
        self.destination.mkdir()
        (self.destination / 'previous').write_text('preserve')
        self.bundle(self.destination, 'cpu')
        (self.destination / self.bundle_filename()).write_bytes(b'old selected archive')
        (self.destination / (self.bundle_filename() + '.sha256')).write_text('old checksum')
        before = self.snapshot(self.destination)
        original = Path.replace
        patcher = None
        try:
            with self.assertRaisesRegex(OSError, 'injected rename failure'):
                with delivery.stage_release(self.destination, [self.bundle_filename()], replace=True) as staged:
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
        self.assertEqual(self.snapshot(self.destination), before)
        self.assertEqual(self.probe_lock(self.destination), 0)

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

    def test_runtime_publication_preserves_an_independently_exported_build_role(self):
        archive, record = self.archive()
        build = {**record, 'role': 'build', 'filename': record['filename'].replace('-runtime-', '-build-')}
        build_archive = archive.with_name(build['filename'])
        build_archive.write_bytes(archive.read_bytes())
        delivery.publish_environment(build_archive, build, self.environment)
        delivery.publish_environment(archive, record, self.environment)
        manifest = json.loads((self.environment / 'manifest.json').read_text())
        self.assertEqual(set(manifest['variants']['cu126']), {'build', 'runtime'})
        self.assertTrue((self.environment / build['filename']).is_file())
        self.assert_checksums(self.environment)

    def test_conflicting_environment_identity_is_rejected(self):
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        before = (self.environment / 'manifest.json').read_bytes()
        for key in ('environment_name', 'environment_version'):
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

    def test_build_provenance_does_not_gate_shared_environment_delivery(self):
        archive, record = self.archive()
        delivery.publish_environment(archive, record, self.environment)
        delivery.publish_environment(archive, {**record, 'config_sha256': 'other-source', 'image_id': 'rebuilt'}, self.environment)
        self.assert_checksums(self.environment)

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

    @contextmanager
    def mocked_build(self):
        (self.root / 'VERSION').write_text('1.1.0\n')
        def runtime_archive(role, variant, *args, **kwargs):
            return self.archive(variant)
        def build(*command, **kwargs):
            output_path = next(arg.split('dest=', 1)[1] for arg in command
                               if isinstance(arg, str) and '.output=type=local,dest=' in arg)
            Path(output_path).mkdir(parents=True, exist_ok=True)
        def assemble(args, matrix, variant, runtime, build, record, *rest, **kwargs):
            manifest = self.bundle(args.output_dir, variant)
            manifest['runtime'] = record
            return manifest
        with ExitStack() as stack:
            mocks = {}
            for obj, name, options in [
                (artifacts, 'ROOT', {'new': self.root}),
                (artifacts.dependencies, 'check_generated_artifacts', {}),
                (artifacts.common, 'config', {'return_value': {'environment_version': '202610'}}),
                (artifacts.environment_archives, 'ensure_image', {'return_value': {'Id': 'image'}}),
                (artifacts.environment_archives, 'image_archive', {'side_effect': runtime_archive}),
                (artifacts, 'release_changelog', {'return_value': 'notes'}),
                (artifacts, 'runtime_compatibility', {'return_value': {}}),
                (artifacts, 'source_digest', {'return_value': 'source'}),
                (artifacts, 'image_info', {'return_value': {'Id': 'image'}}),
                (artifacts, 'run', {'side_effect': build}),
                (artifacts, 'assemble', {'side_effect': assemble}),
            ]:
                mocks[name] = stack.enter_context(patch.object(obj, name, **options))
            yield mocks

    def run_build_cli(self, *arguments):
        with patch('sys.argv', ['artifacts.py', 'build', *arguments]):
            artifacts.main()

    def test_cli_cpu_with_environment_then_default_cuda_preserves_both(self):
        output = self.root / '.release/export' / (self.product + '-v1.1.0')
        environment = output.parent / self.environment.name
        defaults = artifacts.common.default_variants(artifacts.dependencies.load_matrix())
        self.assertNotIn('cpu', defaults)
        with self.mocked_build() as mocks:
            self.run_build_cli('cpu', '--with-env')
            cpu_archive = output / self.bundle_filename('cpu')
            cpu_before = (cpu_archive.read_bytes(),
                          cpu_archive.with_name(cpu_archive.name + '.sha256').read_bytes())
            environment_before = self.snapshot(environment)
            self.run_build_cli()
        self.assertEqual((cpu_archive.read_bytes(),
                          cpu_archive.with_name(cpu_archive.name + '.sha256').read_bytes()), cpu_before)
        self.assertEqual([call.args[2] for call in mocks['assemble'].call_args_list], ['cpu', *defaults])
        self.assertEqual(self.snapshot(environment), environment_before)
        self.assertEqual({path.name for path in output.glob('*.tar.gz')},
                         {self.bundle_filename(variant) for variant in ['cpu', *defaults]})
        self.assert_product_checksums(output, ['cpu', *defaults])
        self.assert_checksums(environment)

    def test_cli_default_cuda_then_cpu_appends_without_replace(self):
        output = self.root / '.release/export' / (self.product + '-v1.1.0')
        defaults = artifacts.common.default_variants(artifacts.dependencies.load_matrix())
        with self.mocked_build():
            self.run_build_cli()
            before = self.snapshot(output)
            self.run_build_cli('cpu')
        self.assertTrue(all((output / name).read_bytes() == contents
                            for name, contents in before.items()))
        self.assert_product_checksums(output, [*defaults, 'cpu'])
        self.assertFalse((output.parent / self.environment.name).exists())

    def test_cli_conflicting_later_target_is_rejected_before_docker(self):
        output = self.root / '.release/export' / (self.product + '-v1.1.0')
        output.mkdir(parents=True)
        self.bundle(output, 'cpu')
        before = self.snapshot(output)
        with self.mocked_build() as mocks, self.assertRaisesRegex(RuntimeError, '--replace'):
            self.run_build_cli('cu126', 'cpu')
        mocks['run'].assert_not_called()
        mocks['ensure_image'].assert_not_called()
        mocks['assemble'].assert_not_called()
        self.assertEqual(self.snapshot(output), before)

    def test_cli_replaces_only_selected_product_files_without_touching_environment(self):
        output = self.root / '.release/export' / (self.product + '-v1.1.0')
        output.mkdir(parents=True)
        self.bundle(output, 'cpu')
        (output / self.bundle_filename('cu126')).write_bytes(b'old cu126')
        (output / (self.bundle_filename('cu126') + '.sha256')).write_text('old checksum')
        environment = output.parent / self.environment.name
        environment.mkdir()
        (environment / 'keep').write_bytes(b'shared environment')
        with self.mocked_build():
            self.run_build_cli('cu126', '--replace')
        self.assertEqual((output / self.bundle_filename('cpu')).read_bytes(), b'application-cpu')
        self.assertEqual((output / self.bundle_filename('cu126')).read_bytes(), b'application-cu126')
        self.assertEqual((environment / 'keep').read_bytes(), b'shared environment')
        self.assertEqual([path.name for path in environment.iterdir()], ['keep'])
        self.assert_product_checksums(output, ['cu126'])

    def test_cli_later_build_failure_preserves_selected_and_unselected_archives(self):
        output = self.root / '.release/export' / (self.product + '-v1.1.0')
        output.mkdir(parents=True)
        self.bundle(output, 'cpu')
        (output / self.bundle_filename('cu118')).write_bytes(b'previous cu118 archive')
        (output / (self.bundle_filename('cu118') + '.sha256')).write_text('previous checksum')
        (output / 'README.md').write_text('previous release metadata')
        before = self.snapshot(output)
        with self.mocked_build() as mocks:
            build = mocks['run'].side_effect
            def fail_later(*command, **kwargs):
                if command[-1] == 'cu126':
                    raise RuntimeError('second variant failed')
                return build(*command, **kwargs)
            mocks['run'].side_effect = fail_later
            with self.assertRaisesRegex(RuntimeError, 'second variant'):
                self.run_build_cli('cu118', 'cu126', '--replace')
        self.assertEqual(self.snapshot(output), before)
        self.assertEqual([call.args[2] for call in mocks['assemble'].call_args_list], ['cu118'])
        self.assertFalse(list(output.parent.glob('.' + output.name + '-*')))
        self.assertEqual(self.probe_lock(output), 0)


if __name__ == '__main__':
    unittest.main()
