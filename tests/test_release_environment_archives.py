"""Versioned environment reuse is independent of repository image records."""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_artifacts_archive_tests', ROOT / 'scripts/release/artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)
archives = artifacts.environment_archives
ENV_SPEC = importlib.util.spec_from_file_location('runtime_archive_cli_tests', ROOT / 'scripts/environment/runtime_archives.py')
runtime_cli = importlib.util.module_from_spec(ENV_SPEC)
ENV_SPEC.loader.exec_module(runtime_cli)


class EnvironmentArchiveTest(unittest.TestCase):
    def test_archive_names_identify_docker_format_for_every_role_and_variant(self):
        for role in ('runtime', 'build'):
            for variant in archives.common.config()['variants']:
                record = archives.requirement(role, variant)
                self.assertEqual(record['filename'],
                                 f'rx-quant-env-{role}-202610-{variant}-linux-amd64.docker.tar.gz')
                self.assertEqual(record['environment_version'], '202610')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.variant = 'cu126'
        self.config = b'{"architecture":"amd64","os":"linux","rootfs":{"type":"layers","diff_ids":[]}}'
        self.image_id = 'sha256:' + hashlib.sha256(self.config).hexdigest()
        self.images = {}
        self.loads = []
        self.tags = []
        self.loaded_tag = None
        self.patch(archives.subprocess, 'run', side_effect=self.docker)
        self.patch(archives.common, 'inspect', side_effect=self.inspect)

    def patch(self, obj, name, **kwargs):
        patcher = patch.object(obj, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def inspect(self, reference, role, variant):
        image = self.images[reference]
        if image.get('role') != role:
            raise RuntimeError('wrong environment role')
        return image

    def docker(self, command, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect']:
            image = self.images.get(command[3])
            return subprocess.CompletedProcess(command, 0 if image else 1, json.dumps([image]) if image else '',
                                               '' if image else 'Error: No such image: ' + command[3])
        if command[:2] == ['docker', 'load']:
            self.loads.append(command[3])
            role = 'build' if '-build-' in command[3] else 'runtime'
            self.images[self.image_id] = {'Id': self.image_id, 'role': role}
            reference = self.loaded_tag or self.image_id
            self.images[reference] = self.images[self.image_id]
            return subprocess.CompletedProcess(command, 0, 'Loaded image ID: ' + reference + '\n')
        if command[:2] == ['docker', 'tag']:
            self.tags.append(command[3])
            self.images[command[3]] = self.images[command[2]]
            return subprocess.CompletedProcess(command, 0, '')
        self.fail(f'Unexpected Docker operation: {command}')

    def write_payload(self, path):
        config_name = self.image_id[7:] + '.json'
        files = {'manifest.json': json.dumps([{'Config': config_name, 'RepoTags': None, 'Layers': []}]).encode(),
                 config_name: self.config}
        with tarfile.open(path, 'w:gz') as output:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))

    def archive(self, role='runtime', *, sidecar=False, checksum=True):
        record = archives.requirement(role, self.variant)
        path = self.directory / record['filename']
        self.write_payload(path)
        record.update(image_id=self.image_id, sha256=archives.common.sha256(path), size_bytes=path.stat().st_size)
        if sidecar:
            path.with_name(path.name + '.json').write_text(json.dumps(record))
        if checksum:
            archives.write_checksum(path, record)
        return path, record

    def existing_image(self, role='runtime', image_id=None):
        reference = archives.common.image_ref(role, self.variant)
        self.images[reference] = {'Id': image_id or self.image_id, 'role': role}
        return reference

    def test_rebuilt_tag_is_reused_without_records_or_archives(self):
        self.existing_image(image_id='different-local-build')
        actual = archives.ensure_image('runtime', self.variant)
        self.assertEqual(actual['Id'], 'different-local-build')
        self.assertFalse(self.loads or self.tags)
        self.assertFalse(list(self.directory.iterdir()))

    def test_wrong_role_is_rejected_without_importing(self):
        reference = self.existing_image()
        self.images[reference]['role'] = 'build'
        with self.assertRaisesRegex(RuntimeError, 'wrong environment role'):
            archives.ensure_image('runtime', self.variant, self.directory)
        self.assertFalse(self.loads or self.tags)

    def test_archive_without_json_record_is_loaded_and_tagged(self):
        path, _ = self.archive()
        image = archives.ensure_image('runtime', self.variant, self.directory)
        self.assertEqual(image['Id'], self.image_id)
        self.assertEqual(self.loads, [str(path)])
        self.assertEqual(self.tags, [archives.common.image_ref('runtime', self.variant)])
        self.assertFalse(path.with_name(path.name + '.json').exists())

    def test_tagged_archive_does_not_need_retagging(self):
        self.loaded_tag = archives.common.image_ref('runtime', self.variant)
        self.archive()
        archives.ensure_image('runtime', self.variant, self.directory)
        self.assertFalse(self.tags)

    def test_missing_build_archive_has_actionable_errors(self):
        with self.assertRaisesRegex(RuntimeError, '--build-env-dir'):
            archives.ensure_image('build', self.variant)
        with self.assertRaisesRegex(RuntimeError, '缺少 build'):
            archives.ensure_image('build', self.variant, self.directory)
        self.assertFalse(self.loads)

    def test_legacy_identity_and_source_digest_do_not_gate_import(self):
        path, record = self.archive(sidecar=True)
        record.update(image_id='old-local-build', config_sha256='old-repository-layout')
        path.with_name(path.name + '.json').write_text(json.dumps(record))
        actual = archives.ensure_image('runtime', self.variant, self.directory)
        self.assertEqual(actual['Id'], self.image_id)

    def test_checksum_corruption_is_rejected_before_load(self):
        path, _ = self.archive()
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'SHA-256 不匹配'):
            archives.ensure_image('runtime', self.variant, self.directory)
        self.assertFalse(self.loads)

    def test_invalid_checksum_is_rejected(self):
        path, _ = self.archive()
        path.with_name(path.name + '.sha256').write_text('not-a-checksum\n')
        with self.assertRaisesRegex(RuntimeError, '校验文件'):
            archives.ensure_image('runtime', self.variant, self.directory)
        self.assertFalse(self.loads)

    def test_archive_integrity_can_use_sha256sums(self):
        path, record = self.archive(checksum=False)
        (self.directory / 'SHA256SUMS').write_text(record['sha256'] + '  ' + path.name + '\n')
        archives.ensure_image('runtime', self.variant, self.directory)
        self.assertEqual(self.loads, [str(path)])

    def test_plain_docker_save_archive_can_be_loaded(self):
        self.archive(checksum=False)
        archives.ensure_image('runtime', self.variant, self.directory)
        self.assertEqual(len(self.loads), 1)

    def test_invalid_matching_sha256sums_is_rejected(self):
        path, _ = self.archive(checksum=False)
        (self.directory / 'SHA256SUMS').write_text('invalid  ' + path.name + '\n')
        with self.assertRaisesRegex(RuntimeError, '校验文件'):
            archives.ensure_image('runtime', self.variant, self.directory)
        self.assertFalse(self.loads)

    def test_archive_recompression_does_not_bind_application(self):
        path, record = self.archive()
        before = artifacts.runtime_requirement(record)
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()), mtime=1))
        record.update(sha256=archives.common.sha256(path), image_id='another-compatible-build')
        archives.write_checksum(path, record)
        self.assertEqual(before, artifacts.runtime_requirement(record))

    def test_app_only_build_needs_no_archive_or_export(self):
        target = self.directory / 'missing'
        save = self.patch(archives, '_save_runtime')
        path, record = archives.image_archive('runtime', self.variant, target)
        self.assertEqual(record, archives.requirement('runtime', self.variant))
        self.assertEqual(path.parent, target)
        self.assertFalse(target.exists())
        save.assert_not_called()

    def test_export_saves_tag_and_records_actual_id_only_as_provenance(self):
        reference = self.existing_image()
        self.images[reference].update(Created='2027-01-02T03:04:05Z', Config={'Labels': {
            'org.opencontainers.image.revision': 'source-commit',
            archives.common.LABEL + 'environment.config-sha256': 'source-digest'}})
        save = self.patch(archives, '_save_runtime', side_effect=lambda image, path: self.write_payload(path))
        archive, record = archives.image_archive('runtime', self.variant, self.directory, export=True)
        self.assertEqual(save.call_args.args[0], reference)
        self.assertEqual(record['image_id'], self.image_id)
        self.assertEqual(record['sha256'], archives.common.sha256(archive))
        self.assertEqual(record['image_created_at'], '2027-01-02T03:04:05Z')
        self.assertEqual(record['source_revision'], 'source-commit')
        self.assertEqual(record['config_sha256'], 'source-digest')
        self.assertEqual(record['environment_version'], '202610')
        self.assertTrue(archive.with_name(archive.name + '.sha256').is_file())
        self.assertNotIn('image_id', artifacts.runtime_requirement(record))
        _, reused = archives.image_archive('runtime', self.variant, self.directory, export=True)
        self.assertEqual(reused, record)
        self.assertEqual(save.call_count, 1)

    def test_existing_archive_reuse_needs_no_write_or_matching_local_id(self):
        path, record = self.archive(sidecar=True)
        save = self.patch(archives, '_save_runtime')
        with patch.object(Path, 'mkdir', side_effect=AssertionError('read-only archive directory')):
            actual, _ = archives.image_archive('runtime', self.variant, self.directory, export=True)
        self.assertEqual(actual, path)
        save.assert_not_called()

    def test_failed_export_leaves_no_partial_archive(self):
        self.existing_image()
        def fail(image, path):
            path.write_bytes(b'partial')
            raise RuntimeError('injected save failure')
        self.patch(archives, '_save_runtime', side_effect=fail)
        with self.assertRaisesRegex(RuntimeError, 'injected save failure'):
            archives.image_archive('runtime', self.variant, self.directory, export=True)
        self.assertFalse(list(self.directory.glob('rx-quant-env-*.tar.gz*')))
        self.assertFalse(list(self.directory.glob('.environment-export-*')))

    def test_daemon_failure_does_not_attempt_import(self):
        self.patch(archives.subprocess, 'run', return_value=subprocess.CompletedProcess(
            [], 1, '', 'Cannot connect to the Docker daemon'))
        with self.assertRaisesRegex(RuntimeError, 'Docker daemon'):
            archives.ensure_image('runtime', self.variant, self.directory)
        self.assertFalse(self.loads)

    def test_imported_metadata_is_checked_before_retagging(self):
        self.archive()
        self.patch(archives.common, 'inspect', side_effect=RuntimeError('wrong environment role'))
        with self.assertRaisesRegex(RuntimeError, 'wrong environment role'):
            archives.ensure_image('runtime', self.variant, self.directory)
        self.assertEqual(len(self.loads), 1)
        self.assertFalse(self.tags)

    def test_standalone_export_does_not_read_or_write_repository_records(self):
        self.existing_image()
        self.patch(archives, '_save_runtime', side_effect=lambda image, path: self.write_payload(path))
        self.patch(archives.common, 'config_digest', side_effect=AssertionError('not a reuse requirement'))
        runtime_cli.export_runtime(self.variant, self.directory)
        document = json.loads((self.directory / 'manifest.json').read_text())
        self.assertEqual(document['variants'][self.variant]['runtime']['image_tag'],
                         archives.common.image_ref('runtime', self.variant))
        self.assertNotIn('config_sha256', document)

    def test_standalone_export_preserves_both_roles_and_full_sidecar_names(self):
        self.existing_image('build')
        self.existing_image('runtime')
        self.patch(archives, '_save_runtime', side_effect=lambda image, path: self.write_payload(path))
        for role in ('build', 'runtime', 'build'):
            runtime_cli.export_environment(self.variant, self.directory, role)
        document = json.loads((self.directory / 'manifest.json').read_text())
        self.assertEqual(set(document['variants'][self.variant]), {'build', 'runtime'})
        for role, record in document['variants'][self.variant].items():
            archive = self.directory / record['filename']
            self.assertTrue(archive.with_name(archive.name + '.json').is_file())
            self.assertTrue(archive.with_name(archive.name + '.sha256').is_file())
            self.assertIn(archive.name, (self.directory / 'SHA256SUMS').read_text())

    def test_standalone_cli_handles_both_roles_with_runtime_default(self):
        for action, method in (('export', 'export_environment'), ('load', 'load_environment')):
            for options, expected in (([], ['runtime']), (['--role', 'build'], ['build']),
                                      (['--role', 'all'], ['build', 'runtime'])):
                with patch.object(runtime_cli, method) as operation, \
                        patch('sys.argv', ['runtime_archives.py', action, '--directory', str(self.directory),
                                          '--output-dir', str(self.directory), 'cpu', *options]):
                    runtime_cli.main()
                self.assertEqual([call.args[2] for call in operation.call_args_list], expected)

    def test_standalone_load_uses_archive_without_repository_records(self):
        path, _ = self.archive()
        runtime_cli.load_runtime(self.variant, self.directory)
        self.assertEqual(self.loads, [str(path)])

    def test_application_requirement_ignores_ids_and_source_and_archive_hashes(self):
        _, record = self.archive()
        modified = {**record, 'image_id': 'rebuilt', 'config_sha256': 'other-source', 'sha256': 'f' * 64}
        before = artifacts.runtime_requirement(record)
        self.assertEqual(before, artifacts.runtime_requirement(modified))
        self.assertEqual(before['image_tag'], archives.common.image_ref('runtime', self.variant))

    def test_cli_passes_archive_directories_without_identity_record(self):
        (self.directory / 'VERSION').write_text('1.0.0\n')
        ensure = self.patch(archives, 'ensure_image', return_value={'Id': self.image_id})
        self.patch(artifacts, 'ROOT', new=self.directory)
        self.patch(artifacts.dependencies, 'check_generated_artifacts')
        self.patch(artifacts, 'release_changelog', return_value='notes')
        self.patch(artifacts, 'runtime_compatibility', side_effect=RuntimeError('stop before compilation'))
        argv = ['artifacts.py', 'build', self.variant, '--build-env-dir', str(self.directory),
                '--runtime-env-dir', str(self.directory)]
        with patch('sys.argv', argv), self.assertRaisesRegex(RuntimeError, 'stop before compilation'):
            artifacts.main()
        self.assertEqual([call.args for call in ensure.call_args_list],
                         [('runtime', self.variant, self.directory), ('build', self.variant, self.directory)])


class RuntimeArchiveCliTest(unittest.TestCase):
    def test_cpu_runtime_archive_requires_explicit_selection(self):
        matrix = runtime_cli.common.config()
        defaults = [variant for variant in matrix['variants'] if variant != 'cpu']
        selections = [([], defaults), (['cpu'], ['cpu']),
                      (['cu126', '--output-dir', '/tmp/runtime', 'cpu'], ['cu126', 'cpu'])]
        for action in ('export', 'load'):
            for selection, expected in selections:
                with self.subTest(action=action, selection=selection), \
                        patch.object(runtime_cli, 'export_environment') as export, \
                        patch.object(runtime_cli, 'load_environment') as load, \
                        patch('sys.argv', ['runtime_archives.py', action, '--directory', '/tmp/runtime', *selection]):
                    runtime_cli.main()
                    operation = export if action == 'export' else load
                    self.assertEqual([call.args[0] for call in operation.call_args_list], expected)
                    (load if action == 'export' else export).assert_not_called()

    def test_unknown_runtime_archive_variant_is_rejected(self):
        with patch('sys.argv', ['runtime_archives.py', 'export', 'unknown']), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch.object(runtime_cli, 'export_environment') as export:
            with self.assertRaises(SystemExit) as error:
                runtime_cli.main()
        self.assertEqual(error.exception.code, 2)
        export.assert_not_called()


class OciArchiveIdentityTest(unittest.TestCase):
    def fixture(self, path, *, corrupt_layer=False, missing_layer=False, extra_root=False):
        files = {}
        def blob(data, media_type):
            digest = hashlib.sha256(data).hexdigest()
            files['blobs/sha256/' + digest] = data
            return {'mediaType': media_type, 'digest': 'sha256:' + digest, 'size': len(data)}
        layer = blob(b'layer payload', 'application/vnd.oci.image.layer.v1.tar')
        config = blob(b'{"architecture":"amd64","os":"linux"}', 'application/vnd.oci.image.config.v1+json')
        manifest = blob(json.dumps({'schemaVersion': 2, 'config': config, 'layers': [layer]}).encode(),
                        'application/vnd.oci.image.manifest.v1+json')
        root = blob(json.dumps({'schemaVersion': 2, 'manifests': [manifest]}).encode(),
                    'application/vnd.oci.image.index.v1+json')
        files['index.json'] = json.dumps({'schemaVersion': 2, 'manifests': [root] * (2 if extra_root else 1)}).encode()
        files['manifest.json'] = json.dumps([{'Config': 'blobs/sha256/' + config['digest'][7:], 'Layers': []}]).encode()
        if corrupt_layer:
            files['blobs/sha256/' + layer['digest'][7:]] = b'wrong payload'
        if missing_layer:
            del files['blobs/sha256/' + layer['digest'][7:]]
        with tarfile.open(path, 'w:gz') as output:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
        return root['digest']

    def test_docker29_oci_index_identity_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'oci.tar.gz'
            image_id = self.fixture(path)
            archives.verify_runtime_payload(path, image_id)

    def test_invalid_oci_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'oci.tar.gz'
            for corruption in ('corrupt_layer', 'missing_layer', 'extra_root'):
                with self.subTest(corruption=corruption):
                    image_id = self.fixture(path, **{corruption: True})
                    with self.assertRaises(RuntimeError):
                        archives.verify_runtime_payload(path, image_id)


class SavePipelineTest(unittest.TestCase):
    def test_docker_save_and_compression_pipeline(self):
        # Run real gzip/pigz against a fake Docker executable, including save failure.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            docker = directory / 'docker'
            docker.write_text(f'#!{sys.executable}\nimport os,sys\nassert sys.argv[1:] == ["save", "pinned-image"]\nsys.stdout.buffer.write(b"image payload")\nsys.exit(int(os.environ.get("SAVE_EXIT", "0")))\n')
            docker.chmod(0o755)
            with patch.dict(os.environ, {'PATH': str(directory) + os.pathsep + os.environ['PATH'], 'SAVE_EXIT': '0'}):
                output = directory / 'runtime.tar.gz'
                archives._save_runtime('pinned-image', output)
                self.assertEqual(gzip.decompress(output.read_bytes()), b'image payload')
                os.environ['SAVE_EXIT'] = '1'
                with self.assertRaisesRegex(RuntimeError, '导出失败'):
                    archives._save_runtime('pinned-image', directory / 'failed.tar.gz')


if __name__ == '__main__':
    unittest.main()
