"""Offline environment reuse must preserve the published image identity."""
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
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = b'{"architecture":"amd64","os":"linux","rootfs":{"type":"layers","diff_ids":[]}}'
        self.published = {'build_image_id': 'sha256:' + 'b' * 64,
                          'runtime_image_id': 'sha256:' + hashlib.sha256(self.config).hexdigest()}
        self.images = {}
        self.loads = []
        self.tags = []
        self.variant = 'cu126'
        self.matrix = archives.common.config()
        self.build_ref = archives.common.image_ref('build', self.variant)
        self.loaded_role = 'build'
        self.install_patch(archives.subprocess, 'run', side_effect=self.docker)
        self.install_patch(archives.common, 'inspect', side_effect=lambda ref, *args: self.images[ref])

    def install_patch(self, obj, name, **kwargs):
        patcher = patch.object(obj, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def docker(self, command, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect']:
            info = self.images.get(command[3])
            return subprocess.CompletedProcess(command, 0 if info else 1,
                                               json.dumps([info]) if info else '[]',
                                               '' if info else 'Error: No such image: ' + command[3])
        if command[:2] == ['docker', 'load']:
            self.loads.append(command[3])
            image_id = self.published[self.loaded_role + '_image_id']
            self.images[image_id] = {'Id': image_id}
        elif command[:2] == ['docker', 'tag']:
            self.tags.append(command[3])
            self.images[command[3]] = self.images[command[2]]
        else:
            self.fail(f'Unexpected Docker operation: {command}')
        return subprocess.CompletedProcess(command, 0)

    def write_payload(self, path, *, config=None, image_count=1, config_path=None):
        name = config_path or (hashlib.sha256(self.config).hexdigest() + '.json')
        manifest = [{'Config': name, 'RepoTags': None, 'Layers': []}] * image_count
        with tarfile.open(path, 'w:gz') as output:
            for filename, data in [('manifest.json', json.dumps(manifest).encode()),
                                   (name, self.config if config is None else config)]:
                member = tarfile.TarInfo(filename)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))

    def update_sidecar(self, path, sidecar, record):
        record = {**record, 'sha256': archives.common.sha256(path), 'size_bytes': path.stat().st_size}
        sidecar.write_text(json.dumps(record))
        return record

    def archive(self, role='build'):
        name = f"rx-quant-env-{role}-{self.matrix['environment_version']}-{self.variant}-linux-amd64.tar.gz"
        path = self.directory / name
        if role == 'runtime':
            self.write_payload(path)
        else:
            path.write_bytes(b'a verified docker archive fixture')
        record = {'schema_version': 1, 'environment_name': 'rx-quant-env',
                  'environment_version': self.matrix['environment_version'], 'variant': self.variant,
                  'config_sha256': archives.common.config_digest(), 'image_id': self.published[role + '_image_id'],
                  'filename': name, 'size_bytes': path.stat().st_size, 'sha256': archives.common.sha256(path)}
        if role == 'build':
            record['role'] = role
        else:
            self.published['runtime'] = dict(record)
        sidecar = path.with_name(name + '.json')
        sidecar.write_text(json.dumps(record))
        return path, sidecar, record

    def ensure(self, role='build', directory=True):
        return archives.ensure_image(role, self.variant, self.published, self.directory if directory else None)

    def test_existing_image_does_not_require_archive_files(self):
        self.images[self.build_ref] = {'Id': self.published['build_image_id']}
        self.assertEqual(self.ensure()['Id'], self.published['build_image_id'])
        self.assertFalse(self.loads or self.tags)

    def test_existing_wrong_tag_is_not_overwritten(self):
        self.images[self.build_ref] = {'Id': 'wrong-image'}
        self.archive()
        with self.assertRaisesRegex(RuntimeError, '同名镜像'):
            self.ensure()
        self.assertFalse(self.loads or self.tags)

    def test_untagged_published_image_is_reused_without_loading(self):
        image_id = self.published['build_image_id']
        self.images[image_id] = {'Id': image_id}
        self.ensure(directory=False)
        self.assertEqual(self.tags, [self.build_ref])
        self.assertFalse(self.loads)

    def test_missing_build_and_runtime_are_loaded_and_tagged(self):
        for role in ('build', 'runtime'):
            with self.subTest(role=role):
                self.loaded_role = role
                path, _, _ = self.archive(role)
                self.assertEqual(self.ensure(role)['Id'], self.published[role + '_image_id'])
                self.assertIn(str(path), self.loads)
                self.assertIn(archives.common.image_ref(role, self.variant), self.tags)

    def test_missing_archive_or_directory_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, '--build-env-dir'):
            self.ensure(directory=False)
        with self.assertRaisesRegex(RuntimeError, '缺少 build 镜像归档'):
            self.ensure()
        self.assertFalse(self.loads or self.tags)

    def test_corrupted_archive_is_rejected_before_load(self):
        path, _, _ = self.archive()
        path.write_bytes(b'x' * path.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, 'SHA-256 不匹配'):
            self.ensure()
        self.assertFalse(self.loads or self.tags)

    def test_wrong_identity_and_size_are_rejected_before_load(self):
        _, sidecar, valid = self.archive()
        for key, value in [('image_id', 'wrong'), ('config_sha256', 'wrong'), ('variant', 'cu118'),
                           ('role', 'runtime'), ('environment_version', 'v2'), ('filename', '../archive'),
                           ('size_bytes', 0), ('sha256', 'invalid')]:
            with self.subTest(field=key):
                sidecar.write_text(json.dumps({**valid, key: value}))
                with self.assertRaises(RuntimeError):
                    self.ensure()
                self.assertFalse(self.loads or self.tags)

    def test_recompressed_runtime_is_verified_and_loaded(self):
        path, sidecar, original = self.archive('runtime')
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()), mtime=1))
        record = self.update_sidecar(path, sidecar, original)
        self.assertNotEqual(record['sha256'], original['sha256'])
        self.loaded_role = 'runtime'
        self.assertEqual(self.ensure('runtime')['Id'], self.published['runtime_image_id'])
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.published['runtime'], original)

    def test_reexport_with_false_identity_is_rejected_even_with_matching_file_checksum(self):
        path, sidecar, original = self.archive('runtime')
        self.write_payload(path, config=b'{"different":"image"}')
        self.update_sidecar(path, sidecar, original)
        with self.assertRaisesRegex(RuntimeError, '镜像身份不匹配'):
            self.ensure('runtime')
        self.assertFalse(self.loads or self.tags)

    def test_classic_archive_with_wrong_layer_content_is_rejected(self):
        config = json.dumps({'rootfs': {'diff_ids': ['sha256:' + hashlib.sha256(b'expected layer').hexdigest()]}}).encode()
        image_id = 'sha256:' + hashlib.sha256(config).hexdigest()
        path = self.directory / 'wrong-layer.tar.gz'
        config_name = image_id[7:] + '.json'
        with tarfile.open(path, 'w:gz') as output:
            files = {'manifest.json': json.dumps([{'Config': config_name, 'Layers': ['layer.tar']}]).encode(),
                     config_name: config, 'layer.tar': b'changed layer'}
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
        with self.assertRaisesRegex(RuntimeError, '镜像层不匹配'):
            archives.verify_runtime_payload(path, image_id)

    def test_reexport_cannot_include_other_images(self):
        path, sidecar, original = self.archive('runtime')
        self.write_payload(path, image_count=2)
        self.update_sidecar(path, sidecar, original)
        with self.assertRaisesRegex(RuntimeError, '仅能包含'):
            self.ensure('runtime')
        self.assertFalse(self.loads or self.tags)

    def test_oci_config_path_is_supported(self):
        path, sidecar, original = self.archive('runtime')
        self.write_payload(path, config_path='blobs/sha256/' + self.published['runtime_image_id'][7:])
        self.update_sidecar(path, sidecar, original)
        self.loaded_role = 'runtime'
        self.ensure('runtime')
        self.assertEqual(len(self.loads), 1)

    def test_missing_archive_is_exported_once_without_changing_environment_record(self):
        _, _, original = self.archive('runtime')
        target = self.directory / 'fresh-cache'
        image_id = self.published['runtime_image_id']
        self.images[image_id] = {'Id': image_id}
        save = self.install_patch(archives, '_save_runtime', side_effect=lambda image, path: self.write_payload(path))
        path, record = archives.runtime_archive(self.variant, self.published, target, export=True)
        self.assertTrue(path.is_file())
        self.assertEqual(record['sha256'], archives.common.sha256(path))
        self.assertEqual(record['size_bytes'], path.stat().st_size)
        self.assertEqual(record['image_id'], image_id)
        self.assertEqual(archives.runtime_archive(self.variant, self.published, target, export=True), (path, record))
        self.assertEqual(archives.runtime_archive(self.variant, self.published, target), (path, record))
        save.assert_called_once()
        self.assertEqual(self.published['runtime'], original)

    def test_app_only_without_archive_does_not_export_or_create_directory(self):
        self.archive('runtime')
        target = self.directory / 'absent-cache'
        save = self.install_patch(archives, '_save_runtime')
        _, record = archives.runtime_archive(self.variant, self.published, target)
        self.assertEqual(record, self.published['runtime'])
        save.assert_not_called()
        self.assertFalse(target.exists())

    def test_valid_archive_reuse_requires_no_directory_write(self):
        path, _, record = self.archive('runtime')
        save = self.install_patch(archives, '_save_runtime')
        with patch.object(Path, 'mkdir', side_effect=AssertionError('read-only archive source')):
            self.assertEqual(archives.runtime_archive(self.variant, self.published, self.directory, export=True), (path, record))
        save.assert_not_called()

    def test_corrupt_or_incomplete_archives_are_not_overwritten(self):
        path, sidecar, _ = self.archive('runtime')
        save = self.install_patch(archives, '_save_runtime')
        path.write_bytes(b'damaged')
        with self.assertRaises(RuntimeError):
            archives.runtime_archive(self.variant, self.published, self.directory, export=True)
        self.assertEqual(path.read_bytes(), b'damaged')
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, '缺少 runtime'):
            archives.runtime_archive(self.variant, self.published, self.directory, export=True)
        self.assertTrue(sidecar.exists())
        save.assert_not_called()

    def test_export_failure_leaves_no_partial_archive_and_can_be_retried(self):
        self.archive('runtime')
        target = self.directory / 'failed-export'
        image_id = self.published['runtime_image_id']
        self.images[image_id] = {'Id': image_id}
        def fail(image, path):
            path.write_bytes(b'partial')
            raise RuntimeError('injected save failure')
        save = self.install_patch(archives, '_save_runtime', side_effect=fail)
        with self.assertRaisesRegex(RuntimeError, 'injected save failure'):
            archives.runtime_archive(self.variant, self.published, target, export=True)
        self.assertFalse(list(target.glob('rx-quant-env-*.tar.gz*')))
        self.assertFalse(list(target.glob('.runtime-export-*')))
        save.side_effect = lambda image, path: self.write_payload(path)
        path, _ = archives.runtime_archive(self.variant, self.published, target, export=True)
        self.assertTrue(path.is_file())

    def test_docker_connection_failure_does_not_attempt_import(self):
        self.install_patch(archives.subprocess, 'run', return_value=subprocess.CompletedProcess(
            [], 1, '', 'Cannot connect to the Docker daemon'))
        with self.assertRaisesRegex(RuntimeError, 'Docker daemon'):
            self.ensure()
        self.assertFalse(self.loads or self.tags)

    def test_imported_identity_and_labels_are_checked_before_tagging(self):
        self.archive()
        self.install_patch(archives.common, 'inspect', side_effect=RuntimeError('wrong imported identity'))
        with self.assertRaisesRegex(RuntimeError, 'wrong imported identity'):
            self.ensure()
        self.assertEqual(len(self.loads), 1)
        self.assertFalse(self.tags)

    def test_cli_passes_both_archive_directories_to_environment_reuse(self):
        self.archive('runtime')
        (self.directory / 'VERSION').write_text('1.0.0\n')
        argv = ['artifacts.py', 'build', 'cu126', '--build-env-dir', str(self.directory),
                '--runtime-env-dir', str(self.directory)]
        ensure = self.install_patch(archives, 'ensure_image', side_effect=lambda role, *args: {'Id': self.published[role + '_image_id']})
        self.install_patch(artifacts, 'ROOT', new=self.directory)
        self.install_patch(artifacts.dependencies, 'load_matrix', return_value=self.matrix)
        self.install_patch(artifacts.dependencies, 'check_generated_artifacts')
        self.install_patch(artifacts, 'release_changelog', return_value='notes')
        self.install_patch(artifacts.common, 'load_record', return_value=self.published)
        self.install_patch(artifacts, 'runtime_compatibility', side_effect=RuntimeError('stop before compilation'))
        with patch('sys.argv', argv), self.assertRaisesRegex(RuntimeError, 'stop before compilation'):
            artifacts.main()
        self.assertEqual([call.args for call in ensure.call_args_list], [
            ('runtime', 'cu126', self.published, self.directory),
            ('build', 'cu126', self.published, self.directory)])

    def test_cli_with_env_exports_to_missing_directory_and_passes_actual_record_to_bundle(self):
        _, _, original = self.archive('runtime')
        (self.directory / 'VERSION').write_text('1.0.0\n')
        runtime_dir = self.directory / 'new-runtime-cache'
        output_dir = self.directory / 'delivery'
        image_id = self.published['runtime_image_id']
        self.images[image_id] = {'Id': image_id}
        def save(image, path):
            self.write_payload(path, config_path='blobs/sha256/' + image_id[7:])
        self.install_patch(archives, '_save_runtime', side_effect=save)
        self.install_patch(archives, 'ensure_image', side_effect=lambda role, *args: {'Id': self.published[role + '_image_id']})
        self.install_patch(artifacts, 'ROOT', new=self.directory)
        self.install_patch(artifacts.dependencies, 'load_matrix', return_value=self.matrix)
        self.install_patch(artifacts.dependencies, 'check_generated_artifacts')
        self.install_patch(artifacts, 'release_changelog', return_value='notes')
        self.install_patch(artifacts.common, 'load_record', return_value=self.published)
        self.install_patch(artifacts, 'runtime_compatibility', return_value={})
        self.install_patch(artifacts, 'source_digest', return_value='source')
        self.install_patch(artifacts, 'image_info', return_value={'Id': self.published['build_image_id']})
        def compile_artifacts(*command, **kwargs):
            output = next(value.split('dest=', 1)[1] for value in command if isinstance(value, str) and '.output=type=local,dest=' in value)
            Path(output).mkdir(parents=True)
        self.install_patch(artifacts, 'run', side_effect=compile_artifacts)
        def assemble_bundle(args, matrix, variant, runtime, build, record, archive, *rest, **kwargs):
            manifest = {'bundle_name': 'rx-met-v1.0.0-' + variant + '-linux-amd64',
                        'variant': variant, 'runtime': artifacts.runtime_requirement(record)}
            (args.output_dir / (manifest['bundle_name'] + '.tar.gz')).write_bytes(b'application fixture')
            return manifest
        assemble = self.install_patch(artifacts, 'assemble', side_effect=assemble_bundle)
        argv = ['artifacts.py', 'build', 'cu126', '--with-env', '--runtime-env-dir', str(runtime_dir),
                '--output-dir', str(output_dir)]
        with patch('sys.argv', argv):
            artifacts.main()
        environment = output_dir.parent / 'rx-quant-env-v1'
        archive = environment / original['filename']
        actual = json.loads(archive.with_name(archive.name + '.json').read_text())
        self.assertEqual(actual['sha256'], archives.common.sha256(archive))
        self.assertNotEqual(actual['sha256'], original['sha256'])
        self.assertEqual(assemble.call_args.args[5], actual)
        self.assertEqual(self.published['runtime'], original)
        self.assertIn(actual['sha256'], (environment / 'SHA256SUMS').read_text())
        self.assertNotIn(actual['sha256'], (output_dir / 'SHA256SUMS').read_text())
        self.assertFalse((output_dir / 'images').exists())
        release = json.loads((output_dir / 'release.json').read_text())
        self.assertTrue(release['bundles'][0]['environment_included'])
        self.assertEqual(release['bundles'][0]['environment_directory'], '../rx-quant-env-v1')

    def test_application_requirement_is_independent_of_archive_bytes(self):
        _, _, original = self.archive('runtime')
        changed = {**original, 'sha256': 'f' * 64, 'size_bytes': original['size_bytes'] + 100,
                   'repo_digests': ['another-host-copy']}
        url = 'https://example.invalid/' + original['filename']
        requirement = artifacts.runtime_requirement(original, url)
        self.assertEqual(requirement, artifacts.runtime_requirement(changed, url))
        self.assertNotIn('sha256', requirement)
        self.assertNotIn('size_bytes', requirement)
        self.assertEqual(requirement['image_id'], self.published['runtime_image_id'])
        self.assertEqual(requirement['checksum_url'], url + '.sha256')

    def test_standalone_export_accepts_recompression_without_rewriting_publication(self):
        path, sidecar, original = self.archive('runtime')
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()), mtime=1))
        actual = self.update_sidecar(path, sidecar, original)
        repository = self.directory / 'repository'
        publication = repository / 'docker/environment-images.json'
        publication.parent.mkdir(parents=True)
        before = json.dumps({'variants': {self.variant: self.published}})
        publication.write_text(before)
        digest = archives.common.config_digest()
        self.install_patch(archives.common, 'ROOT', new=repository)
        self.install_patch(archives.common, 'config', return_value=self.matrix)
        self.install_patch(archives.common, 'config_digest', return_value=digest)
        self.install_patch(archives.common, 'load_record', return_value=self.published)
        initial_export = self.install_patch(archives.common, 'export')
        runtime_cli.export_runtime(self.variant, self.directory)
        initial_export.assert_not_called()
        self.assertEqual(publication.read_text(), before)
        emitted = json.loads((self.directory / 'manifest.json').read_text())['variants'][self.variant]['runtime']
        self.assertEqual(emitted['sha256'], actual['sha256'])
        self.assertIn(actual['sha256'], path.with_name(path.name + '.sha256').read_text())

    def test_standalone_load_accepts_recompressed_runtime(self):
        path, sidecar, original = self.archive('runtime')
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()), mtime=1))
        self.update_sidecar(path, sidecar, original)
        self.loaded_role = 'runtime'
        self.install_patch(archives.common, 'load_record', return_value=self.published)
        runtime_cli.load_runtime(self.variant, self.directory)
        self.assertEqual(self.loads, [str(path)])
        self.assertIn(archives.common.image_ref('runtime', self.variant), self.tags)

    def test_removed_with_runtime_option_is_rejected(self):
        with patch('sys.argv', ['artifacts.py', 'build', '--with-runtime']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            artifacts.main()
        self.assertEqual(raised.exception.code, 2)

    def test_old_runtime_option_is_rejected(self):
        with patch('sys.argv', ['artifacts.py', 'build', '--runtime-dir', str(self.directory)]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            artifacts.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.loads or self.tags)


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
