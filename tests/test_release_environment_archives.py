"""Offline environment reuse must preserve the published image identity."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_artifacts_archive_tests', ROOT / 'scripts/release/artifacts.py')
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)
archives = artifacts.environment_archives


class EnvironmentArchiveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.published = {'build_image_id': 'sha256:' + 'b' * 64, 'runtime_image_id': 'sha256:' + 'c' * 64}
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

    def archive(self, role='build'):
        name = f"rx-quant-env-{role}-{self.matrix['environment_version']}-{self.variant}-linux-amd64.tar.gz"
        path = self.directory / name
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

    def test_runtime_checksum_must_match_published_record(self):
        path, sidecar, record = self.archive('runtime')
        path.write_bytes(b'x' * path.stat().st_size)
        record['sha256'] = archives.common.sha256(path)
        sidecar.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, 'sha256.*发布记录'):
            self.ensure('runtime')
        self.assertFalse(self.loads or self.tags)

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

    def test_old_runtime_option_is_rejected(self):
        with patch('sys.argv', ['artifacts.py', 'build', '--runtime-dir', str(self.directory)]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            artifacts.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.loads or self.tags)


if __name__ == '__main__':
    unittest.main()
