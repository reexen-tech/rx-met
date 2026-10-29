"""Regression tests for the staged, versioned onnxsim shutdown fix."""
from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


from tests.support.paths import REPO_ROOT as ROOT


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = _load('onnxsim_wheel_patch', ROOT / 'scripts/environment/patch_onnxsim_wheel.py')
common = _load('onnxsim_patch_environment', ROOT / 'scripts/environment/common.py')


class OnnxsimWheelPatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = b'class PyModelExecutor: pass\n' + patcher.BEFORE + b'\n'
        for name, data in [('SOURCE_SHA256', self.source),
                           ('PATCHED_SHA256', self.source.replace(patcher.BEFORE, patcher.AFTER))]:
            replacement = patch.object(patcher, name, hashlib.sha256(data).hexdigest())
            replacement.start()
            self.addCleanup(replacement.stop)

    def make_wheel(self, version='0.7.0', source=None):
        path = self.directory / f'onnxsim-{version}-cp312-abi3-manylinux_2_27_x86_64.whl'
        self.prefix = f'onnxsim-{version}.dist-info'
        self.contents = {
            patcher.SOURCE: self.source if source is None else source,
            'onnxsim/onnxsim_cpp2py_export.abi3.so': b'\x7fELF-native-extension-must-stay-identical\x00',
            'onnxsim/__init__.py': b'from .onnx_simplifier import simplify\n',
            self.prefix + '/METADATA': f'Metadata-Version: 2.1\nName: onnxsim\nVersion: {version}\n'.encode(),
            self.prefix + '/WHEEL': b'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: cp312-abi3-manylinux_2_27_x86_64\n',
            self.prefix + '/RECORD': b'',
        }
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in self.contents.items():
                archive.writestr(name, data)
        return path

    def test_patch_changes_only_annotation_wheel_marker_and_record(self):
        original = self.make_wheel()
        result = patcher.patch_wheel(original)
        self.assertFalse(original.exists())
        self.assertEqual(list(self.directory.glob('*.whl')), [result])
        self.assertIn('-0.7.0-1rxmet-', result.name)
        with zipfile.ZipFile(result) as archive:
            self.assertEqual(archive.namelist(), list(self.contents))
            for name, data in self.contents.items():
                actual = archive.read(name)
                if name == patcher.SOURCE:
                    self.assertEqual(actual, data.replace(patcher.BEFORE, patcher.AFTER))
                elif name == self.prefix + '/WHEEL':
                    self.assertEqual(actual, b'Build: 1rxmet\n' + data)
                elif name != self.prefix + '/RECORD':
                    self.assertEqual(actual, data)
            rows = list(csv.reader(io.StringIO(archive.read(self.prefix + '/RECORD').decode())))
            self.assertEqual({row[0] for row in rows}, set(self.contents))
            for name, digest, size in rows:
                if name == self.prefix + '/RECORD':
                    self.assertEqual((digest, size), ('', ''))
                else:
                    data = archive.read(name)
                    expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=')
                    self.assertEqual(digest, 'sha256=' + expected.decode())
                    self.assertEqual(int(size), len(data))

    def test_repeated_patch_is_byte_identical_and_leaves_only_fixed_wheel(self):
        fixed = patcher.patch_wheel(self.make_wheel())
        before = fixed.read_bytes()
        self.assertEqual(patcher.patch_wheel(fixed), fixed)
        self.assertEqual(fixed.read_bytes(), before)
        self.assertEqual(list(self.directory.iterdir()), [fixed])

    def test_unknown_source_or_version_is_rejected_without_writes(self):
        for arguments, error in [({'source': b'unknown upstream code'}, 'unreviewed'),
                                 ({'version': '0.7.1'}, 'only supports onnxsim==0.7.0')]:
            with self.subTest(arguments=arguments):
                path = self.make_wheel(**arguments)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, error):
                    patcher.patch_wheel(path)
                self.assertEqual(path.read_bytes(), before)
                path.unlink()

    def test_conflicting_target_cannot_overwrite_another_wheel(self):
        path = self.make_wheel()
        target = path.with_name(path.name.replace('-0.7.0-', '-0.7.0-1rxmet-'))
        target.write_bytes(b'existing artifact')
        with self.assertRaisesRegex(ValueError, 'target already exists'):
            patcher.patch_wheel(path)
        self.assertEqual(target.read_bytes(), b'existing artifact')
        self.assertTrue(path.exists())

    def test_source_fix_is_an_environment_build_input(self):
        self.assertIn('scripts/environment/patch_onnxsim_wheel.py', common.BUILD_FILES)

    def test_download_patches_before_verification_and_final_checksums(self):
        original = self.make_wheel()
        verifier = _load('onnxsim_patch_verifier', ROOT / 'scripts/internal/verify_dependency_wheelhouse.py')
        patch_calls = []
        events = []
        original_patch = patcher.patch_wheel

        def apply(path):
            events.append('patch')
            result = original_patch(path)
            patch_calls.append(result)
            return result

        def verify(*args):
            events.append('verify')
            self.assertEqual(len(patch_calls), 1)
            self.assertFalse(original.exists())

        def module_from_spec(spec):
            if spec.name == 'wheelhouse':
                return verifier
            return patcher

        # Keep the real module code loaded above; only redirect download's dynamic imports.
        with patch.object(common, 'config', return_value={
                'common_packages': {'onnxsim': '0.7.0'}, 'build_packages': {},
                'variants': {'test': {'torch': {'packages': {}, 'index_url': ''},
                                      'onnxruntime': {'packages': {}, 'index_url': ''}}}}), \
                patch.object(common.importlib.util, 'module_from_spec', side_effect=module_from_spec), \
                patch.object(common.importlib.util, 'spec_from_file_location') as spec_from, \
                patch.object(patcher, 'patch_wheel', side_effect=apply), \
                patch.object(verifier, 'verify', side_effect=verify), \
                patch.object(common, 'run', side_effect=AssertionError('cached wheel needs no network')):
            spec_from.side_effect = lambda name, path: SimpleNamespace(name=name, loader=SimpleNamespace(exec_module=lambda module: None))
            common.download('test', self.directory)
        self.assertEqual(events, ['patch', 'verify'])
        result = patch_calls[0]
        self.assertEqual((self.directory / 'SHA256SUMS').read_text(),
                         f'{hashlib.sha256(result.read_bytes()).hexdigest()}  {result.name}\n')


class OnnxsimShutdownVerificationTest(unittest.TestCase):
    def test_clean_exit_relays_all_output_and_uses_bounded_cpu_container(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        result = SimpleNamespace(returncode=0, stdout='folded correctly\n', stderr='ordinary warning\n')
        with patch.object(common.subprocess, 'run', return_value=result) as run, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            common.verify_onnxsim_shutdown('image-id')
        self.assertEqual(stdout.getvalue(), result.stdout)
        self.assertEqual(stderr.getvalue(), result.stderr)
        command = run.call_args.args[0]
        self.assertIn('--read-only', command)
        self.assertIn('--network', command)
        self.assertNotIn('--gpus', command)
        self.assertEqual(command[command.index('--cpus') + 1], '1')
        self.assertTrue(run.call_args.kwargs['capture_output'])
        self.assertEqual(run.call_args.kwargs['timeout'], 90)
        self.assertIn('signal.alarm(60)', run.call_args.kwargs['input'])
        self.assertIn('import torch\n    import onnxsim', run.call_args.kwargs['input'])

    def test_zero_exit_with_nanobind_leak_is_failure_and_stderr_is_preserved(self):
        warning = 'nanobind: leaked 1 instances!\n - leaked type "ModelExecutor"\n'
        stderr = io.StringIO()
        with patch.object(common.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout='', stderr=warning)), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, 'shutdown verification failed'):
                common.verify_onnxsim_shutdown('image-id')
        self.assertEqual(stderr.getvalue(), warning)

    def test_nonzero_exit_fails_even_without_leak_warning(self):
        with patch.object(common.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stdout='', stderr='')), \
                self.assertRaisesRegex(RuntimeError, 'exit code 1'):
            common.verify_onnxsim_shutdown('image-id')

    def test_timeout_preserves_partial_stderr_and_removes_only_its_container(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        expired = subprocess.TimeoutExpired('docker', 90, output=b'partial stdout\n', stderr=b'partial stderr\n')
        with patch.object(common.subprocess, 'run', side_effect=[
                expired, SimpleNamespace(returncode=0, stdout='', stderr='')]) as run, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, 'timed out; container removed'):
                common.verify_onnxsim_shutdown('image-id')
        self.assertEqual(stdout.getvalue(), 'partial stdout\n')
        self.assertEqual(stderr.getvalue(), 'partial stderr\n')
        initial, cleanup = run.call_args_list
        name = initial.args[0][initial.args[0].index('--name') + 1]
        self.assertTrue(name.startswith('rx-met-onnxsim-verify-'))
        self.assertEqual(cleanup.args[0], ['docker', 'rm', '-f', name])
        self.assertEqual(cleanup.kwargs['timeout'], 10)

    def test_timeout_reports_when_container_cleanup_cannot_be_confirmed(self):
        with patch.object(common.subprocess, 'run', side_effect=[
                subprocess.TimeoutExpired('docker', 90), subprocess.TimeoutExpired('docker rm', 10)]):
            with self.assertRaisesRegex(RuntimeError, 'cleanup not confirmed: rx-met-onnxsim-verify-'):
                common.verify_onnxsim_shutdown('image-id')

    def test_public_environment_verification_runs_shutdown_check_for_both_roles(self):
        with patch.object(common, 'inspect', return_value={'Id': 'image-id'}), \
                patch.object(common, 'run'), patch.object(common, 'verify_fx_source_cache'), \
                patch.object(common, 'verify_onnxsim_shutdown') as shutdown:
            common.verify('cu130')
        self.assertEqual(shutdown.call_count, 2)


if __name__ == '__main__':
    unittest.main()
