"""Regression tests for reviewed, streaming Torch source-cache wheel fixes."""
from __future__ import annotations

import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile


PATCHER_PATH = Path(__file__).with_name('patch_torch_wheel.py')
if not PATCHER_PATH.exists():
    PATCHER_PATH = Path(__file__).resolve().parents[1] / 'scripts/environment/patch_torch_wheel.py'
spec = importlib.util.spec_from_file_location('torch_wheel_patch', PATCHER_PATH)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class TorchWheelPatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = b'''import linecache
class _EvalCacheLoader:
    def cache(self, src, globals, co_fields=None):
        return '<eval_with_key>.0'
_loader = _EvalCacheLoader()
def _exec_with_source(src, globals, co_fields=None):
    key = _loader.cache(src, globals, co_fields)
    exec(compile(src, key, "exec"), globals)
def untouched_function():
    return 42
'''
        self.helper = b'class OwnedEvalCacheLoader: pass\n'
        self.helper_path = self.directory / 'helper.py'
        self.helper_path.write_bytes(self.helper)
        reviewed = (hashlib.sha256(self.source).hexdigest(),
                    hashlib.sha256(patcher.patch_source(self.source)).hexdigest())
        for name, value in [('REVIEWED_SOURCES', {version: reviewed for version in
                             ('2.10.0', '2.10.0+cu130', '2.8.0', '2.8.0+cu126')}),
                            ('HELPER_PATH', self.helper_path),
                            ('HELPER_SHA256', hashlib.sha256(self.helper).hexdigest())]:
            replacement = patch.object(patcher, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def make_wheel(self, version='2.10.0+cu130', source=None, extra=None):
        path = self.directory / f'torch-{version}-cp312-cp312-manylinux_2_28_x86_64.whl'
        self.prefix = f'torch-{version}.dist-info'
        self.native_name = 'torch/lib/libtorch_cuda.so'
        self.contents = {
            patcher.SOURCE: self.source if source is None else source,
            self.native_name: b'\x7fELF' + b'native-member\x00' * 35,
            'torch/__init__.py': b'__version__ = "test"\n',
            self.prefix + '/METADATA': f'Metadata-Version: 2.1\nName: torch\nVersion: {version}\n'.encode(),
            self.prefix + '/WHEEL': b'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-manylinux_2_28_x86_64\n',
        }
        self.contents.update(extra or {})
        output = io.StringIO(newline='')
        writer = csv.writer(output, lineterminator='\n')
        for name, data in self.contents.items():
            writer.writerow((name, *patcher._record_values(data)))
        writer.writerow((self.prefix + '/RECORD', '', ''))
        self.contents[self.prefix + '/RECORD'] = output.getvalue().encode()
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.comment = b'preserve archive comment'
            archive.writestr('torch/lib/', b'')
            for name, data in self.contents.items():
                archive.writestr(name, data)
        return path

    def rewrite_member(self, path, name, data):
        with zipfile.ZipFile(path) as original:
            members = {entry.filename: original.read(entry) for entry in original.infolist()}
        members[name] = data
        with zipfile.ZipFile(path, 'w') as archive:
            for member, value in members.items():
                archive.writestr(member, value)

    def test_only_reviewed_python_source_helper_and_wheel_metadata_change(self):
        original = self.make_wheel()
        fixed = patcher.patch_wheel(original)
        self.assertFalse(original.exists())
        self.assertIn('-2.10.0+cu130-1rxmet-', fixed.name)
        with zipfile.ZipFile(fixed) as archive:
            self.assertEqual(archive.comment, b'preserve archive comment')
            self.assertEqual(set(archive.namelist()), {'torch/lib/', patcher.HELPER, *self.contents})
            for name, data in self.contents.items():
                if name == patcher.SOURCE:
                    self.assertEqual(archive.read(name), patcher.patch_source(data))
                elif name == self.prefix + '/WHEEL':
                    self.assertEqual(archive.read(name), b'Build: 1rxmet\n' + data)
                elif name != self.prefix + '/RECORD':
                    self.assertEqual(archive.read(name), data)
            self.assertEqual(archive.read(patcher.HELPER), self.helper)
            rows = list(csv.reader(io.StringIO(archive.read(self.prefix + '/RECORD').decode())))
            self.assertEqual({row[0] for row in rows}, {*self.contents, patcher.HELPER})
            for name, digest, size in rows:
                if name == self.prefix + '/RECORD':
                    self.assertEqual((digest, size), ('', ''))
                else:
                    self.assertEqual((digest, size), patcher._record_values(archive.read(name)))

    def test_source_patch_preserves_unrelated_source_and_compiles(self):
        patched = patcher.patch_source(self.source)
        self.assertIn(b'from ._rx_source_cache import OwnedEvalCacheLoader as _EvalCacheLoader\n', patched)
        self.assertIn(b'_loader = _EvalCacheLoader()\n', patched)
        self.assertIn(b'_loader.exec_with_source(src, globals, co_fields)\n', patched)
        self.assertIn(b'def untouched_function():\n    return 42\n', patched)
        self.assertNotIn(b'key = _loader.cache', patched)
        compile(patched, 'graph_module.py', 'exec')

    def test_all_four_reviewed_metadata_versions_are_supported(self):
        for version in patcher.REVIEWED_SOURCES:
            with self.subTest(version=version):
                result = patcher.patch_wheel(self.make_wheel(version))
                self.assertIn(f'-{version}-1rxmet-', result.name)

    def test_unsupported_torch_versions_remain_byte_identical_and_reported(self):
        for version in ('2.9.1+cpu', '2.7.1+cu118', '2.10.0+cu128', '2.11.0'):
            with self.subTest(version=version):
                original = self.make_wheel(version, source=b'unreviewed source')
                before = original.read_bytes()
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(patcher.patch_wheel(original, helper_path=self.directory / 'absent'), original)
                self.assertIn('not covered by reviewed sources', stderr.getvalue())
                self.assertIn(version, stderr.getvalue())
                self.assertEqual(original.read_bytes(), before)

    def test_repeated_patch_is_byte_identical(self):
        fixed = patcher.patch_wheel(self.make_wheel())
        before = fixed.read_bytes()
        self.assertEqual(patcher.patch_wheel(fixed), fixed)
        self.assertEqual(fixed.read_bytes(), before)
        self.assertEqual(list(self.directory.glob('*.whl')), [fixed])

    def test_unknown_source_for_supported_version_fails_without_writes(self):
        original = self.make_wheel(source=b'unreviewed source')
        before = original.read_bytes()
        with self.assertRaisesRegex(ValueError, 'unreviewed torch/fx/graph_module.py SHA256'):
            patcher.patch_wheel(original)
        self.assertEqual(original.read_bytes(), before)

    def test_unknown_helper_fails_without_writes(self):
        original = self.make_wheel()
        self.helper_path.write_bytes(b'changed helper')
        before = original.read_bytes()
        with self.assertRaisesRegex(ValueError, 'unreviewed source-cache helper'):
            patcher.patch_wheel(original)
        self.assertEqual(original.read_bytes(), before)

    def test_conflicting_target_is_preserved(self):
        original = self.make_wheel()
        target = original.with_name(original.name.replace('-cp312-', '-1rxmet-cp312-', 1))
        target.write_bytes(b'existing wheel')
        with self.assertRaisesRegex(ValueError, 'target already exists'):
            patcher.patch_wheel(original)
        self.assertEqual(target.read_bytes(), b'existing wheel')
        self.assertTrue(original.exists())

    def test_concurrent_target_creation_is_not_overwritten(self):
        original = self.make_wheel()
        before = original.read_bytes()

        def link(source, target):
            target.write_bytes(b'racing artifact')
            raise FileExistsError('target appeared')

        with patch.object(patcher.os, 'link', side_effect=link), \
                self.assertRaisesRegex(ValueError, 'target already exists'):
            patcher.patch_wheel(original)
        self.assertEqual(original.read_bytes(), before)
        targets = [path for path in self.directory.glob('*.whl') if path != original]
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].read_bytes(), b'racing artifact')
        self.assertEqual(list(self.directory.glob('.torch-fx-patch-*')), [])

    def test_signed_wheels_are_rejected(self):
        for suffix in ('.jws', '.p7s'):
            with self.subTest(suffix=suffix):
                original = self.make_wheel(extra={
                    'torch-2.10.0+cu130.dist-info/RECORD' + suffix: b'signature'})
                before = original.read_bytes()
                with self.assertRaisesRegex(ValueError, 'signed wheel'):
                    patcher.patch_wheel(original)
                self.assertEqual(original.read_bytes(), before)

    def test_duplicate_members_are_rejected(self):
        original = self.make_wheel()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(original, 'a') as archive:
                archive.writestr(patcher.SOURCE, self.source)
        with self.assertRaisesRegex(ValueError, 'duplicate wheel members'):
            patcher.patch_wheel(original)

    def test_preexisting_helper_is_rejected(self):
        original = self.make_wheel(extra={patcher.HELPER: self.helper})
        with self.assertRaisesRegex(ValueError, 'helper already exists'):
            patcher.patch_wheel(original)

    def test_inconsistent_source_and_build_marker_are_rejected(self):
        original = self.make_wheel()
        renamed = original.with_name(original.name.replace('-cp312-', '-1rxmet-cp312-', 1))
        original.rename(renamed)
        with self.assertRaisesRegex(ValueError, 'source and build marker disagree'):
            patcher.patch_wheel(renamed)

    def test_patched_helper_tampering_is_rejected_on_repeat(self):
        fixed = patcher.patch_wheel(self.make_wheel())
        self.rewrite_member(fixed, patcher.HELPER, b'changed helper')
        with self.assertRaisesRegex(ValueError, 'helper disagrees'):
            patcher.patch_wheel(fixed)

    def test_patched_record_tampering_is_rejected_on_repeat(self):
        fixed = patcher.patch_wheel(self.make_wheel())
        record_name = self.prefix + '/RECORD'
        with zipfile.ZipFile(fixed) as archive:
            data = archive.read(record_name)
        self.rewrite_member(fixed, record_name, data.replace(b'sha256=', b'sha512=', 1))
        with self.assertRaisesRegex(ValueError, 'patched RECORD disagrees'):
            patcher.patch_wheel(fixed)

    def test_invalid_record_rows_are_rejected(self):
        original = self.make_wheel()
        self.rewrite_member(original, self.prefix + '/RECORD', b'only-one-column\n')
        with self.assertRaisesRegex(ValueError, 'invalid or duplicate RECORD'):
            patcher.patch_wheel(original)

    def test_missing_record_members_are_rejected(self):
        original = self.make_wheel()
        self.rewrite_member(original, self.prefix + '/RECORD',
                            f'{self.prefix}/RECORD,,\n'.encode())
        with self.assertRaisesRegex(ValueError, 'RECORD members disagree'):
            patcher.patch_wheel(original)

    def test_native_members_are_streamed_with_bounded_reads(self):
        payload = b'\x7fELF' + b'a' * (3 * patcher.COPY_CHUNK_SIZE + 17)
        original = self.make_wheel(extra={'torch/lib/libtorch_cuda.so': payload})
        original_read = zipfile.ZipFile.read
        stream_read = zipfile.ZipExtFile.read
        sizes = []

        def read(archive, name, *args, **kwargs):
            self.assertNotEqual(getattr(name, 'filename', name), self.native_name)
            return original_read(archive, name, *args, **kwargs)

        def bounded(stream, size=-1):
            if stream.name == self.native_name:
                sizes.append(size)
                self.assertGreater(size, 0)
                self.assertLessEqual(size, patcher.COPY_CHUNK_SIZE)
            return stream_read(stream, size)

        with patch.object(zipfile.ZipFile, 'read', new=read), \
                patch.object(zipfile.ZipExtFile, 'read', new=bounded):
            fixed = patcher.patch_wheel(original)
        self.assertGreaterEqual(len(sizes), 4)
        with zipfile.ZipFile(fixed) as archive:
            self.assertEqual(hashlib.sha256(archive.read(self.native_name)).digest(),
                             hashlib.sha256(payload).digest())

    def test_copy_failure_preserves_input_and_removes_temporary_output(self):
        original = self.make_wheel()
        before = original.read_bytes()
        with patch.object(patcher.shutil, 'copyfileobj', side_effect=OSError('copy failed')), \
                self.assertRaisesRegex(OSError, 'copy failed'):
            patcher.patch_wheel(original)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(list(self.directory.glob('*.whl')), [original])
        self.assertEqual(list(self.directory.glob('.torch-fx-patch-*')), [])

    def test_download_patches_before_dependency_verification_and_final_checksums(self):
        root = Path(__file__).resolve().parents[1]

        def load(name, path):
            specification = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module

        common = load('torch_patch_environment', root / 'scripts/environment/common.py')
        verifier = load('torch_patch_verifier', root / 'scripts/internal/verify_dependency_wheelhouse.py')
        original = self.make_wheel()
        actual_patch = patcher.patch_wheel
        events, patched = [], []

        def apply(path):
            events.append('patch')
            fixed = actual_patch(path)
            patched.append(fixed)
            return fixed

        def verify(*args):
            events.append('verify')
            self.assertEqual(len(patched), 1)
            self.assertFalse(original.exists())

        modules = {'wheelhouse': verifier, 'torch_wheel_patch': patcher,
                   'onnxsim_wheel_patch': SimpleNamespace(patch_wheel=lambda path: self.fail('unexpected onnxsim wheel'))}
        matrix = {'common_packages': {}, 'build_packages': {}, 'variants': {'test': {
            'torch': {'packages': {'torch': '2.10.0+cu130'}, 'index_url': ''},
            'onnxruntime': {'packages': {}, 'index_url': ''}}}}
        with patch.object(common, 'config', return_value=matrix), \
                patch.object(common.importlib.util, 'module_from_spec', side_effect=lambda spec: modules[spec.name]), \
                patch.object(common.importlib.util, 'spec_from_file_location') as spec_from, \
                patch.object(patcher, 'patch_wheel', side_effect=apply), \
                patch.object(verifier, 'verify', side_effect=verify), \
                patch.object(common, 'run', side_effect=AssertionError('cached wheel needs no network')):
            spec_from.side_effect = lambda name, path: SimpleNamespace(
                name=name, loader=SimpleNamespace(exec_module=lambda module: None))
            common.download('test', self.directory)
        self.assertEqual(events, ['patch', 'verify'])
        fixed = patched[0]
        self.assertEqual((self.directory / 'SHA256SUMS').read_text(),
                         f'{hashlib.sha256(fixed.read_bytes()).hexdigest()}  {fixed.name}\n')


if __name__ == '__main__':
    unittest.main()
