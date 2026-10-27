"""Automatic execution layout and logging, without Torch or a native runtime."""
import contextlib
import datetime
import errno
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from unittest.mock import patch

from aimet_common import run_context as runtime


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        runtime.configure_output()
        self.addCleanup(runtime.configure_output)
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-execution-')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(runtime._finish_implicit)
        self.root = Path(temporary.name)
        patcher = patch.dict(os.environ, {'RX_MET_OUTPUT_DIR': str(self.root / 'automatic')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def read_record(self, run):
        return json.loads((run.root / 'run.json').read_text())

    def test_encoding_versions_follow_document_structure_for_json_and_raw_outputs(self):
        compiler = {'version': '1.0.0', 'schema_version': 4,
                    'activation_encodings': {}, 'param_encodings': {}}
        raw = {'version': '0.6.1', 'activation_encodings': {}, 'param_encodings': {}}
        ordinary = {'version': 'application-1', 'schema_version': 99, 'model': 'demo'}
        with runtime.execution(self.root / 'encoding-documents') as run:
            for name, document in (('model.json', compiler), ('model.encodings', raw),
                                   ('model.meta.json', ordinary), ('empty.encodings', ordinary)):
                (run.exports / name).write_text(json.dumps(document))
        outputs = self.read_record(run)['outputs']
        self.assertEqual(outputs['exports/model.json']['schema_version'], 4)
        self.assertEqual(outputs['exports/model.json']['version'], '1.0.0')
        self.assertEqual(outputs['exports/model.encodings']['version'], '0.6.1')
        self.assertNotIn('schema_version', outputs['exports/model.encodings'])
        for name in ('model.meta.json', 'empty.encodings'):
            self.assertNotIn('version', outputs['exports/' + name])
            self.assertNotIn('schema_version', outputs['exports/' + name])

    def test_output_configuration_is_validated_and_snapshotted_per_task(self):
        for value in (None, 0, 1, 'False'):
            with self.subTest(value=value), self.assertRaises(TypeError):
                runtime.configure_output(timestamp_output=value)
        with runtime.execution(self.root / 'timestamped') as timestamped:
            runtime.configure_output(timestamp_output=False)
            self.assertEqual(timestamped.root.parent, self.root / 'timestamped')
            self.assertTrue(timestamped.timestamp_output)
        with runtime.execution(self.root / 'fixed') as fixed:
            runtime.configure_output()
            self.assertEqual(fixed.root, self.root / 'fixed')
            self.assertFalse(fixed.timestamp_output)
        self.assertEqual(self.read_record(fixed)['output_config'], {'timestamp_output': False})
        with runtime.execution(self.root / 'restored') as restored:
            self.assertTrue(restored.timestamp_output)

    def test_fixed_output_replaces_same_named_files_and_preserves_inputs(self):
        runtime.configure_output(timestamp_output=False)
        for name in ('models/source.onnx', 'datasets/calib/sample.npy', 'exports/user.txt', 'notes.txt'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('user data')
        with runtime.execution(self.root) as first:
            runtime.log_print('first execution only')
            first.snapshot_config({'bits': 8})
            (first.exports / 'model.onnx').write_bytes(b'first')
            (first.exports / 'old.encodings').write_text('{}')
            (first.checkpoints / 'model.pth').write_bytes(b'first checkpoint')
        with runtime.execution(self.root) as second:
            runtime.log_print('second execution only')
            second.snapshot_config({'bits': 16})
            (second.exports / 'model.onnx').write_bytes(b'second')
            (second.checkpoints / 'model.pth').write_bytes(b'second checkpoint')
        self.assertEqual(first.root, second.root)
        self.assertEqual(second.root, self.root)
        self.assertNotEqual(first.execution_id, second.execution_id)
        self.assertEqual((self.root / 'exports/model.onnx').read_bytes(), b'second')
        self.assertEqual((self.root / 'checkpoints/model.pth').read_bytes(), b'second checkpoint')
        log = (self.root / 'run.log').read_text()
        self.assertIn('second execution only', log)
        self.assertNotIn('first execution only', log)
        self.assertEqual(json.loads((self.root / 'config/quantization.json').read_text()), {'bits': 16})
        record = self.read_record(second)
        self.assertEqual(record['execution_id'], second.execution_id)
        self.assertEqual(set(record['outputs']), {'exports/model.onnx', 'checkpoints/model.pth'})
        for name in ('models/source.onnx', 'datasets/calib/sample.npy', 'exports/user.txt', 'notes.txt'):
            self.assertEqual((self.root / name).read_text(), 'user data')
        self.assertFalse((self.root / first.execution_id).exists())
        self.assertFalse((self.root / second.execution_id).exists())

    def test_fixed_models_keep_early_logs_separate_and_migrate_without_truncation(self):
        runtime.configure_output(timestamp_output=False)

        class Sim:
            @runtime.logged_operation
            def __init__(self, name):
                self.name = name
                runtime.log_print('prepare ' + name)

        @runtime.logged_export('output_dir', task='export')
        def export(sim, output_dir, filename_prefix):
            runtime.log_print('export ' + sim.name)
            path = Path(output_dir) / (filename_prefix + '.onnx')
            path.write_text(sim.name)
            return path

        first = Sim('first')
        first_stage = runtime.get_run(first).root
        second = Sim('second')
        second_stage = runtime.get_run(second).root
        self.assertNotEqual(first_stage, second_stage)
        target = self.root / 'exported'
        target.mkdir(mode=0o750)
        permissions = target.stat().st_mode
        first_path = export(first, target, 'model')
        first_log = (target / 'run.log').read_text()
        self.assertIn('prepare first', first_log)
        self.assertIn('export first', first_log)
        self.assertNotIn('prepare second', first_log)
        second_path = export(second, target, 'model')
        second_log = (target / 'run.log').read_text()
        self.assertIn('prepare second', second_log)
        self.assertIn('export second', second_log)
        self.assertNotIn('prepare first', second_log)
        self.assertEqual(first_path, target / 'exports/model.onnx')
        self.assertEqual(second_path, first_path)
        self.assertEqual(second_path.read_text(), 'second')
        self.assertFalse(first_stage.exists())
        self.assertFalse(second_stage.exists())
        self.assertEqual(target.stat().st_mode, permissions)

    def test_fixed_implicit_run_finishes_in_default_output_root(self):
        runtime.configure_output(timestamp_output=False)

        @runtime.logged_operation
        def prepare():
            runtime.log_print('library diagnostic without export')

        prepare()
        run = runtime.get_run()
        stage = run.root
        runtime._finish_implicit()
        self.assertEqual(run.root, self.root / 'automatic')
        self.assertFalse(stage.exists())
        self.assertIn('library diagnostic without export', (run.root / 'run.log').read_text())
        self.assertEqual(self.read_record(run)['status'], 'success')

    def test_fixed_work_cleanup_preserves_unrelated_and_failed_files(self):
        runtime.configure_output(timestamp_output=False)
        user = self.root / '_work/user.txt'
        user.parent.mkdir()
        user.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'failed task'):
            with runtime.execution(self.root) as failed:
                diagnostic = failed.work / 'graph.onnx'
                diagnostic.write_bytes(b'diagnostic')
                raise ValueError('failed task')
        self.assertEqual(self.read_record(failed)['status'], 'failed')
        with runtime.execution(self.root) as success:
            work = success.work
            (work / 'new-intermediate').write_bytes(b'temporary')
        self.assertFalse(work.exists())
        self.assertEqual(user.read_text(), 'preserve')
        self.assertEqual(diagnostic.read_bytes(), b'diagnostic')

    def test_fixed_directory_is_locked_across_processes_before_log_truncation(self):
        runtime.configure_output(timestamp_output=False)
        script = '''import sys
from aimet_common.run_context import configure_output, execution
configure_output(timestamp_output=False)
try:
    with execution(sys.argv[1]):
        raise AssertionError('overlapping task entered output directory')
except RuntimeError:
    print('directory-busy')
'''
        with runtime.execution(self.root) as run:
            runtime.log_print('lock owner')
            original = (run.root / 'run.log').read_bytes()
            result = subprocess.run([sys.executable, '-c', script, str(self.root)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('directory-busy', result.stdout)
            self.assertEqual((run.root / 'run.log').read_bytes(), original)
        with runtime.execution(self.root) as replacement:
            self.assertEqual(replacement.root, self.root)

    def test_fixed_metadata_is_recorded_and_stale_tasks_cannot_rewrite_record(self):
        runtime.configure_output(timestamp_output=False)
        with runtime.export_execution(self.root) as first:
            (first.exports / 'model.onnx').write_bytes(b'first')
        with runtime.metadata_output(self.root) as directory:
            metadata = directory / 'model.meta.json'
            metadata.write_text('{}')
            runtime.note_output(metadata)
        self.assertIn('exports/model.meta.json', self.read_record(first)['outputs'])
        with runtime.execution(self.root) as second:
            (second.exports / 'model.onnx').write_bytes(b'second')
        original = (self.root / 'run.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, '后续任务覆盖'):
            first.save()
        self.assertEqual((self.root / 'run.json').read_bytes(), original)

    def test_fixed_directory_lock_is_not_retained_by_forked_workers(self):
        runtime.configure_output(timestamp_output=False)
        context = multiprocessing.get_context('fork')
        ready, stop = context.Event(), context.Event()

        def worker():
            ready.set()
            stop.wait(10)

        child = context.Process(target=worker)
        try:
            with runtime.execution(self.root):
                child.start()
                self.assertTrue(ready.wait(10))
            with runtime.execution(self.root) as replacement:
                self.assertFalse(replacement.timestamp_output)
        finally:
            stop.set()
            if child.pid is not None:
                child.join(5)
                if child.is_alive():
                    child.terminate()
                    child.join(5)

    def test_historical_metadata_does_not_modify_new_fixed_run_record(self):
        with runtime.execution(self.root) as historical:
            (historical.exports / 'model.onnx').write_bytes(b'historical')
        runtime.configure_output(timestamp_output=False)
        with runtime.execution(self.root) as current:
            (current.exports / 'model.onnx').write_bytes(b'current')
        original = (current.root / 'run.json').read_bytes()
        with runtime.metadata_output(historical.exports) as directory:
            metadata = directory / 'model.meta.json'
            metadata.write_text('{}')
            runtime.note_output(metadata)
        self.assertEqual((current.root / 'run.json').read_bytes(), original)
        self.assertEqual((historical.exports / 'model.onnx').read_bytes(), b'historical')

    def test_fixed_directory_rejects_symlinks_before_overwriting_files(self):
        runtime.configure_output(timestamp_output=False)
        source = self.root / 'outside.txt'
        source.write_text('preserve')
        (self.root / 'run.log').symlink_to(source)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            with runtime.execution(self.root):
                self.fail('must reject symlink')
        self.assertEqual(source.read_text(), 'preserve')
        (self.root / 'run.log').unlink()
        exports = self.root / 'exports'
        exports.mkdir()
        (exports / 'model.onnx').symlink_to(source)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            with runtime.execution(self.root):
                self.fail('must reject nested symlink')
        self.assertEqual(source.read_text(), 'preserve')
        (exports / 'model.onnx').unlink()
        with runtime.execution(self.root):
            pass

    def test_timestamp_collision_preserves_previous_execution(self):
        beijing = datetime.timezone(datetime.timedelta(hours=8))
        now = datetime.datetime(2026, 10, 21, 14, 30, 25, 123456, beijing)
        with patch.object(runtime, '_now', return_value=now):
            with runtime.execution(self.root) as first:
                (first.exports / 'model.onnx').write_bytes(b'first')
            with runtime.execution(self.root) as second:
                (second.exports / 'model.onnx').write_bytes(b'second')
        self.assertEqual(first.execution_id, '20261021_143025_123456')
        self.assertEqual(second.execution_id, first.execution_id + '-1')
        self.assertEqual((first.root / 'exports/model.onnx').read_bytes(), b'first')
        self.assertFalse((self.root / 'runs').exists())
        self.assertEqual(set(self.root.iterdir()), {first.root, second.root})

    def test_beijing_timestamps_do_not_follow_host_timezone(self):
        script = '''import sys, time
from aimet_common.run_context import execution
time.tzset()
assert time.localtime(0).tm_hour == 0
with execution(sys.argv[1]):
    pass
'''
        before = datetime.datetime.now(datetime.timezone.utc)
        result = subprocess.run([sys.executable, '-c', script, str(self.root)],
                                env={**os.environ, 'TZ': 'UTC'},
                                capture_output=True, text=True, timeout=20)
        after = datetime.datetime.now(datetime.timezone.utc)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = list(self.root.glob('*/run.json'))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())
        self.assertRegex(record['execution_id'], r'^\d{8}_\d{6}_\d{6}$')
        started = datetime.datetime.fromisoformat(record['started_at'])
        finished = datetime.datetime.fromisoformat(record['finished_at'])
        for key, value in (('started_at', started), ('finished_at', finished)):
            self.assertRegex(record[key], r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+08:00$')
            self.assertEqual(value.utcoffset(), datetime.timedelta(hours=8))
        self.assertEqual(record['execution_id'], started.strftime('%Y%m%d_%H%M%S_%f'))
        self.assertEqual(records[0].parent.name, record['execution_id'])
        self.assertLessEqual(before, started)
        self.assertLessEqual(started, finished)
        self.assertLessEqual(finished, after)

    def test_json_timestamps_keep_zero_microseconds(self):
        beijing = datetime.timezone(datetime.timedelta(hours=8))
        now = datetime.datetime(2026, 10, 27, 10, 53, 8, tzinfo=beijing)
        with patch.object(runtime, '_now', return_value=now):
            with runtime.execution(self.root) as run:
                pass
        record = self.read_record(run)
        self.assertEqual(run.root.name, '20261027_105308_000000')
        self.assertEqual(record['started_at'], '2026-10-27T10:53:08.000000+08:00')
        self.assertEqual(record['finished_at'], record['started_at'])

    def test_log_timestamps_convert_utc_epoch_to_beijing_time(self):
        utc = datetime.datetime(2026, 10, 26, 18, 53, 8, 820401, datetime.timezone.utc)
        message = logging.LogRecord('execution-time-test', logging.INFO, '', 0,
                                    'known UTC instant', (), None)
        message.created = utc.timestamp()
        message.msecs = utc.microsecond / 1000
        with runtime.execution(self.root) as run:
            run.emit(message)
        line = next(line for line in (run.root / 'run.log').read_text().splitlines()
                    if 'known UTC instant' in line)
        timestamp = datetime.datetime.fromisoformat(line.split()[0])
        self.assertEqual(timestamp.utcoffset(), datetime.timedelta(hours=8))
        self.assertEqual(timestamp.replace(microsecond=0).isoformat(), '2026-10-27T02:53:08+08:00')

    def test_source_version_does_not_use_unrelated_installed_bundle(self):
        checkout = self.root / 'checkout'
        (checkout / '.git').mkdir(parents=True)
        bundle = self.root / 'installed'
        bundle.mkdir()
        (bundle / 'manifest.json').write_text(json.dumps({
            'bundle_name': 'old-installed-bundle', 'source_revision': 'old-revision',
        }))
        source_module = checkout / 'src/aimet_common/run_context.py'
        with patch.object(runtime, '__file__', str(source_module)), \
                patch.dict(os.environ, {'RX_MET_HOME': str(bundle)}), \
                patch.object(runtime.subprocess, 'check_output', side_effect=['current-revision\n', b' M source.py\n']):
            identity = runtime._tool_identity()
        self.assertEqual(identity['build'], {'source_revision': 'current-revision', 'dirty': True})
        self.assertEqual(identity['python_module'], str(source_module))

    def test_installed_version_uses_bundle_manifest(self):
        bundle = self.root / 'installed'
        bundle.mkdir()
        (bundle / 'manifest.json').write_text(json.dumps({
            'bundle_name': 'installed-bundle', 'source_revision': 'bundle-revision',
            'variant': 'cu126', 'runtime': {'image': 'runtime:v1'},
        }))
        module = bundle / 'python/aimet_common/run_context.py'
        with patch.object(runtime, '__file__', str(module)):
            identity = runtime._tool_identity()
        self.assertEqual(identity['build']['source_revision'], 'bundle-revision')
        self.assertEqual(identity['build']['variant'], 'cu126')
        self.assertEqual(identity['runtime']['image'], 'runtime:v1')

    def test_verbose_only_filters_console_and_does_not_duplicate_file_messages(self):
        console = io.StringIO()
        with contextlib.redirect_stdout(console), runtime.execution(self.root) as run:
            with runtime.log_details(False):
                runtime.log_print('file-only detail')
            with runtime.log_details(True):
                runtime.log_print('console detail')
            print('ordinary output')
        text = (run.root / 'run.log').read_text()
        self.assertNotIn('file-only detail', console.getvalue())
        self.assertIn('console detail', console.getvalue())
        self.assertIn('ordinary output', console.getvalue())
        for message in ('file-only detail', 'console detail', 'ordinary output'):
            self.assertEqual(text.count(message), 1)

    def test_nested_export_reuses_workflow_and_snapshots_changed_config(self):
        config = self.root / 'quantization.json'
        config.write_text('{"bitwidth": 8}')
        with runtime.execution(self.root / 'out') as run:
            run.snapshot_config(config, 'quantization')
            config.write_text('{"bitwidth": 16}')
            run.snapshot_config(config, 'quantization')
            with runtime.export_execution(self.root / 'out') as nested:
                self.assertIs(run, nested)
                (nested.exports / 'model.encodings').write_text('{}')
                self.assertEqual(runtime.metadata_directory(nested.exports), nested.exports)
        self.assertEqual(len(self.read_record(run)['config']), 2)
        self.assertEqual(json.loads((run.root / 'config/quantization.json').read_text()), {'bitwidth': 8})
        for identity in self.read_record(run)['config'].values():
            self.assertEqual(identity['sha256'], runtime._sha256(run.root / identity['path']))
        self.assertTrue(self.read_record(run)['tool']['versions']['rx-met'])

    def test_failure_retains_work_and_preserves_previous_success(self):
        with runtime.execution(self.root) as good:
            (good.exports / 'model.onnx').write_bytes(b'valid')
        original_record = (good.root / 'run.json').read_bytes()
        error = ValueError('calibration failed')
        with self.assertRaises(ValueError) as raised:
            with runtime.execution(self.root) as failed:
                (failed.work / 'graph.onnx').write_bytes(b'diagnostic')
                raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(self.read_record(failed)['status'], 'failed')
        self.assertEqual((failed.root / '_work/graph.onnx').read_bytes(), b'diagnostic')
        self.assertIn('ValueError: calibration failed', (failed.root / 'run.log').read_text())
        self.assertEqual((good.root / 'run.json').read_bytes(), original_record)
        self.assertEqual((good.exports / 'model.onnx').read_bytes(), b'valid')

    def test_metadata_after_failed_export_belongs_to_failed_attempt(self):
        with runtime.export_execution(self.root) as good:
            (good.exports / 'model.onnx').write_bytes(b'valid')
            (good.exports / 'model.meta.json').write_text('{"status": "success"}')
        original_record = (good.root / 'run.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'failed export'):
            with runtime.export_execution(self.root) as failed:
                raise ValueError('failed export')
        metadata = runtime.metadata_directory(self.root) / 'model.meta.json'
        metadata.write_text('{"status": "failed"}')
        runtime.note_output(metadata)
        self.assertEqual(metadata.parent, failed.exports)
        self.assertEqual((good.root / 'run.json').read_bytes(), original_record)
        self.assertEqual(json.loads((good.exports / 'model.meta.json').read_text()), {'status': 'success'})
        self.assertIn('exports/model.meta.json', self.read_record(failed)['outputs'])
        self.assertEqual(self.read_record(failed)['status'], 'failed')

    def test_concurrent_workflows_restore_streams_and_isolate_logs(self):
        first_active = threading.Event()
        second_active = threading.Event()
        first_done = threading.Event()
        runs, errors = {}, []

        def first():
            try:
                with runtime.execution(self.root) as run:
                    runs['first'] = run
                    print('first-thread-before')
                    first_active.set()
                    if not second_active.wait(10):
                        raise TimeoutError('second workflow did not start')
                    warnings.warn('first-thread-warning', UserWarning)
                    print('first-thread-after', end='')
            except BaseException as error:
                errors.append(error)
            finally:
                first_done.set()

        def second():
            try:
                if not first_active.wait(10):
                    raise TimeoutError('first workflow did not start')
                with runtime.execution(self.root) as run:
                    runs['second'] = run
                    second_active.set()
                    print('second-thread-before')
                    if not first_done.wait(10):
                        raise TimeoutError('first workflow did not finish')
                    warnings.warn('second-thread-warning', UserWarning)
                    print('second-thread-after', end='')
            except BaseException as error:
                errors.append(error)

        stdout, stderr, showwarning = sys.stdout, sys.stderr, warnings.showwarning
        threads = [threading.Thread(target=first), threading.Thread(target=second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(sys.stdout, stdout)
        self.assertIs(sys.stderr, stderr)
        self.assertIs(warnings.showwarning, showwarning)
        for name, run in runs.items():
            text = (run.root / 'run.log').read_text()
            for suffix in ('before', 'after'):
                self.assertEqual(text.count(f'{name}-thread-{suffix}'), 1)
            self.assertEqual(text.count(f'UserWarning: {name}-thread-warning'), 1)
            other = 'second' if name == 'first' else 'first'
            self.assertNotIn(f'{other}-thread-', text)

    def test_verbose_false_also_hides_raw_diagnostics_from_console(self):
        console = io.StringIO()
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console), runtime.execution(self.root) as run:
            with runtime.log_details(False):
                print('raw file-only diagnostic')
                warnings.warn('file-only warning', UserWarning)
        self.assertNotIn('raw file-only diagnostic', console.getvalue())
        self.assertNotIn('file-only warning', console.getvalue())
        self.assertIn('raw file-only diagnostic', (run.root / 'run.log').read_text())
        self.assertIn('UserWarning: file-only warning', (run.root / 'run.log').read_text())

    def test_capture_restores_streams_when_console_flush_fails(self):
        class BrokenFlush(io.StringIO):
            broken = False

            def flush(self):
                if self.broken:
                    raise OSError('console flush failed')
                super().flush()

        console = BrokenFlush()
        with contextlib.redirect_stdout(console):
            stdout, stderr, showwarning = sys.stdout, sys.stderr, warnings.showwarning
            with self.assertRaisesRegex(OSError, 'console flush failed'):
                with runtime.execution(self.root) as run:
                    print('before failed flush')
                    console.broken = True
            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)
            self.assertIs(warnings.showwarning, showwarning)
        self.assertEqual(self.read_record(run)['status'], 'failed')

    def test_success_cleans_work_and_records_checkpoints(self):
        with runtime.execution(self.root) as run:
            (run.work / 'intermediate').write_bytes(b'temporary')
            (run.checkpoints / 'model.pth').write_bytes(b'checkpoint')
            (run.exports / 'model.onnx').write_bytes(b'exported')
        self.assertFalse((run.root / '_work').exists())
        self.assertFalse((run.root / 'artifacts').exists())
        record = self.read_record(run)
        self.assertEqual(record['status'], 'success')
        self.assertEqual(record['outputs']['checkpoints/model.pth']['sha256'],
                         runtime._sha256(run.root / 'checkpoints/model.pth'))

    def test_library_calls_start_automatically_and_move_logs_to_export_root(self):
        @runtime.logged_operation
        def prepare(config_file, verbose=False):
            with runtime.log_details(verbose):
                runtime.log_print('early diagnostic')
            return 'unchanged result'

        config = self.root / 'config.json'
        config.write_text('{}')
        self.assertEqual(prepare(config), 'unchanged result')
        automatic = runtime.get_run().root
        target = self.root / 'user root'
        with runtime.export_execution(target) as run:
            (run.exports / 'model.onnx').write_bytes(b'model')
        self.assertFalse(automatic.exists())
        self.assertEqual(run.output_root, target)
        self.assertIn('early diagnostic', (run.root / 'run.log').read_text())
        self.assertEqual(runtime.metadata_directory(target), run.exports)
        with runtime.export_execution(target) as second:
            (second.exports / 'model.onnx').write_bytes(b'new model')
        self.assertNotEqual(run.root, second.root)

    def test_aimet_area_bridge_preserves_console_level(self):
        console = io.StringIO()
        logger = logging.getLogger('execution-test-area')
        handler = logging.StreamHandler(console)
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            runtime.attach_aimet_logger(logger)
            runtime.attach_aimet_logger(logger)
            with runtime.execution(self.root) as run:
                logger.debug('file debug')
                logger.info('console info')
            self.assertNotIn('file debug', console.getvalue())
            self.assertEqual(console.getvalue().count('console info'), 1)
            text = (run.root / 'run.log').read_text()
            self.assertEqual(text.count('file debug'), 1)
            self.assertEqual(text.count('console info'), 1)
        finally:
            logger.handlers = []

    def test_logger_configuration_does_not_lose_open_run(self):
        import logging.config
        for timestamp_output in (True, False):
            runtime.configure_output(timestamp_output=timestamp_output)
            with self.subTest(timestamp_output=timestamp_output), runtime.execution(self.root) as run:
                runtime.log_print('before logger configuration')
                logging.config.dictConfig({'version': 1, 'disable_existing_loggers': False})
                runtime.log_print('after logger configuration')
            text = (run.root / 'run.log').read_text()
            self.assertIn('before logger configuration', text)
            self.assertIn('after logger configuration', text)

    def test_nested_calls_record_warning_once_and_strip_terminal_colors(self):
        @runtime.logged_operation
        def inner():
            warnings.warn('\x1b[31mcalibration warning\x1b[0m', UserWarning)

        with runtime.execution(self.root) as run:
            inner()
        text = (run.root / 'run.log').read_text()
        self.assertEqual(text.count('UserWarning: calibration warning'), 1)
        self.assertNotIn('\x1b', text)

    def test_processes_reserve_distinct_directories_for_identical_timestamps(self):
        script = '''import contextlib, datetime, io, json, sys
from unittest.mock import patch
from aimet_common import run_context as runtime
now = datetime.datetime(2026, 10, 21, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
with contextlib.redirect_stdout(io.StringIO()), patch.object(runtime, '_now', return_value=now):
    run = runtime.Execution(sys.argv[1])
    run.finish()
print(json.dumps(run.execution_id))
'''
        processes = [subprocess.Popen([sys.executable, '-c', script, str(self.root)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(4)]
        ids = []
        for process in processes:
            output, error = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, error)
            ids.append(json.loads(output))
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(len(list(self.root.glob('*/run.json'))), 4)

    def test_uncaught_error_after_api_call_preserves_python_traceback_and_marks_failure(self):
        script = '''from aimet_common.run_context import logged_operation
@logged_operation
def prepare():
    return 42
assert prepare() == 42
raise ValueError('failure after preparation')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                                env={**os.environ, 'RX_MET_OUTPUT_DIR': str(self.root / 'uncaught')})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ValueError: failure after preparation', result.stderr)
        record_path, = (self.root / 'uncaught').glob('*/run.json')
        self.assertEqual(json.loads(record_path.read_text())['status'], 'failed')
        self.assertIn('ValueError: failure after preparation',
                      record_path.with_name('run.log').read_text())

    def test_console_level_changes_do_not_disable_file_debug(self):
        logger = logging.getLogger('execution-console-level-test')
        output = io.StringIO()
        logger.handlers = [logging.StreamHandler(output)]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            runtime.attach_aimet_logger(logger)
            runtime.set_console_level(logger, logging.WARNING)
            with runtime.execution(self.root) as run:
                logger.debug('detailed file record')
                logger.info('quiet file record')
            self.assertEqual(output.getvalue(), '')
            text = (run.root / 'run.log').read_text()
            self.assertIn('detailed file record', text)
            self.assertIn('quiet file record', text)
        finally:
            logger.handlers = []

    def test_shared_console_handler_preserves_independent_area_levels(self):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        areas = [logging.getLogger('execution-shared-first'), logging.getLogger('execution-shared-second')]
        for logger in areas:
            logger.handlers = [handler]
            logger.setLevel(logging.INFO)
            logger.propagate = False
            runtime.attach_aimet_logger(logger)
        first, second = areas
        try:
            runtime.set_console_level(first, logging.WARNING)
            with runtime.execution(self.root) as run:
                first.info('first-area-hidden')
                second.info('second-area-visible')
                first.debug('first-area-debug')
                second.debug('second-area-debug')
                runtime.set_console_level(second, logging.DEBUG)
                first.info('first-area-still-hidden')
                second.debug('second-area-debug-visible')
            console = output.getvalue()
            self.assertNotIn('first-area-', console)
            self.assertIn('second-area-visible', console)
            self.assertIn('second-area-debug-visible', console)
            text = (run.root / 'run.log').read_text()
            for message in ('first-area-hidden', 'second-area-visible', 'first-area-debug',
                            'first-area-still-hidden', 'second-area-debug-visible'):
                self.assertEqual(text.count(message), 1)
        finally:
            for logger in areas:
                logger.handlers = []

    def test_models_created_together_keep_their_own_early_diagnostics(self):
        class Sim:
            @runtime.logged_operation
            def __init__(self, name):
                self.name = name
                runtime.log_print('prepare ' + name)

        @runtime.logged_export('output_dir', task='export')
        def export(sim, output_dir, filename_prefix):
            path = Path(output_dir) / (filename_prefix + '.onnx')
            path.write_text(sim.name)
            return path

        first, second = Sim('first'), Sim('second')
        first_path = export(first, self.root / 'models', 'model')
        second_path = export(second, self.root / 'models', 'model')
        self.assertNotEqual(first_path.parent, second_path.parent)
        first_log = first_path.parent.parent / 'run.log'
        second_log = second_path.parent.parent / 'run.log'
        self.assertIn('prepare first', first_log.read_text())
        self.assertNotIn('prepare second', first_log.read_text())
        self.assertIn('prepare second', second_log.read_text())
        self.assertNotIn('prepare first', second_log.read_text())

    def test_package_warnings_reach_file_with_old_or_new_console_handlers(self):
        logger = runtime.get_logger('aimet_common.test_runtime_warning')
        root = logging.getLogger()
        saved = root.handlers[:]
        try:
            for configure_inside in (False, True):
                console = io.StringIO()
                root.handlers = [] if configure_inside else [logging.StreamHandler(console)]
                with runtime.execution(self.root) as run:
                    if configure_inside:
                        root.handlers = [logging.StreamHandler(sys.stderr)]
                    logger.warning('owned package warning')
                self.assertEqual((run.root / 'run.log').read_text().count('owned package warning'), 1)
        finally:
            root.handlers = saved

    def test_run_does_not_configure_root_logger(self):
        root = logging.getLogger()
        handlers = root.handlers[:]
        level = root.level
        with runtime.execution(self.root):
            runtime.get_logger('aimet_common.test_owned_logger').warning('owned diagnostic')
        self.assertEqual(root.handlers, handlers)
        self.assertEqual(root.level, level)

    def test_read_only_working_directory_does_not_require_caller_setup(self):
        runtime._finish_implicit()
        readonly = self.root / 'readonly'
        readonly.mkdir()
        requested = readonly / 'output'
        original = Path.mkdir

        def mkdir(path, *args, **kwargs):
            if path == requested:
                raise PermissionError(errno.EACCES, 'read-only test directory')
            return original(path, *args, **kwargs)

        @runtime.logged_operation
        def prepare():
            runtime.log_print('early read-only-directory diagnostic')
            return 42

        with patch.dict(os.environ), patch.object(Path, 'cwd', return_value=readonly), \
                patch.object(Path, 'mkdir', mkdir):
            os.environ.pop('RX_MET_OUTPUT_DIR', None)
            self.assertEqual(prepare(), 42)
            run = runtime.get_run()
            temporary = run.output_root
            self.assertNotEqual(temporary, requested)
            with runtime.export_execution(self.root / 'writable') as exported:
                (exported.exports / 'model.onnx').write_bytes(b'model')
        self.assertFalse(temporary.exists())
        self.assertIn('early read-only-directory diagnostic', (run.root / 'run.log').read_text())


if __name__ == '__main__':
    unittest.main()
