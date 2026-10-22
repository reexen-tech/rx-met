"""Automatic execution layout and logging, without Torch or a native runtime."""
import contextlib
import datetime
import errno
import io
import json
import logging
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
        temporary = tempfile.TemporaryDirectory(prefix='rx-met-execution-')
        self.addCleanup(temporary.cleanup)
        self.addCleanup(runtime._finish_implicit)
        self.root = Path(temporary.name)
        patcher = patch.dict(os.environ, {'RX_MET_OUTPUT_DIR': str(self.root / 'automatic')})
        patcher.start()
        self.addCleanup(patcher.stop)

    def read_record(self, run):
        return json.loads((run.root / 'run.json').read_text())

    def test_timestamp_collision_preserves_previous_execution(self):
        now = datetime.datetime(2026, 10, 21, 6, 30, 25, 123456, datetime.timezone.utc)
        with patch.object(runtime, '_now', return_value=now):
            with runtime.execution(self.root) as first:
                (first.exports / 'model.onnx').write_bytes(b'first')
            with runtime.execution(self.root) as second:
                (second.exports / 'model.onnx').write_bytes(b'second')
        self.assertEqual(first.execution_id, '20261021T063025123456Z')
        self.assertEqual(second.execution_id, first.execution_id + '-1')
        self.assertEqual((first.root / 'exports/model.onnx').read_bytes(), b'first')
        self.assertFalse((self.root / 'runs').exists())
        self.assertEqual(set(self.root.iterdir()), {first.root, second.root})

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
        with runtime.execution(self.root) as run:
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
now = datetime.datetime(2026, 10, 21, tzinfo=datetime.timezone.utc)
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
