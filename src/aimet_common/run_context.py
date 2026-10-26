"""Automatic RX execution directories and independent console/file logging."""
from __future__ import annotations

import atexit
import builtins
import contextlib
import contextvars
import datetime
import errno
import fcntl
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import weakref
from importlib import metadata

from ._version import __version__

_current = contextvars.ContextVar('rx_met_execution', default=None)
_details = contextvars.ContextVar('rx_met_log_details', default=None)
_passthrough = contextvars.ContextVar('rx_met_console_passthrough', default=False)
_implicit = {}
_last_runs = {}
_model_runs = weakref.WeakKeyDictionary()
_runs = weakref.WeakSet()
_lock = threading.RLock()
_ansi = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
_exception_hook_installed = False
_capture_lock = threading.RLock()
_capture_depth = 0
_capture_state = None
_timestamp_output = True


def configure_output(*, timestamp_output: bool = True) -> None:
    """Configure output directories for subsequently created RX tasks.

    Call before creating QuantSim or starting a quantization workflow. False
    writes directly into the output root, replacing same-named files and logs.
    Existing tasks retain their configuration. Calling without arguments
    restores timestamped output, which is the default.
    """
    if type(timestamp_output) is not bool:
        raise TypeError('timestamp_output must be a bool')
    global _timestamp_output
    with _lock:
        _timestamp_output = timestamp_output


def _lock_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException as error:
        os.close(descriptor)
        if isinstance(error, BlockingIOError):
            raise RuntimeError(f'输出目录正在被另一个任务使用: {directory}') from error
        raise
    return descriptor


def _output_state(root):
    state = {}
    for directory in ('exports', 'checkpoints'):
        for path in (root / directory).rglob('*'):
            if path.is_file():
                stat = path.stat()
                state[path.relative_to(root).as_posix()] = (
                    stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return state


def _prepare_fixed_directory(root):
    for name in ('run.log', 'run.json', 'config', 'exports', 'checkpoints', '_work'):
        path = root / name
        if path.is_symlink():
            raise ValueError(f'固定输出目录中的工具路径不能是符号链接: {path}')
        if path.is_dir():
            for item in path.rglob('*'):
                if item.is_symlink():
                    raise ValueError(f'固定输出目录中的工具路径不能是符号链接: {item}')
    return _output_state(root)


class _PlainFormatter(logging.Formatter):
    def format(self, record):
        return _ansi.sub('', super().format(record))


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=f'.{path.name}.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
            stream.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resolve_root(output_root):
    directory = os.environ.get('RX_MET_OUTPUT_DIR', Path.cwd() / 'output') if output_root is None else output_root
    return Path(directory).resolve()


def _root_key(output_root):
    return (*_key(), str(_resolve_root(output_root)))


def _tool_identity():
    versions = {'rx-met': __version__, 'python': sys.version.split()[0]}
    for name in ('rx-met', 'quant-gru', 'quant-lstm', 'torch', 'onnx', 'onnxruntime-gpu'):
        try:
            versions['rx-met-distribution' if name == 'rx-met' else name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    package = Path(__file__).resolve()
    source = package.parents[2]
    if (source / '.git').exists():
        build = {}
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source,
                                               stderr=subprocess.DEVNULL, text=True, timeout=3).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source,
                                                 stderr=subprocess.DEVNULL, timeout=3).strip())
            build = {'source_revision': revision, 'dirty': dirty}
        except (OSError, subprocess.SubprocessError):
            pass
        return {'versions': versions, 'build': build, 'python_module': str(package)}
    for parent in (package.parents[2], package.parents[3],
                   Path(os.environ.get('RX_MET_HOME', '/opt/rx-met'))):
        manifest = parent / 'manifest.json'
        if manifest.is_file():
            try:
                bundle = json.loads(manifest.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if not isinstance(bundle, dict) or 'bundle_name' not in bundle:
                continue
            return {'versions': versions, 'build': {
                key: bundle.get(key) for key in ('source_revision', 'variant', 'wheel_source_sha256')
            }, 'runtime': bundle.get('runtime', {}), 'python_module': str(package)}
    return {'versions': versions, 'build': {}, 'python_module': str(package)}


class Execution:
    """Own one task's paths, log stream and durable running/success/failure record."""

    def __init__(self, output_root=None, task='quantization', *, provisional=False):
        self.output_root = _resolve_root(output_root)
        with _lock:
            self.timestamp_output = _timestamp_output
        self.task = task
        self.pid = os.getpid()
        self.provisional = provisional
        self.model_bound = False
        self.closed = False
        self._lock = threading.RLock()
        self._errors = set()
        self._started = time.monotonic()
        self._temporary_root = None
        self._run_dir = None
        self._directory_lock = None
        self._baseline_outputs = {}
        self._work_dir = None
        timestamp = _now().strftime('%Y%m%dT%H%M%S%fZ')
        try:
            try:
                self._reserve(timestamp)
            except OSError as error:
                if (not provisional or output_root is not None or 'RX_MET_OUTPUT_DIR' in os.environ
                        or error.errno not in (errno.EACCES, errno.EROFS)):
                    raise
                self.output_root = Path(tempfile.mkdtemp(prefix='rx-met-run-'))
                self._temporary_root = self.output_root
                self._reserve(timestamp)
            self._handler = self._open_handler(fresh=not self.timestamp_output)
            self.record = {'schema_version': 1, 'execution_id': self.execution_id, 'task': task,
                           'status': 'running', 'started_at': _now().isoformat(), 'pid': self.pid,
                           'output_config': {'timestamp_output': self.timestamp_output},
                           'tool': _tool_identity(), 'inputs': {}, 'config': {}, 'outputs': {}, 'operations': {}}
            self.save()
        except BaseException:
            if hasattr(self, '_handler'):
                self._handler.close()
            self._release_directory()
            if self._temporary_root is not None:
                shutil.rmtree(self._temporary_root, ignore_errors=True)
            raise
        _runs.add(self)
        self.message(f'RX-MET execution {self.execution_id}: {task}', console=False)
        self.message('Tool identity: ' + json.dumps(self.record['tool'], ensure_ascii=False), console=False)
        builtins.print(f'[rx-met] 日志: {self.root / "run.log"}', flush=True)

    def _reserve(self, timestamp):
        self.execution_id = timestamp
        if not self.timestamp_output and self.provisional:
            # Unknown final roots must not make independent models share a log.
            self._temporary_root = Path(tempfile.mkdtemp(prefix='rx-met-run-'))
            self._run_dir = self._temporary_root
            return
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not self.timestamp_output:
            self._run_dir = self.output_root
            self._directory_lock = _lock_directory(self.output_root)
            self._baseline_outputs = _prepare_fixed_directory(self.output_root)
            return
        index = 0
        while True:
            self.execution_id = timestamp if not index else f'{timestamp}-{index}'
            self._run_dir = self.output_root / self.execution_id
            try:
                self.root.mkdir()
                break
            except FileExistsError:
                index += 1

    @property
    def root(self):
        return self._run_dir

    def _release_directory(self):
        if self._directory_lock is not None:
            os.close(self._directory_lock)
            self._directory_lock = None

    def _assert_current(self):
        if not self.timestamp_output and self.closed:
            record = json.loads((self.root / 'run.json').read_text(encoding='utf-8'))
            if record.get('execution_id') != self.execution_id:
                raise RuntimeError(f'输出目录已被后续任务覆盖，不能修改旧任务记录: {self.root}')

    def directory(self, name):
        path = self.root / name
        if not self.timestamp_output and path.is_symlink():
            raise ValueError(f'固定输出目录中的工具路径不能是符号链接: {path}')
        path.mkdir(exist_ok=True)
        return path

    @property
    def exports(self):
        return self.directory('exports')

    @property
    def checkpoints(self):
        return self.directory('checkpoints')

    @property
    def work(self):
        if self._work_dir is None:
            parent = self.directory('_work')
            self._work_dir = parent if self.timestamp_output else Path(
                tempfile.mkdtemp(prefix='run-', dir=parent))
        return self._work_dir

    def _open_handler(self, *, fresh=False):
        if fresh:
            with (self.root / 'run.log').open('w', encoding='utf-8'):
                pass
        # Append handlers can reopen after an external logging.dictConfig closes them.
        handler = logging.FileHandler(self.root / 'run.log', mode='a', encoding='utf-8')
        formatter = _PlainFormatter('%(asctime)sZ %(levelname)s %(name)s: %(message)s',
                                      datefmt='%Y-%m-%dT%H:%M:%S')
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        return handler

    def bind_root(self, output_root):
        """Move early library diagnostics with the task once its export root is known."""
        output_root = Path(output_root).resolve()
        if (self.timestamp_output or not self.provisional) and output_root in (
                self.output_root, self.root, self.root / 'exports'):
            self.provisional = False
            return
        if not self.timestamp_output and output_root in (self.root, self.root / 'exports'):
            output_root = self.output_root
        if not self.provisional:
            raise ValueError(f'当前任务的输出根目录为 {self.output_root}，不能改为 {output_root}')
        with self._lock:
            old = self.root
            output_root.mkdir(parents=True, exist_ok=True)
            directory_lock = None
            baseline = {}
            if self.timestamp_output:
                name = self.execution_id
                index = 0
                while True:
                    candidate = output_root / (name if not index else f'{name}-{index}')
                    try:
                        candidate.mkdir()
                        break
                    except FileExistsError:
                        index += 1
            else:
                candidate = output_root
                directory_lock = _lock_directory(candidate)
                try:
                    baseline = _prepare_fixed_directory(candidate)
                except BaseException:
                    os.close(directory_lock)
                    raise
            self._handler.close()
            try:
                if not self.timestamp_output:
                    # Keep the user's output root permissions when merging a private stage.
                    for entry in old.iterdir():
                        target = candidate / entry.name
                        if entry.is_dir():
                            shutil.copytree(entry, target, dirs_exist_ok=True)
                        else:
                            shutil.copy2(entry, target)
                    shutil.rmtree(old, ignore_errors=True)
                else:
                    try:
                        os.replace(old, candidate)
                    except OSError as error:
                        if error.errno != errno.EXDEV:
                            raise
                        shutil.copytree(old, candidate, dirs_exist_ok=True)
                        shutil.rmtree(old, ignore_errors=True)
            except BaseException:
                if self.timestamp_output:
                    shutil.rmtree(candidate, ignore_errors=True)
                elif directory_lock is not None:
                    os.close(directory_lock)
                self._handler = self._open_handler()
                raise
            self.output_root = output_root
            self._run_dir = candidate
            self._directory_lock = directory_lock
            self._baseline_outputs = baseline
            if self._work_dir is not None:
                self._work_dir = candidate / self._work_dir.relative_to(old)
            if self.timestamp_output:
                self.execution_id = candidate.name
            try:
                self._handler = self._open_handler()
            except BaseException:
                self._release_directory()
                raise
            self.provisional = False
            if self._temporary_root is not None:
                try:
                    self._temporary_root.rmdir()
                except OSError:
                    pass
                self._temporary_root = None
            self.record['execution_id'] = self.execution_id
            self.save()
            builtins.print(f'[rx-met] 日志: {self.root / "run.log"}', flush=True)

    def save(self):
        with self._lock:
            self._assert_current()
            _atomic_json(self.root / 'run.json', self.record)

    def emit(self, record):
        with self._lock:
            if not self.closed:
                self._handler.handle(record)
                self._handler.flush()

    def message(self, message, *, console=True, level=logging.INFO):
        self.emit(logging.LogRecord('rx-met', level, '', 0, message, (), None))
        if console:
            with console_passthrough():
                builtins.print(message)

    def snapshot_config(self, value, name='quantization'):
        if value is None:
            return
        if isinstance(value, dict):
            payload = value
            source = None
        elif isinstance(value, (str, os.PathLike)) and Path(value).is_file():
            source = str(Path(value).resolve())
            try:
                payload = json.loads(Path(value).read_text(encoding='utf-8'))
            except (ValueError, UnicodeError):
                return
        else:
            return
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        directory = self.directory('config')
        filename = f'{name}.json'
        previous = self.record['config'].get(name)
        if previous and previous['content_sha256'] == digest:
            return
        if previous and previous['content_sha256'] != digest:
            filename = f'{name}-{digest[:12]}.json'
            name = f'{name}-{digest[:12]}'
        _atomic_json(directory / filename, payload)
        self.record['config'][name] = {'path': f'config/{filename}', 'source': source,
                                      'sha256': _sha256(directory / filename), 'content_sha256': digest}
        self.save()

    def note_inputs(self, **values):
        for name, value in values.items():
            if value is None:
                continue
            path = Path(value).resolve()
            identity = {'path': str(path), 'exists': path.exists()}
            if path.is_file():
                identity.update(size_bytes=path.stat().st_size, sha256=_sha256(path))
            self.record['inputs'][name] = identity
        self.save()

    def exception(self, error):
        if id(error) in self._errors:
            return
        self._errors.add(id(error))
        record = logging.LogRecord('rx-met', logging.ERROR, '', 0, str(error), (),
                                   (type(error), error, error.__traceback__))
        self.emit(record)
        self.record['error'] = {'type': type(error).__name__, 'message': str(error)}
        self.save()

    def finish(self, error=None):
        with self._lock:
            if self.closed:
                return
            try:
                if error is not None:
                    self.exception(error)
                self.record['status'] = 'failed' if error is not None else 'success'
                self.record['finished_at'] = _now().isoformat()
                self.record['elapsed_s'] = round(time.monotonic() - self._started, 6)
                if not self.timestamp_output and self.provisional:
                    self.save()
                    try:
                        self.bind_root(self.output_root)
                    except BaseException as binding_error:
                        self.record['status'] = 'failed'
                        self.exception(binding_error)
                        raise
                for name, state in sorted(_output_state(self.root).items()):
                    if self.timestamp_output or self._baseline_outputs.get(name) != state:
                        path = self.root / name
                        identity = {
                            'size_bytes': path.stat().st_size, 'sha256': _sha256(path)}
                        if path.suffix in ('.encodings', '.json'):
                            try:
                                document = json.loads(path.read_text(encoding='utf-8'))
                                if isinstance(document, dict) and all(
                                        isinstance(document.get(key), (dict, list))
                                        for key in ('activation_encodings', 'param_encodings')):
                                    identity.update({key: document[key] for key in ('version', 'schema_version')
                                                     if key in document})
                            except (OSError, ValueError):
                                pass
                        self.record['outputs'][name] = identity
                self.save()
                if self.record['status'] == 'success':
                    if self.timestamp_output:
                        shutil.rmtree(self.root / '_work', ignore_errors=True)
                    elif self._work_dir is not None:
                        shutil.rmtree(self._work_dir, ignore_errors=True)
                        try:
                            self._work_dir.parent.rmdir()
                        except OSError:
                            pass
            finally:
                self._handler.close()
                self.closed = True
                self._release_directory()


def _key():
    return os.getpid(), threading.get_ident()


def active_run():
    run = _current.get()
    return run if run is not None and run.pid == os.getpid() and not run.closed else None


def _associated_run(owner):
    try:
        return _model_runs.get(owner)
    except TypeError:
        return None


def _associate(owner, run):
    owners = [owner]
    model = getattr(owner, 'model', None)
    if model is not None and hasattr(model, 'named_modules'):
        owners.extend(module for _, module in model.named_modules())
    for value in owners:
        try:
            _model_runs[value] = run
        except TypeError:
            pass
    run.model_bound = True


def get_run(owner=None, *, model_init=False):
    run = active_run()
    if run is not None:
        return run
    with _lock:
        associated = _associated_run(owner) if owner is not None and not model_init else None
        run = associated or _implicit.get(_key())
        if run is None or run.closed or model_init and run.model_bound:
            run = Execution(provisional=True)
            _install_exception_hook()
            if associated is not None:
                _associate(owner, run)
        _implicit[_key()] = run
        return run


@contextlib.contextmanager
def console_passthrough():
    token = _passthrough.set(True)
    try:
        yield
    finally:
        _passthrough.reset(token)


@contextlib.contextmanager
def log_details(verbose):
    """Produce diagnostic text for the file while preserving console verbosity."""
    parent = _details.get()
    token = _details.set(bool(verbose) if parent is None else parent and bool(verbose))
    try:
        yield
    finally:
        _details.reset(token)


def log_print(*values, sep=' ', end='\n', file=None, flush=False):
    if file is not None and file not in (sys.stdout, sys.stderr):
        return builtins.print(*values, sep=sep, end=end, file=file, flush=flush)
    message = sep.join(str(value) for value in values) + end
    run = get_run()
    level = logging.DEBUG if _details.get() is not None else logging.INFO
    name = sys._getframe(1).f_globals.get('__name__', 'rx-met')
    run.emit(logging.LogRecord(name, level, '', 0, message.rstrip('\n'), (), None))
    if _details.get() is not False:
        with console_passthrough():
            builtins.print(*values, sep=sep, end=end, file=file, flush=flush)


class _Tee:
    def __init__(self, original):
        self.original = original.original if isinstance(original, _Tee) else original
        self._local = threading.local()

    def write(self, text):
        run = active_run()
        quiet = run is not None and _details.get() is False and not _passthrough.get()
        result = len(text) if quiet else self.original.write(text)
        if not _passthrough.get() and '\r' not in text and run is not None:
            pending = getattr(self._local, 'pending', {})
            text = pending.get(run, '') + text
            while '\n' in text:
                line, text = text.split('\n', 1)
                if line:
                    run.message(line, console=False)
            pending[run] = text
            self._local.pending = pending
        return result

    def flush_run(self, run):
        self.original.flush()
        pending = getattr(self._local, 'pending', {})
        text = pending.pop(run, '')
        if text and not run.closed:
            run.message(text, console=False)

    def flush(self):
        run = active_run()
        if run is not None:
            self.flush_run(run)
        else:
            self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextlib.contextmanager
def _capture(run):
    global _capture_depth, _capture_state
    with _capture_lock:
        if _capture_depth == 0:
            stdout, stderr = _Tee(sys.stdout), _Tee(sys.stderr)
            original_warning = warnings.showwarning

            def showwarning(message, category, filename, lineno, file=None, line=None):
                current = active_run()
                if current is not None:
                    current.message(warnings.formatwarning(message, category, filename, lineno, line),
                                    console=False, level=logging.WARNING)
                if _details.get() is not False:
                    with console_passthrough():
                        original_warning(message, category, filename, lineno, file=file, line=line)

            _capture_state = stdout, stderr, original_warning, showwarning
            sys.stdout, sys.stderr, warnings.showwarning = stdout, stderr, showwarning
        _capture_depth += 1
        state = _capture_state
    try:
        yield
    finally:
        stdout, stderr, original_warning, showwarning = state
        try:
            stdout.flush_run(run)
            stderr.flush_run(run)
        finally:
            with _capture_lock:
                _capture_depth -= 1
                if _capture_depth == 0:
                    if sys.stdout is stdout:
                        sys.stdout = stdout.original
                    if sys.stderr is stderr:
                        sys.stderr = stderr.original
                    if warnings.showwarning is showwarning:
                        warnings.showwarning = original_warning
                    _capture_state = None


def _record_failure(run, error, *, finish=False):
    try:
        if finish:
            run.finish(error)
        else:
            run.exception(error)
    except Exception as logging_error:
        if finish:
            run._handler.close()
            run.closed = True
        with console_passthrough():
            builtins.print(f'[rx-met] 无法完成日志记录: {logging_error}', file=sys.stderr)


@contextlib.contextmanager
def execution(output_root, task='quantization'):
    """Internal complete-workflow boundary; callers of public APIs need no setup."""
    _last_runs.pop(_root_key(output_root), None)
    run = Execution(output_root, task)
    _last_runs[_root_key(run.output_root)] = run
    token = _current.set(run)
    try:
        with _capture(run):
            yield run
    except BaseException as error:
        _record_failure(run, error, finish=True)
        raise
    else:
        run.finish()
    finally:
        _current.reset(token)


@contextlib.contextmanager
def export_execution(output_root, task='export', *, source=None):
    run = active_run()
    owned = run is None
    if owned:
        run = get_run(source)
        run.record['task'] = task
        _last_runs[_root_key(output_root)] = run
    token = _current.set(run)
    try:
        run.bind_root(output_root)
        _last_runs[_root_key(run.output_root)] = run
        with _capture(run):
            yield run
    except BaseException as error:
        _record_failure(run, error, finish=owned)
        raise
    else:
        if owned:
            run.finish()
    finally:
        _current.reset(token)


def _metadata_run(output_dir):
    run = active_run()
    if run is not None:
        run.bind_root(output_dir)
    else:
        requested = Path(output_dir).resolve()
        run = _last_runs.get(_root_key(requested))
        if run is None:
            run = next((candidate for candidate in _last_runs.values()
                        if requested in (candidate.root, candidate.root / 'exports')), None)
    return run


def metadata_directory(output_dir):
    run = _metadata_run(output_dir)
    if run is not None:
        run._assert_current()
    return run.exports if run else Path(output_dir)


@contextlib.contextmanager
def metadata_output(output_dir):
    """Keep post-export metadata writes within the fixed directory's ownership."""
    run = _metadata_run(output_dir)
    descriptor = None
    try:
        if run is not None:
            if run.closed and not run.timestamp_output:
                descriptor = _lock_directory(run.root)
            run._assert_current()
        yield run.exports if run else Path(output_dir)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def note_output(path):
    path = Path(path).resolve()
    candidates = [active_run(), *sorted(set(_last_runs.values()), key=lambda run: run._started, reverse=True)]
    for run in candidates:
        if run is not None and path.is_relative_to(run.root):
            relative = path.relative_to(run.root)
            if not relative.parts or relative.parts[0] not in ('exports', 'checkpoints'):
                continue
            if run.closed:
                run._assert_current()
                run.record['outputs'][relative.as_posix()] = {
                    'size_bytes': path.stat().st_size, 'sha256': _sha256(path)}
                run.save()
            return


def logged_operation(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        owned = active_run() is None
        run = get_run(args[0] if args else None, model_init=function.__name__ == '__init__')
        token = _current.set(run)
        started = time.monotonic()
        failure = None
        try:
            bound = signature.bind(*args, **kwargs)
            for name in ('config_file', 'config_path', 'bitwidth_config_file'):
                if name in bound.arguments:
                    run.snapshot_config(bound.arguments[name], name)
            with _capture(run):
                result = function(*args, **kwargs)
                if function.__name__ == '__init__' and args:
                    _associate(args[0], run)
                return result
        except BaseException as error:
            failure = error
            if not owned:
                _record_failure(run, error)
            raise
        finally:
            try:
                name = function.__module__ + '.' + function.__qualname__
                with run._lock:
                    operation = run.record['operations'].setdefault(name, {'calls': 0, 'failures': 0, 'elapsed_s': 0})
                    operation['calls'] += 1
                    operation['failures'] += failure is not None
                    operation['elapsed_s'] = round(operation['elapsed_s'] + time.monotonic() - started, 6)
                if failure is not None and owned:
                    _record_failure(run, failure, finish=True)
            finally:
                _current.reset(token)
    return wrapped


def logged_workflow(output_argument, *, task):
    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            with execution(bound.arguments[output_argument], task) as run:
                run.note_inputs(**{name: bound.arguments[name] for name in
                                   ('model_path', 'calibration_dir', 'validation_dir') if name in bound.arguments})
                for name in ('config_file', 'bitwidth_config_file'):
                    run.snapshot_config(bound.arguments.get(name), name)
                bound.arguments[output_argument] = run.root
                return function(*bound.args, **bound.kwargs)
        return wrapped
    return decorate


def logged_export(output_argument, *, task):
    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            root = bound.arguments[output_argument]
            with export_execution(root, task, source=bound.arguments.get('sim')) as run:
                prefix = bound.arguments.get('filename_prefix', '')
                if not prefix or Path(prefix).name != prefix or prefix in ('.', '..'):
                    raise ValueError('filename_prefix 必须是单个文件名，不能包含目录')
                bound.arguments[output_argument] = run.exports
                return function(*bound.args, **bound.kwargs)
        return wrapped
    return decorate


class _FileRelay(logging.Handler):
    def emit(self, record):
        run = active_run() or _implicit.get(_key())
        if run is not None and not run.closed:
            run.emit(record)


class _ConsoleFilter(logging.Filter):
    def __init__(self, logger_name, minimum):
        super().__init__()
        self.logger_name = logger_name
        self.minimum = minimum

    def filter(self, record):
        if record.name != self.logger_name and not record.name.startswith(self.logger_name + '.'):
            return True
        return _details.get() is not False and record.levelno >= self.minimum


def attach_aimet_logger(logger):
    """AIMET areas do not propagate, so attach directly to each configured area."""
    if any(isinstance(handler, _FileRelay) for handler in logger.handlers):
        return
    minimum = logger.getEffectiveLevel()
    for handler in logger.handlers:
        handler.addFilter(_ConsoleFilter(logger.name, max(minimum, handler.level)))
    logger.addHandler(_FileRelay(logging.DEBUG))
    logger.setLevel(logging.DEBUG)


def set_console_level(logger, level):
    if not any(isinstance(handler, _FileRelay) for handler in logger.handlers):
        logger.setLevel(level)
        return
    for handler in logger.handlers:
        for filter_ in handler.filters:
            if isinstance(filter_, _ConsoleFilter) and filter_.logger_name == logger.name:
                filter_.minimum = max(level, handler.level)
    logger.setLevel(logging.DEBUG)


class _Route(logging.Handler):
    def emit(self, record):
        get_run().emit(record)
        if _details.get() is not False and record.levelno >= logging.INFO:
            with console_passthrough():
                builtins.print(self.format(record), file=sys.stderr if record.levelno >= logging.WARNING else sys.stdout)


def get_logger(name):
    logger = logging.getLogger('rx-met.' + name)
    if not any(isinstance(handler, _Route) for handler in logger.handlers):
        logger.addHandler(_Route())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def _finish_implicit():
    for run in sorted(list(_runs), key=lambda run: run._started):
        if run.pid == os.getpid() and not run.closed:
            run.finish()


def _close_inherited_directory_locks():
    # DataLoader workers must not keep their parent's fixed output directory locked.
    for run in list(_runs):
        if run.pid != os.getpid():
            run._release_directory()


def _install_exception_hook():
    global _exception_hook_installed
    if _exception_hook_installed:
        return
    original = sys.excepthook

    def hook(error_type, error, traceback):
        run = active_run() or _implicit.get(_key())
        try:
            if run is not None and not run.closed:
                _record_failure(run, error, finish=True)
        finally:
            original(error_type, error, traceback)

    sys.excepthook = hook
    _exception_hook_installed = True


os.register_at_fork(after_in_child=_close_inherited_directory_locks)
atexit.register(_finish_implicit)
