"""FX lifecycle environment verification, without Torch or Docker installed."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import linecache
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load('fx_environment', ROOT / 'scripts/environment/common.py')
helper = load('fx_environment_helper', ROOT / 'scripts/environment/patches/torch_fx_source_cache.py')
PROBE = ROOT / 'scripts/environment/verify_fx_source_cache.py'


class FxEnvironmentTest(unittest.TestCase):
    def test_all_patch_and_verification_sources_are_environment_build_inputs(self):
        for name in ('patch_torch_wheel.py', 'patches/torch_fx_source_cache.py', 'verify_fx_source_cache.py'):
            self.assertIn('scripts/environment/' + name, common.BUILD_FILES)

    def test_verification_uses_bounded_cpu_container_and_captures_exit_stderr(self):
        result = SimpleNamespace(returncode=0, stdout='FX_SOURCE_CACHE_OK\n', stderr='ordinary warning\n')
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(common.subprocess, 'run', return_value=result) as run, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            common.verify_fx_source_cache('candidate-image-id')
        self.assertEqual(stdout.getvalue(), result.stdout)
        self.assertEqual(stderr.getvalue(), result.stderr)
        command = run.call_args.args[0]
        for option in ('--read-only', '--memory', '--pids-limit', '--network'):
            self.assertIn(option, command)
        self.assertNotIn('--gpus', command)
        self.assertEqual(command[command.index('--network') + 1], 'none')
        self.assertEqual(command[command.index('--cpus') + 1], '1')
        self.assertEqual(run.call_args.kwargs['timeout'], 90)
        self.assertTrue(run.call_args.kwargs['capture_output'])
        self.assertIn('signal.alarm(60)', run.call_args.kwargs['input'])
        self.assertIn('FX_SOURCE_CACHE_SKIP', run.call_args.kwargs['input'])

    def test_failure_with_zero_exit_and_nanobind_warning_is_not_accepted(self):
        result = SimpleNamespace(returncode=0, stdout='', stderr='nanobind: leaked 1 instances!\n')
        with patch.object(common.subprocess, 'run', return_value=result), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, 'FX source-cache verification failed'):
            common.verify_fx_source_cache('candidate-image-id')

    def test_timeout_removes_only_the_named_fx_container(self):
        with patch.object(common.subprocess, 'run', side_effect=[
                subprocess.TimeoutExpired('docker', 90, stderr=b'partial FX stderr\n'),
                SimpleNamespace(returncode=0, stdout='', stderr='')]) as run, \
                contextlib.redirect_stderr(io.StringIO()) as stderr, \
                self.assertRaisesRegex(RuntimeError, 'FX source-cache verification timed out; container removed'):
            common.verify_fx_source_cache('candidate-image-id')
        first, cleanup = run.call_args_list
        name = first.args[0][first.args[0].index('--name') + 1]
        self.assertTrue(name.startswith('rx-met-fx-source-cache-verify-'))
        self.assertEqual(cleanup.args[0], ['docker', 'rm', '-f', name])
        self.assertEqual(stderr.getvalue(), 'partial FX stderr\n')

    def test_both_build_and_runtime_images_run_the_fx_probe(self):
        with patch.object(common, 'inspect', return_value={'Id': 'candidate-image-id'}), \
                patch.object(common, 'run'), patch.object(common, 'verify_onnxsim_shutdown'), \
                patch.object(common, 'verify_fx_source_cache') as verify:
            common.verify('cu130')
        self.assertEqual(verify.call_count, 2)

    def fake_runtime(self, version='2.10.0+cu130', *, leaking=False):
        torch = ModuleType('torch')
        torch.__version__ = version
        torch.__path__ = []
        torch.set_num_threads = Mock()
        torch.set_num_interop_threads = Mock()
        fx = ModuleType('torch.fx')
        fx.__path__ = []
        graph_module = ModuleType('torch.fx.graph_module')
        graph_module._loader = helper.OwnedEvalCacheLoader()
        if leaking:
            def original_exec(source, namespace, co_fields=None):
                key = graph_module._loader.cache(source, namespace, co_fields)
                exec(compile(source, key, 'exec'), namespace)
            graph_module._exec_with_source = original_exec
        else:
            graph_module._exec_with_source = graph_module._loader.exec_with_source
        torch.fx, fx.graph_module = fx, graph_module
        modules = {'torch': torch, 'torch.fx': fx, 'torch.fx.graph_module': graph_module,
                   'torch.fx._rx_source_cache': helper}

        def cleanup():
            for key in list(graph_module._loader.eval_cache):
                linecache.cache.pop(key, None)
            graph_module._loader.eval_cache.clear()
        self.addCleanup(cleanup)
        return modules

    def test_actual_probe_checks_live_function_traceback_and_reclamation(self):
        with patch.dict(sys.modules, self.fake_runtime()), contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(PROBE), run_name='__main__')
        self.assertIn('FX_SOURCE_CACHE_OK', output.getvalue())

    def test_actual_probe_rejects_the_original_unbounded_execution_path(self):
        with patch.dict(sys.modules, self.fake_runtime(leaking=True)), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(AssertionError, 'dead generated function retained its source'):
            runpy.run_path(str(PROBE), run_name='__main__')

    def test_actual_probe_explicitly_skips_uncovered_versions(self):
        with patch.dict(sys.modules, self.fake_runtime('2.9.1+cpu')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(PROBE), run_name='__main__')
        self.assertIn('FX_SOURCE_CACHE_SKIP torch=2.9.1+cpu', output.getvalue())
        self.assertIn('not covered', output.getvalue())


if __name__ == '__main__':
    unittest.main()
