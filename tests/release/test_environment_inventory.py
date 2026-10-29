"""Image inventories describe installed versions without importing GPU libraries."""
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.support.paths import REPO_ROOT as ROOT

SPEC = importlib.util.spec_from_file_location('environment_inventory_tests', ROOT / 'scripts/release/environment_inventory.py')
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class EnvironmentInventoryTest(unittest.TestCase):
    image_id = 'sha256:' + 'a' * 64

    def result(self):
        components = {name: None for name in ('torch', 'torchvision', 'torchaudio', 'onnx',
                                               'onnxruntime', 'onnxruntime-gpu', 'numpy')}
        components.update(cuda={'version': None, 'python_packages': {}, 'system_packages': {}},
                          cudnn={'python_packages': {}, 'system_packages': {}})
        return {'source': 'image', 'os': {'id': 'ubuntu', 'version_id': '22.04'},
                'architecture': 'x86_64', 'python': {'version': '3.10.12', 'implementation': 'CPython'},
                'components': components, 'python_packages': {}, 'system_packages': {}}

    def test_only_exact_image_id_is_accepted(self):
        with patch.object(inventory.subprocess, 'run') as docker:
            for reference in ('rx-quant-env-runtime:202610-cpu', 'sha256:abc', '', None):
                with self.subTest(reference=reference), self.assertRaisesRegex(RuntimeError, '镜像 ID'):
                    inventory.collect_image_inventory(reference)
            docker.assert_not_called()

    def test_docker_run_is_bounded_read_only_offline_and_without_gpu_access(self):
        expected = self.result()
        with patch.object(inventory.subprocess, 'run', return_value=subprocess.CompletedProcess(
                [], 0, json.dumps(expected), '')) as docker:
            self.assertEqual(inventory.collect_image_inventory(self.image_id), expected)
        docker.assert_called_once()
        command = docker.call_args.args[0]
        for option in ('--rm', '--interactive', '--read-only'):
            self.assertIn(option, command)
        for option, value in (('--network', 'none'), ('--pull', 'never'), ('--cpus', '1'),
                              ('--memory', '1g'), ('--memory-swap', '1g'), ('--pids-limit', '64'),
                              ('--user', f'{os.getuid()}:{os.getgid()}'), ('--group-add', '986'),
                              ('--tmpfs', '/tmp:rw,nosuid,size=64m'), ('--entrypoint', 'python3')):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertEqual(command[-4:], [self.image_id, '-I', '-B', '-'])
        for setting in ('HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1', 'NVIDIA_VISIBLE_DEVICES=void',
                        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1'):
            self.assertIn(setting, command)
        self.assertNotIn('--gpus', command)
        self.assertNotIn('--volume', command)
        self.assertEqual(docker.call_args.kwargs['timeout'], 60)
        self.assertEqual(docker.call_args.kwargs['input'], inventory._COLLECT_SCRIPT)

    def test_collection_script_reads_installed_versions_and_preserves_metadata_precedence(self):
        packages = [('Torch', '2.9.0'), ('torch', '2.8.0'), ('onnxruntime_gpu', '1.23.1'),
                    ('nvidia-cuda-runtime-cu13', '13.0.96'), ('nvidia-cudnn-cu13', '9.13.0.50'),
                    ('numpy', '2.1.3')]
        distributions = [SimpleNamespace(metadata={'Name': name}, version=version) for name, version in packages]
        dpkg_output = ('libcudnn9-cuda-13:amd64\t9.13.0.50-1\tii \n'
                       'cuda-cudart-13-0\t13.0.96-1\tii \n'
                       'held-package\t1.2.3\thi \n'
                       'removed-package\t1.0\trc \n')
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(importlib.metadata, 'distributions', return_value=distributions))
            stack.enter_context(patch.object(platform, 'freedesktop_os_release', return_value={
                'ID': 'ubuntu', 'NAME': 'Ubuntu', 'VERSION_ID': '24.04', 'PRETTY_NAME': 'Ubuntu 24.04.3 LTS'}))
            stack.enter_context(patch.object(platform, 'machine', return_value='x86_64'))
            stack.enter_context(patch.object(platform, 'python_version', return_value='3.12.3'))
            stack.enter_context(patch.object(platform, 'python_implementation', return_value='CPython'))
            stack.enter_context(patch.dict(os.environ, {'CUDA_VERSION': '13.0.0'}, clear=True))
            dpkg = stack.enter_context(patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess(
                [], 0, dpkg_output, '')))
            stack.enter_context(contextlib.redirect_stdout(output))
            exec(inventory._COLLECT_SCRIPT, {})
        actual = inventory._parse_inventory(output.getvalue())
        self.assertEqual(actual['python']['version'], '3.12.3')
        self.assertEqual(actual['os']['pretty_name'], 'Ubuntu 24.04.3 LTS')
        self.assertEqual(actual['components']['torch'], '2.9.0')
        self.assertEqual(actual['components']['onnxruntime-gpu'], '1.23.1')
        self.assertIsNone(actual['components']['onnxruntime'])
        self.assertEqual(actual['components']['cuda'], {
            'version': '13.0.0', 'python_packages': {'nvidia-cuda-runtime-cu13': '13.0.96'},
            'system_packages': {'cuda-cudart-13-0': '13.0.96-1'}})
        self.assertEqual(actual['components']['cudnn'], {
            'python_packages': {'nvidia-cudnn-cu13': '9.13.0.50'},
            'system_packages': {'libcudnn9-cuda-13:amd64': '9.13.0.50-1'}})
        self.assertEqual(actual['python_packages']['torch'], '2.9.0')
        self.assertEqual(actual['system_packages']['held-package'], '1.2.3')
        self.assertNotIn('removed-package', actual['system_packages'])
        self.assertEqual(dpkg.call_args.args[0][0], 'dpkg-query')
        self.assertNotIn('import torch', inventory._COLLECT_SCRIPT)
        self.assertNotIn('import onnxruntime', inventory._COLLECT_SCRIPT)

    @unittest.skipUnless(shutil.which('dpkg-query'), 'Debian package metadata is unavailable')
    def test_collection_script_runs_with_real_python_and_system_package_metadata(self):
        result = subprocess.run([sys.executable, '-I', '-B', '-'], input=inventory._COLLECT_SCRIPT,
                                check=True, capture_output=True, text=True, timeout=30)
        actual = inventory._parse_inventory(result.stdout)
        self.assertEqual(actual['python']['version'], platform.python_version())
        self.assertEqual(actual['architecture'], platform.machine())
        self.assertTrue(actual['system_packages'])
        for name in ('torch', 'onnx', 'numpy'):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = None
            self.assertEqual(actual['components'][name], version)

    def test_failed_collection_cleans_only_its_unique_container(self):
        failures = [subprocess.CompletedProcess([], 1, '', 'container failed'),
                    subprocess.TimeoutExpired(['docker', 'run'], 60), OSError('docker unavailable')]
        for failure in failures:
            with self.subTest(failure=failure):
                def docker(command, **kwargs):
                    if command[:2] == ['docker', 'run']:
                        if isinstance(failure, Exception):
                            raise failure
                        return failure
                    return subprocess.CompletedProcess(command, 0, '', '')
                with patch.object(inventory.subprocess, 'run', side_effect=docker) as runner:
                    with self.assertRaisesRegex(RuntimeError, '无法采集'):
                        inventory.collect_image_inventory(self.image_id)
                run_command = runner.call_args_list[0].args[0]
                run_name = run_command[run_command.index('--name') + 1]
                self.assertRegex(run_name, r'^rx-env-inventory-[0-9a-f]{32}$')
                self.assertEqual(runner.call_args_list[1].args[0], ['docker', 'rm', '--force', run_name])

    def test_invalid_json_and_inventory_shapes_are_rejected_and_cleaned(self):
        invalid_mapping = self.result()
        invalid_mapping['python_packages'] = {'torch': 2.9}
        for output in ('not JSON', '[]', '{"source":"image","source":"other"}',
                       '{"source":NaN}', json.dumps(invalid_mapping)):
            with self.subTest(output=output), patch.object(inventory.subprocess, 'run', return_value=
                    subprocess.CompletedProcess([], 0, output, '')) as docker:
                with self.assertRaisesRegex(RuntimeError, '版本信息无效'):
                    inventory.collect_image_inventory(self.image_id)
                self.assertEqual(docker.call_count, 2)
                self.assertEqual(docker.call_args.args[0][:3], ['docker', 'rm', '--force'])


if __name__ == '__main__':
    unittest.main()
