"""Check an installed application through the same paths used by Docker users.

The release packager embeds this file in install.sh; it is not a user-facing script.
"""
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

root = Path('/opt/rx-met')
manifest = json.loads((root / 'manifest.json').read_text())
expected = manifest['compatibility']
assert platform.machine() == 'x86_64', platform.machine()
assert sysconfig.get_config_var('SOABI') == expected['python_abi']
environment = dict(line.split('=', 1) for line in Path('/opt/rx-quant-env/environment.env').read_text().splitlines())
assert environment['environment_name'] == manifest['environment_name'] == 'rx-quant-env'
assert environment['config_sha256'] == manifest['runtime']['config_sha256']
assert environment['environment_role'] == 'runtime'
assert environment['environment_version'] == manifest['environment_version']
assert environment['cuda_variant'] == manifest['variant']
for name, version in expected['packages'].items():
    assert metadata.version(name) == version, (name, metadata.version(name), version)
for name in ('rx-met', 'quant-gru', 'quant-lstm'):
    assert metadata.version(name) == manifest['wheel_version'], (name, metadata.version(name))
for name in ('aimet_common', 'aimet_onnx', 'aimet_torch', 'quant_gru', 'quant_lstm', '_quant_lstm', 'gru_interface_binding'):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(root / 'python'), (name, module.__file__)
import torch
assert torch.version.cuda == expected['cuda_version']
from quant_lstm import QuantLSTM, _DEFAULT_CONFIG_PATH
assert _DEFAULT_CONFIG_PATH.is_file(), _DEFAULT_CONFIG_PATH
QuantLSTM(4, 8, batch_first=True)
from aimet_onnx.rx_ptq import DEFAULT_QUANTSIM_CONFIG, DEFAULT_BITWIDTH_CONFIG
for path in (DEFAULT_QUANTSIM_CONFIG, DEFAULT_BITWIDTH_CONFIG):
    assert path.is_file(), path
    json.loads(path.read_text())
patterns = ('aimet_common/_libpymo*.so', 'aimet_common/libquant_info*.so',
            'aimet_common/libaimet_onnxrt_ops.so', 'gru_interface_binding*.so',
            'lib/libgru_quant_shared.so', '_quant_lstm*.so')
for pattern in patterns:
    libraries = list((root / 'python').glob(pattern))
    assert libraries, pattern
    for library in libraries:
        output = subprocess.check_output(['ldd', str(library)], text=True)
        missing = [line for line in output.splitlines() if 'not found' in line and 'libcuda.so.1' not in line]
        assert not missing, (library, missing)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
print('Installation verified:', manifest['bundle_name'], 'from', root / 'python')
