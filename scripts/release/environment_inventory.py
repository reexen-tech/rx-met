"""Read installed environment versions from an exact Docker image."""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid


_COLLECT_SCRIPT = r'''
import importlib.metadata
import json
import os
import platform
import re
import subprocess

python_packages = {}
for distribution in importlib.metadata.distributions():
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
    # Preserve sys.path precedence when duplicate distribution metadata exists.
    python_packages.setdefault(name, distribution.version)

result = subprocess.run(
    ["dpkg-query", "--show", "--showformat=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n"],
    check=True, capture_output=True, text=True, timeout=20,
)
system_packages = {}
for line in result.stdout.splitlines():
    name, version, status = line.split("\t")
    if status[1:2] == "i":
        system_packages[name] = version

os_release = platform.freedesktop_os_release()
components = {name: python_packages.get(name) for name in (
    "torch", "torchvision", "torchaudio", "onnx", "onnxruntime", "onnxruntime-gpu", "numpy",
)}
components["cuda"] = {
    "version": os.environ.get("CUDA_VERSION") or None,
    "python_packages": {name: version for name, version in python_packages.items()
                        if name.startswith("nvidia-cuda-runtime-cu")},
    "system_packages": {name: version for name, version in system_packages.items()
                        if name.startswith(("cuda-cudart", "libcudart"))},
}
components["cudnn"] = {
    "python_packages": {name: version for name, version in python_packages.items()
                        if name.startswith("nvidia-cudnn-cu")},
    "system_packages": {name: version for name, version in system_packages.items()
                        if name.startswith("libcudnn")},
}
print(json.dumps({
    "source": "image",
    "os": {name.lower(): os_release.get(name, "")
           for name in ("ID", "NAME", "VERSION_ID", "PRETTY_NAME")},
    "architecture": platform.machine(),
    "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
    "components": components,
    "python_packages": python_packages,
    "system_packages": system_packages,
}, sort_keys=True))
'''


def _string_mapping(value):
    return isinstance(value, dict) and all(isinstance(key, str) and isinstance(version, str)
                                          for key, version in value.items())


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'invalid JSON constant: {value}')


def _parse_inventory(output):
    try:
        inventory = json.loads(output, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        if not isinstance(inventory, dict) or inventory.get('source') != 'image':
            raise ValueError('invalid inventory source')
        if not _string_mapping(inventory.get('os')) or not isinstance(inventory.get('architecture'), str):
            raise ValueError('invalid operating system information')
        python = inventory.get('python')
        if not _string_mapping(python) or not python.get('version') or not python.get('implementation'):
            raise ValueError('invalid Python information')
        for key in ('python_packages', 'system_packages'):
            if not _string_mapping(inventory.get(key)):
                raise ValueError(f'invalid {key}')
        components = inventory.get('components')
        if not isinstance(components, dict):
            raise ValueError('invalid components')
        for name in ('torch', 'torchvision', 'torchaudio', 'onnx', 'onnxruntime', 'onnxruntime-gpu', 'numpy'):
            if name not in components or (components[name] is not None and not isinstance(components[name], str)):
                raise ValueError(f'invalid component: {name}')
        for name in ('cuda', 'cudnn'):
            component = components.get(name)
            if not isinstance(component, dict) or any(not _string_mapping(component.get(key))
                                                     for key in ('python_packages', 'system_packages')):
                raise ValueError(f'invalid component: {name}')
        if 'version' not in components['cuda'] or (components['cuda']['version'] is not None
                                                  and not isinstance(components['cuda']['version'], str)):
            raise ValueError('invalid CUDA version')
    except (ValueError, TypeError) as error:
        raise RuntimeError(f'镜像环境版本信息无效: {error}') from error
    return inventory


def collect_image_inventory(image_id: str) -> dict:
    """Collect installed packages without network access, GPUs, or image writes."""
    if not isinstance(image_id, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
        raise RuntimeError('采集环境版本必须使用完整 Docker 镜像 ID')
    name = 'rx-env-inventory-' + uuid.uuid4().hex
    command = ['docker', 'run', '--rm', '--interactive', '--name', name, '--pull', 'never',
               '--network', 'none', '--read-only', '--cpus', '1',
               '--memory', '1g', '--memory-swap', '1g', '--pids-limit', '64',
               '--user', f'{os.getuid()}:{os.getgid()}', '--group-add', '986',
               '--tmpfs', '/tmp:rw,nosuid,size=64m']
    for setting in ('HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1', 'NVIDIA_VISIBLE_DEVICES=void',
                    'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                    'NUMEXPR_NUM_THREADS=1'):
        command.extend(['--env', setting])
    command.extend(['--entrypoint', 'python3', image_id, '-I', '-B', '-'])
    completed = False
    try:
        try:
            result = subprocess.run(command, input=_COLLECT_SCRIPT, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(f'无法采集镜像 {image_id} 的环境版本: {error}') from error
        if result.returncode:
            raise RuntimeError(f'无法采集镜像 {image_id} 的环境版本: {result.stderr.strip()[:2000]}')
        inventory = _parse_inventory(result.stdout)
        completed = True
        return inventory
    finally:
        if not completed:
            try:
                subprocess.run(['docker', 'rm', '--force', name], capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.SubprocessError):
                pass
