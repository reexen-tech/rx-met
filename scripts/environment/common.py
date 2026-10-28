#!/usr/bin/env python3
"""Public rx-quant CPU/CUDA environment contract."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'rx-quant-env'
LABEL = 'com.reexen.rx-quant.'
THIRD_PARTY = 'docker/environment/third_party'
BUILD_FILES = ('docker/Dockerfile.environment', 'scripts/environment/common.py',
               'scripts/environment/patch_onnxsim_wheel.py',
               'scripts/environment/verify_onnxsim_shutdown.py',
               'scripts/environment/patch_torch_wheel.py',
               'scripts/environment/patches/torch_fx_source_cache.py',
               'scripts/environment/verify_fx_source_cache.py',
               'scripts/internal/prepare_onnxruntime_headers.sh', 'scripts/internal/verify_dependency_wheelhouse.py')


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def config():
    matrix = json.loads((ROOT / 'docker/variants.json').read_text())
    return {key: matrix[key] for key in ('schema_version', 'environment_name', 'environment_version', 'build_packages', 'common_packages', 'variants')}


def default_variants(matrix):
    return [variant for variant in matrix['variants'] if variant != 'cpu']


def config_digest():
    h = hashlib.sha256(json.dumps(config(), sort_keys=True, separators=(',', ':')).encode())
    for name in BUILD_FILES:
        h.update(name.encode() + b'\0' + bytes.fromhex(sha256(ROOT / name)))
    for path in sorted((ROOT / THIRD_PARTY).rglob('*')):
        if path.is_file():
            h.update(str(path.relative_to(ROOT)).encode() + bytes.fromhex(sha256(path)))
    return h.hexdigest()


def packages(matrix, variant):
    return {**matrix['common_packages'], **matrix['variants'][variant]['torch']['packages'],
            **matrix['variants'][variant]['onnxruntime']['packages']}


def image_ref(role, variant):
    return f"{NAME}-{role}:{config()['environment_version']}-{variant}"


def source_revision():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def inspect(reference, role, variant):
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', reference]))[0]
    variant_label = 'platform.variant' if variant == 'cpu' else 'cuda.variant'
    expected = {'environment.name': NAME, 'environment.version': config()['environment_version'],
                'environment.role': role, variant_label: variant}
    labels = info['Config'].get('Labels') or {}
    for key, value in expected.items():
        if labels.get(LABEL + key) != value:
            raise RuntimeError(f'{reference}: {key} does not match this public environment definition')
    if (info['Os'], info['Architecture']) != ('linux', 'amd64'):
        raise RuntimeError(f'{reference}: expected linux/amd64')
    return info


def _verify_lifecycle(image, script, label, container_prefix):
    name = f'rx-met-{container_prefix}-verify-{os.getpid()}-{os.urandom(6).hex()}'
    command = ['docker', 'run', '--rm', '--name', name, '-i', '--network', 'none', '--read-only',
               '--cpus', '1', '--memory', '2g', '--memory-swap', '2g', '--pids-limit', '192',
               '--tmpfs', '/tmp:rw,exec,size=256m', '-e', 'HOME=/tmp',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=1',
               '-e', 'MKL_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
               '--entrypoint', 'python3', image, '-B', '-']
    code = 'import signal\nsignal.alarm(60)\n' + (ROOT / script).read_text()
    try:
        result = subprocess.run(command, input=code, text=True, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired as error:
        for stream, output in ((sys.stdout, error.stdout), (sys.stderr, error.stderr)):
            if output:
                stream.write(output.decode(errors='replace') if isinstance(output, bytes) else output)
        try:
            cleanup = subprocess.run(['docker', 'rm', '-f', name], text=True, capture_output=True, timeout=10)
            sys.stdout.write(cleanup.stdout)
            sys.stderr.write(cleanup.stderr)
            removed = cleanup.returncode == 0 or 'No such container' in cleanup.stderr
        except (OSError, subprocess.TimeoutExpired):
            removed = False
        cleanup_status = 'container removed' if removed else f'container cleanup not confirmed: {name}'
        raise RuntimeError(f'{image}: {label} verification timed out; {cleanup_status}') from error
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode or 'nanobind: leaked' in result.stderr:
        raise RuntimeError(f'{image}: {label} verification failed '
                           f'(exit code {result.returncode}; inspect stderr above)')


def verify_onnxsim_shutdown(image):
    _verify_lifecycle(image, 'scripts/environment/verify_onnxsim_shutdown.py',
                      'onnxsim shutdown', 'onnxsim')


def verify_fx_source_cache(image):
    _verify_lifecycle(image, 'scripts/environment/verify_fx_source_cache.py',
                      'FX source-cache', 'fx-source-cache')


def verify(variant, gpu=False):
    matrix = config()
    spec = matrix['variants'][variant]
    for role in ('build', 'runtime'):
        info = inspect(image_ref(role, variant), role, variant)
        expected = packages(matrix, variant)
        if role == 'build':
            expected.update(matrix['build_packages'])
        command = ['docker', 'run', '--rm', '-i', '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,exec,size=2g',
                   '-e', 'HOME=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1']
        if gpu and variant != 'cpu':
            command += ['--gpus', 'device=' + os.environ.get('RX_QUANT_GPU_DEVICE', '0')]
        command += ['--entrypoint', 'python3', info['Id'], '-B', '-']
        code = f'''import importlib.metadata as m,sys,subprocess
from pathlib import Path
import torch,onnxruntime,transformers,datasets,gguf
metadata=dict(line.split('=',1) for line in Path('/opt/rx-quant-env/environment.env').read_text().splitlines())
for key,value in {{'environment_name':{NAME!r},'environment_version':{matrix['environment_version']!r},'environment_role':{role!r},'cuda_variant':{variant!r}}}.items():
 assert metadata.get(key)==value,(key,metadata.get(key),value)
expected={expected!r}
for name,version in expected.items(): assert m.version(name)==version,(name,m.version(name),version)
assert f"{{sys.version_info.major}}.{{sys.version_info.minor}}"=={spec['python_version']!r}
assert torch.version.cuda=={(spec['cuda_version'] or None)!r}
assert {('CPUExecutionProvider' if variant == 'cpu' else 'CUDAExecutionProvider')!r} in onnxruntime.get_available_providers()
subprocess.run([sys.executable,'-m','pip','check'],check=True)
if {bool(gpu and variant != 'cpu')!r}:
 assert torch.cuda.is_available()
 assert (torch.tensor([2.],device='cuda')*3).item()==6
print({variant!r},{role!r},"public environment verified")
'''
        run(*command, input=code, text=True)
        verify_onnxsim_shutdown(info['Id'])
        verify_fx_source_cache(info['Id'])


def download(variant, output):
    matrix = config(); spec = matrix['variants'][variant]
    output.mkdir(parents=True, exist_ok=True)
    verifier_path = ROOT / 'scripts/internal/verify_dependency_wheelhouse.py'
    module_spec = importlib.util.spec_from_file_location('wheelhouse', verifier_path)
    verifier = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(verifier)
    groups = [('build', matrix['build_packages'], None, []), ('common', matrix['common_packages'], None, []),
              ('torch', spec['torch']['packages'], spec['torch']['index_url'], spec['torch'].get('extra_index_urls', [])),
              ('ort', spec['onnxruntime']['packages'], spec['onnxruntime']['index_url'], [])]
    expected = {**packages(matrix, variant), **matrix['build_packages']}
    found = {}
    for path in output.glob('*.whl'):
        name, version = verifier.wheel_metadata(path)
        if expected.get(name) != version or name in found:
            raise RuntimeError(f'Unexpected cached wheel: {path}')
        found[name] = version
    with tempfile.TemporaryDirectory() as temporary:
        for group, values, index, extras in groups:
            missing = {name: version for name, version in values.items() if name not in found}
            if not missing:
                continue
            requirements = Path(temporary) / (group + '.txt')
            requirements.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(missing.items())))
            command = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps', '--dest', str(output), '-r', str(requirements)]
            if index: command += ['--index-url', index]
            for extra in extras: command += ['--extra-index-url', extra]
            run(*command)
    patch_spec = importlib.util.spec_from_file_location('onnxsim_wheel_patch', ROOT / 'scripts/environment/patch_onnxsim_wheel.py')
    patcher = importlib.util.module_from_spec(patch_spec); patch_spec.loader.exec_module(patcher)
    for path in output.glob('onnxsim-*.whl'):
        patcher.patch_wheel(path)
    patch_spec = importlib.util.spec_from_file_location('torch_wheel_patch', ROOT / 'scripts/environment/patch_torch_wheel.py')
    patcher = importlib.util.module_from_spec(patch_spec); patch_spec.loader.exec_module(patcher)
    for path in output.glob('torch-*.whl'):
        patcher.patch_wheel(path)
    verifier.verify(output, [ROOT / 'requirements/build.lock', ROOT / 'requirements/runtime.lock'])
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(output.glob('*.whl'))))


def build(variant, wheelhouse=None, builder=None):
    matrix = config(); spec = matrix['variants'][variant]
    missing = []
    for role in ('build', 'runtime'):
        reference = image_ref(role, variant)
        result = subprocess.run(['docker', 'image', 'inspect', reference], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            inspect(reference, role, variant)
            print('Reuse environment:', reference, flush=True)
        else:
            missing.append(role)
    if not missing:
        return
    context = ROOT / '.release/environment-context' / matrix['environment_version'] / variant
    context.mkdir(parents=True, exist_ok=True)
    for name in BUILD_FILES:
        destination = context / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copytree(ROOT / THIRD_PARTY, context / THIRD_PARTY, dirs_exist_ok=True)
    write_json(context / 'docker/variants.json', matrix)
    requirements = context / 'requirements'; requirements.mkdir(exist_ok=True)
    for name, values in [('build', matrix['build_packages']), ('runtime', packages(matrix, variant))]:
        (requirements / (name + '.lock')).write_text(''.join(f'{key}=={value}\n' for key, value in sorted(values.items())))
    cache = context / 'wheelhouse'; cache.mkdir(exist_ok=True)
    if wheelhouse:
        for source in Path(wheelhouse).glob('*.whl'):
            target = cache / source.name
            if not target.exists():
                try: os.link(source, target)
                except OSError: shutil.copy2(source, target)
    args = {'DEVEL_IMAGE': spec['devel_image'], 'BASE_IMAGE': spec['base_image'], 'CUDA_VARIANT': variant,
            'ENV_VERSION': matrix['environment_version'], 'ENV_CONFIG_SHA256': config_digest(),
            'ENV_SOURCE_REVISION': source_revision()}
    for role in missing:
        target = role + '-env'
        reference = image_ref(role, variant)
        command = ['docker', 'buildx', 'build', '--load', '--progress=plain', '--target', target, '-t', reference, '-f', str(context / 'docker/Dockerfile.environment')]
        if builder: command += ['--builder', builder]
        for key, value in args.items(): command += ['--build-arg', f'{key}={value}']
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
            if os.environ.get(key): command += ['--build-arg', key]
        run(*command, context)
        inspect(reference, role, variant)


def export(variant, directory, role='runtime'):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from environment.runtime_archives import export_environment
    export_environment(variant, directory, role)


def main():
    matrix = config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'export', 'load', 'digest', 'download'])
    parser.add_argument('variants', nargs='*', choices=list(matrix['variants']),
                        help='Environment variants (default: CUDA variants; CPU requires cpu)')
    parser.add_argument('--output-dir', '--export-dir', dest='output_dir', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--role', choices=['runtime', 'build', 'all'], default='runtime',
                        help='Image role for export/load (default: runtime)')
    parser.add_argument('--wheelhouse-dir', type=Path)
    parser.add_argument('--builder')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--no-export', action='store_true', help='Compatibility option; build never exports implicitly')
    args = parser.parse_intermixed_args()
    if args.role != 'runtime' and args.action not in ('export', 'load'):
        parser.error('--role only applies to export/load')
    if args.action == 'digest': print(config_digest()); return
    directory = args.output_dir or ROOT / '.release/runtime-images' / matrix['environment_version']
    for variant in args.variants or default_variants(matrix):
        if args.action == 'build':
            build(variant, args.wheelhouse_dir / variant if args.wheelhouse_dir else None, args.builder)
            verify(variant, args.gpu)
        elif args.action == 'verify': verify(variant, args.gpu)
        elif args.action == 'export':
            for role in ('build', 'runtime') if args.role == 'all' else (args.role,):
                export(variant, directory, role)
        elif args.action == 'download': download(variant, directory)
        else:
            if not args.directory: parser.error('load requires --directory')
            sys.path.insert(0, str(ROOT / 'scripts'))
            from environment.runtime_archives import load_environment
            for role in ('build', 'runtime') if args.role == 'all' else (args.role,):
                load_environment(variant, args.directory, role)


if __name__ == '__main__':
    try: main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'public environment failed: {error}') from None
