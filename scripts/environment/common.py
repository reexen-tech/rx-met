#!/usr/bin/env python3
"""Public rx-quant GPU environment contract; kept independently in both products."""
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


def config_digest():
    h = hashlib.sha256(json.dumps(config(), sort_keys=True, separators=(',', ':')).encode())
    for name in ('docker/Dockerfile.environment', 'scripts/environment/common.py',
                 'scripts/internal/prepare_onnxruntime_headers.sh', 'scripts/internal/verify_dependency_wheelhouse.py'):
        h.update(name.encode() + b'\0' + bytes.fromhex(sha256(ROOT / name)))
    for path in sorted((ROOT / 'native/third_party/onnxruntime').rglob('*')):
        if path.is_file():
            h.update(str(path.relative_to(ROOT)).encode() + bytes.fromhex(sha256(path)))
    for path in sorted((ROOT / 'native/third_party/onnxruntime-prepared').rglob('*')):
        if path.is_file():
            h.update(str(path.relative_to(ROOT)).encode() + bytes.fromhex(sha256(path)))
    return h.hexdigest()


def packages(matrix, variant):
    return {**matrix['common_packages'], **matrix['variants'][variant]['torch']['packages'],
            **matrix['variants'][variant]['onnxruntime']['packages']}


def image_ref(role, variant):
    return f"{NAME}-{role}:{config()['environment_version']}-{variant}"


def inspect(reference, role, variant):
    info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', reference]))[0]
    expected = {'environment.name': NAME, 'environment.version': config()['environment_version'],
                'environment.role': role, 'environment.config-sha256': config_digest(), 'cuda.variant': variant}
    labels = info['Config'].get('Labels') or {}
    for key, value in expected.items():
        if labels.get(LABEL + key) != value:
            raise RuntimeError(f'{reference}: {key} does not match this public environment definition')
    if (info['Os'], info['Architecture']) != ('linux', 'amd64'):
        raise RuntimeError(f'{reference}: expected linux/amd64')
    return info


def load_record(variant, record_path=None):
    path = Path(record_path or ROOT / 'docker/environment-images.json')
    doc = json.loads(path.read_text())
    if (doc['environment_name'], doc['environment_version'], doc['config_sha256']) != (NAME, config()['environment_version'], config_digest()):
        raise RuntimeError('Published environment record does not match local common configuration')
    return doc['variants'][variant]


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
        if gpu:
            command += ['--gpus', 'device=' + os.environ.get('RX_QUANT_GPU_DEVICE', '0')]
        command += ['--entrypoint', 'python3', info['Id'], '-B', '-']
        code = f'''import importlib.metadata as m,sys,subprocess
import torch,onnxruntime,transformers,datasets,gguf
expected={expected!r}
for name,version in expected.items(): assert m.version(name)==version,(name,m.version(name),version)
assert f"{{sys.version_info.major}}.{{sys.version_info.minor}}"=={spec['python_version']!r}
assert torch.version.cuda=={spec['cuda_version']!r}
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
subprocess.run([sys.executable,'-m','pip','check'],check=True)
if {gpu!r}:
 assert torch.cuda.is_available()
 assert (torch.tensor([2.],device='cuda')*3).item()==6
print({variant!r},{role!r},"public environment verified")
'''
        run(*command, input=code, text=True)


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
    verifier.verify(output, [ROOT / 'requirements/build.lock', ROOT / 'requirements/runtime.lock'])
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in sorted(output.glob('*.whl'))))


def build(variant, wheelhouse=None, builder=None):
    matrix = config(); spec = matrix['variants'][variant]
    record_path = ROOT / 'docker/environment-images.json'
    published = None
    if record_path.is_file():
        document = json.loads(record_path.read_text())
        if variant in document.get('variants', {}): published = load_record(variant)
    if published:
        for role in ('build', 'runtime'):
            info = inspect(published[role + '_image_id'], role, variant)
            if info['Id'] != published[role + '_image_id']: raise RuntimeError('Published image identity mismatch')
        print('Reuse published environment:', variant, flush=True)
        return
    context = ROOT / '.release/environment-context' / matrix['environment_version'] / variant
    context.mkdir(parents=True, exist_ok=True)
    for name in ('docker/Dockerfile.environment', 'scripts/environment/common.py',
                 'scripts/internal/prepare_onnxruntime_headers.sh', 'scripts/internal/verify_dependency_wheelhouse.py'):
        destination = context / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copytree(ROOT / 'native/third_party/onnxruntime', context / 'native/third_party/onnxruntime', dirs_exist_ok=True)
    shutil.copytree(ROOT / 'native/third_party/onnxruntime-prepared', context / 'native/third_party/onnxruntime-prepared', dirs_exist_ok=True)
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
            'ENV_VERSION': matrix['environment_version'], 'ENV_CONFIG_SHA256': config_digest()}
    for role, target in [('build', 'build-env'), ('runtime', 'runtime-env')]:
        reference = image_ref(role, variant)
        result = subprocess.run(['docker', 'image', 'inspect', reference], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            inspect(reference, role, variant)
            print('Reuse matching candidate:', reference, flush=True)
            continue
        command = ['docker', 'buildx', 'build', '--load', '--progress=plain', '--target', target, '-t', reference, '-f', str(context / 'docker/Dockerfile.environment')]
        if builder: command += ['--builder', builder]
        for key, value in args.items(): command += ['--build-arg', f'{key}={value}']
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
            if os.environ.get(key): command += ['--build-arg', key]
        run(*command, context)
        inspect(reference, role, variant)


def export(variant, directory):
    matrix = config()
    build_info = inspect(image_ref('build', variant), 'build', variant)
    runtime = inspect(image_ref('runtime', variant), 'runtime', variant)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{NAME}-runtime-{matrix['environment_version']}-{variant}-linux-amd64.tar.gz"
    archive = directory / name; sidecar = directory / (name + '.json')
    if archive.exists() or sidecar.exists():
        if not archive.is_file() or not sidecar.is_file(): raise RuntimeError('Incomplete existing runtime archive')
        record = json.loads(sidecar.read_text())
        if record['image_id'] != runtime['Id'] or record['sha256'] != sha256(archive): raise RuntimeError('Refuse to overwrite published runtime')
    else:
        partial = archive.with_name(archive.name + '.partial')
        with partial.open('xb') as output:
            saver = subprocess.Popen(['docker', 'save', runtime['Id']], stdout=subprocess.PIPE)
            try:
                compressor = ['pigz', '-p', '2', '-1', '-n'] if shutil.which('pigz') else ['gzip', '-1', '-n']
                compressed = subprocess.run(compressor, stdin=saver.stdout, stdout=output)
                saver.stdout.close()
                if compressed.returncode or saver.wait(): raise RuntimeError('Runtime archive export failed')
            finally:
                if saver.poll() is None: saver.terminate(); saver.wait()
        run('gzip', '-t', partial)
        partial.replace(archive)
        record = {'schema_version': 1, 'environment_name': NAME, 'environment_version': matrix['environment_version'],
                  'variant': variant, 'config_sha256': config_digest(), 'image_id': runtime['Id'],
                  'repo_digests': runtime.get('RepoDigests', []), 'filename': name, 'sha256': sha256(archive),
                  'size_bytes': archive.stat().st_size, 'url': ''}
        write_json(sidecar, record)
    lock_path = ROOT / 'docker/environment-images.json'
    document = {'schema_version': 1, 'environment_name': NAME, 'environment_version': matrix['environment_version'],
                'config_sha256': config_digest(), 'variants': {}}
    if lock_path.is_file():
        document = json.loads(lock_path.read_text())
        if (document['config_sha256'], document['environment_version']) != (config_digest(), matrix['environment_version']):
            raise RuntimeError('Existing environment publication belongs to a different configuration')
    entry = {'build_image_id': build_info['Id'], 'runtime_image_id': runtime['Id'], 'runtime': record}
    if variant in document['variants'] and document['variants'][variant] != entry:
        raise RuntimeError('Published environment entry cannot change')
    document['variants'][variant] = entry
    write_json(lock_path, document)
    write_json(directory / 'manifest.json', document)
    (directory / 'README.md').write_text(f'# rx-quant 公共 runtime {matrix["environment_version"]}\n\n供 rx-met 与 rx-lm-quant 应用包使用。将对应归档放入应用包旁的 images/，运行应用包 install.sh。\n\n镜像身份、配置摘要和归档校验值见 manifest.json。\n')
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
    (directory / 'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.name}\n' for p in files))
    print('Published environment record:', lock_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'export', 'load', 'digest', 'download'])
    parser.add_argument('variants', nargs='*', choices=['cu118', 'cu126', 'cu130'])
    parser.add_argument('--output-dir', '--export-dir', dest='output_dir', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--wheelhouse-dir', type=Path)
    parser.add_argument('--builder')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--no-export', action='store_true', help='Compatibility option; build never exports implicitly')
    args = parser.parse_intermixed_args()
    if args.action == 'digest': print(config_digest()); return
    directory = args.output_dir or ROOT / '.release/runtime-images' / config()['environment_version']
    for variant in args.variants or config()['variants']:
        if args.action == 'build':
            build(variant, args.wheelhouse_dir / variant if args.wheelhouse_dir else None, args.builder)
            verify(variant, args.gpu)
        elif args.action == 'verify': verify(variant, args.gpu)
        elif args.action == 'export': export(variant, directory)
        elif args.action == 'download': download(variant, directory)
        else:
            if not args.directory: parser.error('load requires --directory')
            record = load_record(variant)['runtime']
            archive = args.directory / record['filename']
            if sha256(archive) != record['sha256']: raise RuntimeError('Runtime archive checksum mismatch')
            with subprocess.Popen(['gzip', '-dc', str(archive)], stdout=subprocess.PIPE) as decompressor:
                run('docker', 'load', stdin=decompressor.stdout)
                decompressor.stdout.close()
                if decompressor.wait(): raise RuntimeError('Runtime archive decompression failed')
            inspect(record['image_id'], 'runtime', variant)


if __name__ == '__main__':
    try: main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'public environment failed: {error}') from None
