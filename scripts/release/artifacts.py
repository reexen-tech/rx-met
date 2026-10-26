#!/usr/bin/env python3
"""Build project-only bundles and export immutable runtime archives."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import dependencies
from environment import common
from release import delivery, environment_archives


def run(*args, **kwargs):
    print('+', shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def image_info(reference, role, variant, env_version):
    return common.inspect(reference, role, variant)



def source_digest():
    paths = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard'], cwd=ROOT).decode().splitlines()
    digest = hashlib.sha256()
    for name in sorted(set(paths)):
        if name.startswith(('src/', 'native/', 'operators/', 'scripts/internal/')) or name in ('VERSION', 'pyproject.toml', 'setup.py', 'scripts/build_aimet_native.sh'):
            path = ROOT / name
            if path.is_file():
                digest.update(name.encode() + b'\0' + bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def release_changelog(version):
    """Render exactly one release from the repository's versioned changelog."""
    document = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    headings = list(re.finditer(r'^##[ \t]+\[([^\]\n]+)\][^\n]*$', document, re.MULTILINE))
    matches = [index for index, heading in enumerate(headings) if heading[1] == version]
    if version == 'Unreleased' or len(matches) != 1:
        raise RuntimeError(f'CHANGELOG.md 必须包含唯一的正式版本 [{version}]，找到 {len(matches)} 处')
    index, = matches
    heading = headings[index]
    end = headings[index + 1].start() if index + 1 < len(headings) else len(document)
    body = document[heading.end():end]
    if not any(line.strip() and not re.match(r'^#{1,6}(?:\s|$)', line.strip()) for line in body.splitlines()):
        raise RuntimeError(f'CHANGELOG.md 中版本 [{version}] 的内容为空')
    return '# Changelog\n\n' + document[heading.start():end].strip() + '\n'


def delivery_digest(changelog):
    digest = hashlib.sha256()
    digest.update(b'CHANGELOG.md' + hashlib.sha256(changelog.encode('utf-8')).digest())
    paths = [ROOT / 'packaging/release-bundle/install.sh.in', ROOT / 'packaging/release-bundle/README.md.in', ROOT / 'scripts/release/verify_install.py', ROOT / 'scripts/release/verify_runtime_archive.py']
    paths += sorted(p for p in (ROOT / 'examples').rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'output' not in p.parts)
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def runtime_compatibility(image_id, matrix, variant):
    packages = dependencies.runtime_packages(matrix, variant)
    code = "import json,sysconfig,importlib.metadata as m; print(json.dumps({'python_abi':sysconfig.get_config_var('SOABI'),'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))"
    actual = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python3', image_id, '-B', '-c', code]))
    for name, version in packages.items():
        if actual['packages'].get(name) != version:
            raise RuntimeError(f'{variant}: runtime {name}={actual["packages"].get(name)}, expected {version}')
    spec = matrix['variants'][variant]
    return {'python_abi': actual['python_abi'], 'python_version': spec['python_version'], 'cuda_version': spec['cuda_version'] or None,
            'platform': 'linux/amd64', 'packages': packages, 'cuda_architectures': spec['cuda_architectures'],
            'minimum_driver_version': spec['minimum_driver_version']}


def inspect_wheels(directory, wheel_version, python_version):
    wheels = []
    names = set()
    tag = 'cp' + python_version.replace('.', '')
    for path in sorted(directory.glob('*.whl')):
        with zipfile.ZipFile(path) as wheel:
            metadata_path, = [name for name in wheel.namelist() if name.endswith('.dist-info/METADATA')]
            metadata = BytesParser().parsebytes(wheel.read(metadata_path))
            name = metadata['Name'].lower().replace('_', '-')
            if name in names or name not in ('rx-met', 'quant-gru', 'quant-lstm') or metadata['Version'] != wheel_version:
                raise RuntimeError(f'意外的 wheel: {path.name}')
            if not path.name.endswith(f'-{tag}-{tag}-linux_x86_64.whl'):
                raise RuntimeError(f'wheel ABI 不匹配: {path.name}')
            required = {'quant-lstm': ['quant_lstm_resources/lstm_quant_default_v1.json'],
                        'quant-gru': ['lib/libgru_quant_shared.so'],
                        'rx-met': ['aimet_onnx/rx_ptq/config/quick_start_full_quant.json']}[name]
            if not all(item in wheel.namelist() for item in required):
                raise RuntimeError(f'wheel 缺少资源: {path.name}')
            unpacked_size = sum(item.file_size for item in wheel.infolist())
            names.add(name)
        wheels.append({'name': name, 'filename': path.name, 'sha256': sha256(path), 'size_bytes': path.stat().st_size, 'unpacked_size_bytes': unpacked_size})
    if names != {'rx-met', 'quant-gru', 'quant-lstm'}:
        raise RuntimeError(f'需要且仅需要三个项目 wheel: {directory}')
    return wheels


def runtime_requirement(record, url=''):
    """Require a compatible environment tag, independently of local image IDs."""
    keys = ('schema_version', 'environment_name', 'environment_version', 'variant',
            'image_tag', 'filename')
    return {**{key: record[key] for key in keys}, 'url': url,
            'checksum_url': url + '.sha256' if url else ''}


def bundle_name(version, variant):
    return f'rx-met-v{version}-{variant}-linux-amd64'


def assemble(args, matrix, variant, runtime, build, runtime_record, archive, wheel_dir, compatibility, *, version, changelog):
    env_version = matrix['environment_version']
    runtime_tag = common.image_ref('runtime', variant)
    runtime_record = {**runtime_record, 'image_tag': runtime_tag}
    name = bundle_name(version, variant)
    packed = args.output_dir / (name + '.tar.gz')
    if packed.exists() and not args.replace:
        raise RuntimeError(f'应用产物已存在: {packed}；开发时可显式 --replace')
    wheel_version = f'{version}+{variant}'
    wheels = inspect_wheels(wheel_dir, wheel_version, compatibility['python_version'])
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        revision += '-dirty'
    manifest = {'schema_version': 1, 'bundle_name': name, 'version': version, 'wheel_version': wheel_version,
                'environment_name': common.NAME, 'environment_version': env_version, 'variant': variant, 'source_revision': revision,
                'wheel_source_sha256': source_digest(), 'delivery_sources_sha256': delivery_digest(changelog), 'build_image_id': build['Id'], 'runtime_image_id': runtime['Id'],
                'runtime': runtime_requirement(runtime_record, (args.runtime_base_url.rstrip('/') + '/' + archive.name) if args.runtime_base_url else ''),
                'compatibility': compatibility, 'wheels': wheels}
    staging = ROOT / '.release/staging'
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bundle-', dir=staging) as temporary:
        staged = Path(temporary) / name
        (staged / 'wheels').mkdir(parents=True)
        for wheel in wheels:
            shutil.copy2(wheel_dir / wheel['filename'], staged / 'wheels' / wheel['filename'])
        shutil.copytree(ROOT / 'examples', staged / 'examples', ignore=shutil.ignore_patterns('__pycache__', 'output', '*.pyc'))
        write_json(staged / 'manifest.json', manifest)
        bootstrap = {'BUNDLE_NAME': name, 'MANIFEST_SHA256': sha256(staged / 'manifest.json'),
                     'RUNTIME_ENV_DIR': f"{runtime_record['environment_name']}-{env_version}",
                     'RUNTIME_VARIANT': variant, 'RUNTIME_ENV_NAME': runtime_record['environment_name'], 'RUNTIME_ENV_VERSION': env_version,
                     'RUNTIME_LABEL_PREFIX': common.LABEL,
                     'RUNTIME_VARIANT_LABEL': 'platform.variant' if variant == 'cpu' else 'cuda.variant',
                     'RUNTIME_IMAGE_TAG': runtime_tag, 'RUNTIME_ARCHIVE': archive.name,
                     'RUNTIME_URL': manifest['runtime']['url'], 'RUNTIME_CHECKSUM_URL': manifest['runtime']['checksum_url'],
                     'APP_INSTALL_BYTES': str(max(512 * 1024**2, sum(w['unpacked_size_bytes'] for w in wheels) * 2))}
        script = (ROOT / 'packaging/release-bundle/install.sh.in').read_text()
        script = script.replace('@BOOTSTRAP@', '\n'.join(f'readonly {key}={shlex.quote(value)}' for key, value in bootstrap.items()))
        script = script.replace('@VERIFY_RUNTIME_ARCHIVE@', (ROOT / 'scripts/release/verify_runtime_archive.py').read_text())
        script = script.replace('@VERIFY_INSTALL@', (ROOT / 'scripts/release/verify_install.py').read_text())
        (staged / 'install.sh').write_text(script)
        (staged / 'install.sh').chmod(0o755)
        readme = (ROOT / 'packaging/release-bundle/README.md.in').read_text()
        for key, value in {'VERSION': version, 'ENV_VERSION': env_version, 'ENV_DIRECTORY': f"{runtime_record['environment_name']}-{env_version}", 'VARIANT': variant, 'BUNDLE_NAME': name, 'RUNTIME_IMAGE_TAG': runtime_tag}.items():
            readme = readme.replace('@' + key + '@', value)
        (staged / 'README.md').write_text(readme)
        (staged / 'CHANGELOG.md').write_text(changelog, encoding='utf-8')
        checksums = ''.join(f'{sha256(path)}  {path.relative_to(staged).as_posix()}\n' for path in sorted(staged.rglob('*')) if path.is_file())
        (staged / 'SHA256SUMS').write_text(checksums)
        run('bash', '-n', staged / 'install.sh')
        install_prefix = args.install_prefix if args.install_prefix is not None else Path(temporary) / 'install'
        run(staged / 'install.sh', '--offline', '--prefix', install_prefix)
        with tempfile.TemporaryDirectory(prefix='.archive-', dir=args.output_dir) as archive_directory:
            temporary_archive = Path(archive_directory) / packed.name
            with tarfile.open(temporary_archive, 'w:gz', compresslevel=1) as tar:
                tar.add(staged, arcname=name)
            temporary_archive.replace(packed)
    print('Application bundle:', packed, packed.stat().st_size, 'bytes', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'export-runtime'])
    parser.add_argument('variants', nargs='*', help='默认选择全部 CUDA 变体；CPU 需显式指定 cpu')
    parser.add_argument('--output-dir', type=Path, help='最终产品交付目录；环境附件放在其父目录下的 <环境名>-<环境版本>/')
    parser.add_argument('--build-env-dir', type=Path, help='本地缺少 build 镜像时，从此目录校验并导入归档')
    parser.add_argument('--runtime-env-dir', type=Path, help='运行环境归档的导入、复用及导出缓存；默认 .release/runtime-images/<环境版本>')
    parser.add_argument('--runtime-base-url', default=os.environ.get('RX_MET_RUNTIME_BASE_URL', ''))
    parser.add_argument('--install-prefix', type=Path, help='保留安装验收结果的根目录；默认使用临时目录并自动清理')
    parser.add_argument('--with-env', action='store_true', help='将公共运行环境归档导出或复用到产品目录平级的环境目录；不附带构建环境')
    parser.add_argument('--skip-build', action='store_true', help='只重新组装源码和环境身份未变的已有 wheels')
    parser.add_argument('--replace', action='store_true', help='仅替换本次所选同名应用归档及校验文件；保留其他变体、安装目录和公共环境')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_intermixed_args()
    if args.action == 'export-runtime':
        raise RuntimeError('请使用 scripts/environment/export_runtime.sh 导出公共环境')
    matrix = dependencies.load_matrix()
    dependencies.check_generated_artifacts(matrix)
    targets = args.variants or common.default_variants(matrix)
    unsupported = sorted(set(targets) - set(matrix['variants']))
    if unsupported:
        parser.error('不支持的变体: ' + ', '.join(unsupported))
    if len(set(targets)) != len(targets):
        parser.error('CUDA 变体重复')
    if args.runtime_base_url and not args.runtime_base_url.startswith('https://'):
        parser.error('--runtime-base-url 必须使用 HTTPS')
    if args.jobs < 1:
        parser.error('--jobs 必须大于零')
    for option, directory in [('--build-env-dir', args.build_env_dir), ('--runtime-env-dir', args.runtime_env_dir)]:
        if directory is not None and not directory.is_dir():
            if option == '--runtime-env-dir' and args.with_env and not directory.exists():
                continue
            parser.error(f'{option} 目录不存在: {directory}')
    env_version = matrix['environment_version']
    version = (ROOT / 'VERSION').read_text().strip()
    destination = (args.output_dir or ROOT / '.release/export' / f'rx-met-v{version}').resolve()
    args.runtime_env_dir = (args.runtime_env_dir or ROOT / '.release/runtime-images' / env_version).resolve()
    if args.build_env_dir is not None:
        args.build_env_dir = args.build_env_dir.resolve()
    if args.install_prefix is not None:
        args.install_prefix = args.install_prefix.resolve()
    archives = [bundle_name(version, variant) + '.tar.gz' for variant in targets]
    delivery.check_destination(destination, archives, args.replace)
    changelog = release_changelog(version)
    with delivery.stage_release(destination, archives, replace=args.replace) as args.output_dir:
        for variant in targets:
            build_ref = common.image_ref('build', variant)
            runtime = environment_archives.ensure_image('runtime', variant, args.runtime_env_dir)
            build = environment_archives.ensure_image('build', variant, args.build_env_dir)
            archive, record = environment_archives.image_archive(
                'runtime', variant, args.runtime_env_dir, export=args.with_env)
            if args.with_env:
                environment = destination.parent / f"{record['environment_name']}-{record['environment_version']}"
                delivery.publish_environment(archive, record, environment)
            compatibility = runtime_compatibility(runtime['Id'], matrix, variant)
            if runtime_compatibility(build['Id'], matrix, variant) != compatibility:
                raise RuntimeError('build-env 与 runtime 的 Python ABI 或运行依赖不一致')
            wheel_output = ROOT / '.release/wheels' / version / env_version / variant
            fingerprint = {'source_sha256': source_digest(), 'build_image_id': build['Id'], 'runtime_image_id': runtime['Id']}
            build_record = wheel_output / 'build.json'
            if not args.skip_build:
                run('docker', 'buildx', 'bake', '-f', ROOT / 'docker/docker-bake.hcl', '--progress=plain',
                    '--set', f'{variant}.args.JOBS={args.jobs}', '--set', f'{variant}.args.BUILD_ENV_IMAGE={build_ref}',
                    '--set', f'{variant}.output=type=local,dest={wheel_output}', variant, cwd=ROOT)
                if image_info(build_ref, 'build', variant, env_version)['Id'] != build['Id']:
                    raise RuntimeError('构建过程中 build-env 身份发生变化')
                if source_digest() != fingerprint['source_sha256']:
                    raise RuntimeError('构建过程中 wheel 源码发生变化')
                write_json(build_record, fingerprint)
            elif not build_record.is_file() or json.loads(build_record.read_text()) != fingerprint:
                raise RuntimeError('--skip-build 的源码或环境身份与已有 wheels 不匹配')
            assemble(args, matrix, variant, runtime, build, record, archive, wheel_output / 'wheels', compatibility, version=version, changelog=changelog)
    print('Product delivery:', destination)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as error:
        raise SystemExit(f'release failed: {error}') from None
