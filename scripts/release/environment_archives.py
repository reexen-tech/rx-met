"""Reuse the published GPU environment locally or import its verified archives."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from environment import common
from release.verify_runtime_archive import verify_runtime_payload


def _local_image(reference):
    result = subprocess.run(['docker', 'image', 'inspect', reference], capture_output=True, text=True)
    if result.returncode:
        if 'No such image:' in result.stderr or 'No such object:' in result.stderr:
            return None
        raise RuntimeError(f'无法检查 Docker 镜像 {reference}: {result.stderr.strip()}')
    return json.loads(result.stdout)[0]


def verified_archive(role, variant, published, directory):
    """Return the archive and its actual file record, pinned to the published image."""
    matrix = common.config()
    filename = f"{common.NAME}-{role}-{matrix['environment_version']}-{variant}-linux-amd64.tar.gz"
    archive = Path(directory) / filename
    sidecar = archive.with_name(filename + '.json')
    if not archive.is_file() or not sidecar.is_file():
        raise RuntimeError(f'缺少 {role} 镜像归档或 .json 身份记录: {archive}')
    record = json.loads(sidecar.read_text())
    if not isinstance(record, dict):
        raise RuntimeError(f'{sidecar}: 无效的镜像身份记录')
    expected = {'schema_version': 1, 'environment_name': common.NAME,
                'environment_version': matrix['environment_version'], 'variant': variant,
                'config_sha256': common.config_digest(), 'image_id': published[f'{role}_image_id'],
                'filename': filename}
    if role == 'build' or 'role' in record:
        expected['role'] = role
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f'{sidecar}: {key} 与公共环境发布记录不一致')
    if not re.fullmatch(r'[0-9a-f]{64}', str(record.get('sha256', ''))):
        raise RuntimeError(f'{sidecar}: 无效的 SHA-256')
    if type(record.get('size_bytes')) is not int or record['size_bytes'] != archive.stat().st_size:
        raise RuntimeError(f'镜像归档大小不匹配: {archive}')
    print(f'校验 {role} 镜像归档: {archive}', flush=True)
    if common.sha256(archive) != record['sha256']:
        raise RuntimeError(f'镜像归档 SHA-256 不匹配: {archive}')
    # The publication checksum identifies the initial export, not every possible
    # docker-save/gzip representation of this immutable image.
    if role == 'runtime' and record['sha256'] != published['runtime']['sha256']:
        verify_runtime_payload(archive, expected['image_id'])
    return archive, record


def write_checksum(archive, record):
    """Publish a standalone file checksum; application manifests do not pin it."""
    checksum = archive.with_name(archive.name + '.sha256')
    temporary = checksum.with_name(checksum.name + '.partial')
    temporary.write_text(f"{record['sha256']}  {archive.name}\n")
    temporary.replace(checksum)


def _save_runtime(image_id, destination):
    compressor = ['pigz', '-p', '2', '-1', '-n'] if shutil.which('pigz') else ['gzip', '-1', '-n']
    with destination.open('xb') as output:
        saver = subprocess.Popen(['docker', 'save', image_id], stdout=subprocess.PIPE)
        try:
            subprocess.run(compressor, stdin=saver.stdout, stdout=output, check=True)
            saver.stdout.close()
            if saver.wait():
                raise RuntimeError('runtime 镜像导出失败')
        finally:
            saver.stdout.close()
            if saver.poll() is None:
                saver.terminate()
                saver.wait()
    subprocess.run(['gzip', '-t', str(destination)], check=True)


def runtime_archive(variant, published, directory, *, export=False):
    """Reuse a checked archive, optionally exporting a missing one from the pinned ID."""
    directory = Path(directory)
    archive = directory / published['runtime']['filename']
    sidecar = archive.with_name(archive.name + '.json')

    def existing():
        if archive.exists() or sidecar.exists():
            return verified_archive('runtime', variant, published, directory)
        return None

    if not export:
        # App-only builds need no archive; the first published export remains the
        # default download record when no local archive has been selected.
        return existing() or (archive, dict(published['runtime']))
    found = existing()
    if found:
        return found
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.' + archive.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        found = existing()
        if found:
            return found
        image_id = published['runtime_image_id']
        runtime = common.inspect(image_id, 'runtime', variant)
        if runtime['Id'] != image_id:
            raise RuntimeError('待导出的 runtime 与公共环境发布记录不一致')
        print(f'自动导出 runtime 镜像: {image_id} -> {archive}', flush=True)
        with tempfile.TemporaryDirectory(prefix='.runtime-export-', dir=directory) as temporary:
            staged = Path(temporary) / archive.name
            _save_runtime(image_id, staged)
            verify_runtime_payload(staged, image_id)
            record = {**published['runtime'], 'image_id': image_id,
                      'sha256': common.sha256(staged), 'size_bytes': staged.stat().st_size, 'url': ''}
            staged_sidecar = staged.with_name(staged.name + '.json')
            common.write_json(staged_sidecar, record)
            staged.replace(archive)
            staged_sidecar.replace(sidecar)
        return archive, record


def ensure_image(role, variant, published, directory=None):
    """Return the exact published image, restoring its tag when necessary."""
    reference = common.image_ref(role, variant)
    expected_id = published[f'{role}_image_id']
    local = _local_image(reference)
    if local is not None and local['Id'] != expected_id:
        raise RuntimeError(f'{reference}: 同名镜像与公共环境发布记录不一致；不会自动覆盖')
    if local is None:
        if _local_image(expected_id) is None:
            option = f'--{role}-env-dir'
            if directory is None:
                raise RuntimeError(f'缺少公共环境镜像 {reference}；使用 {option} 指定已归档镜像目录')
            archive, _ = verified_archive(role, variant, published, directory)
            print(f'导入 {role} 镜像: {archive}', flush=True)
            subprocess.run(['docker', 'load', '--input', str(archive)], check=True)
        # Archives saved by image ID may not carry a tag. Validate before tagging.
        image = common.inspect(expected_id, role, variant)
        if image['Id'] != expected_id:
            raise RuntimeError(f'{reference}: 导入后的镜像身份与发布记录不一致')
        tagged = _local_image(reference)
        if tagged is not None and tagged['Id'] != expected_id:
            raise RuntimeError(f'{reference}: 导入后的标签与发布记录不一致')
        if tagged is None:
            subprocess.run(['docker', 'tag', expected_id, reference], check=True)
    image = common.inspect(reference, role, variant)
    if image['Id'] != expected_id:
        raise RuntimeError(f'{reference}: 镜像身份与发布记录不一致')
    print(f'复用 {role} 镜像: {reference} ({expected_id})', flush=True)
    return image
