"""Reuse versioned environment tags and verify offline archive integrity."""
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


def requirement(role, variant, *, contract=common):
    matrix = contract.config()
    return {'schema_version': 1, 'environment_name': contract.NAME,
            'environment_version': matrix['environment_version'], 'variant': variant, 'role': role,
            'image_tag': contract.image_ref(role, variant),
            'filename': f"{contract.NAME}-{role}-{matrix['environment_version']}-{variant}-linux-amd64.tar.gz",
            'url': ''}


def verified_archive(role, variant, directory, *, contract=common):
    """Check transport integrity; image compatibility is checked after Docker loads it."""
    record = requirement(role, variant, contract=contract)
    archive = Path(directory) / record['filename']
    if not archive.is_file():
        raise RuntimeError(f'缺少 {role} 镜像归档: {archive}')
    sidecar = archive.with_name(archive.name + '.json')
    metadata = {}
    if sidecar.is_file():
        try:
            metadata = json.loads(sidecar.read_text())
        except (ValueError, UnicodeError):
            pass
        if not isinstance(metadata, dict):
            metadata = {}
    checksum = archive.with_name(archive.name + '.sha256')
    manifest = archive.parent / 'SHA256SUMS'
    expected = None
    if checksum.is_file() or manifest.is_file():
        source = checksum if checksum.is_file() else manifest
        matches = []
        for line in source.read_text().splitlines():
            match = re.fullmatch(r'([0-9a-f]{64})(?:[ \t]+[* ]?(.+))?', line.strip())
            listed = re.fullmatch(r'\S+[ \t]+[* ]?(.+)', line.strip())
            if source == manifest and listed and listed[1] in (archive.name, './' + archive.name) and match is None:
                raise RuntimeError(f'无效的镜像校验文件: {source}')
            if match and (match[2] in (archive.name, './' + archive.name) or (source == checksum and match[2] is None)):
                matches.append(match[1])
            elif source == checksum and line.strip():
                raise RuntimeError(f'无效的镜像校验文件: {source}')
        if len(matches) > 1 or (source == checksum and not matches):
            raise RuntimeError(f'无效的镜像校验文件: {source}')
        expected = matches[0] if matches else None
    elif metadata.get('sha256'):
        expected = metadata['sha256']
    actual = common.sha256(archive)
    if expected is not None and actual != expected:
        raise RuntimeError(f'镜像归档 SHA-256 不匹配: {archive}')
    record.update(sha256=actual, size_bytes=archive.stat().st_size)
    if metadata.get('image_id'):
        record['image_id'] = metadata['image_id']
    return archive, record


def write_checksum(archive, record):
    checksum = archive.with_name(archive.name + '.sha256')
    temporary = checksum.with_name(checksum.name + '.partial')
    temporary.write_text(f"{record['sha256']}  {archive.name}\n")
    temporary.replace(checksum)


def _save_runtime(reference, destination):
    compressor = ['pigz', '-p', '2', '-1', '-n'] if shutil.which('pigz') else ['gzip', '-1', '-n']
    with destination.open('xb') as output:
        saver = subprocess.Popen(['docker', 'save', reference], stdout=subprocess.PIPE)
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



def image_archive(role, variant, directory, *, export=False, contract=common):
    """Reuse any archive for the environment version, or export the local tagged image."""
    directory = Path(directory)
    record = requirement(role, variant, contract=contract)
    archive = directory / record['filename']

    def existing():
        if archive.exists():
            return verified_archive(role, variant, directory, contract=contract)
        if archive.with_name(archive.name + '.json').exists() or archive.with_name(archive.name + '.sha256').exists():
            raise RuntimeError(f'缺少 {role} 镜像归档: {archive}')
        return None

    found = existing()
    if found or not export:
        return found or (archive, record)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.' + archive.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        found = existing()
        if found:
            return found
        image = contract.inspect(record['image_tag'], role, variant)
        print(f"导出 {role} 镜像: {record['image_tag']} -> {archive}", flush=True)
        with tempfile.TemporaryDirectory(prefix='.environment-export-', dir=directory) as temporary:
            staged = Path(temporary) / archive.name
            _save_runtime(record['image_tag'], staged)
            verify_runtime_payload(staged, image['Id'])
            record.update(image_id=image['Id'], sha256=common.sha256(staged), size_bytes=staged.stat().st_size)
            common.write_json(staged.with_name(staged.name + '.json'), record)
            staged.replace(archive)
            staged.with_name(staged.name + '.json').replace(archive.with_name(archive.name + '.json'))
            write_checksum(archive, record)
        return archive, record


def ensure_image(role, variant, directory=None, *, contract=common):
    """Resolve a compatible version tag, importing an archive only when it is absent."""
    reference = contract.image_ref(role, variant)
    if _local_image(reference) is None:
        if directory is None:
            raise RuntimeError(f'缺少环境镜像 {reference}；使用 --{role}-env-dir 指定已归档镜像目录')
        archive, record = verified_archive(role, variant, directory, contract=contract)
        print(f'导入 {role} 镜像: {archive}', flush=True)
        loaded = subprocess.run(['docker', 'load', '--input', str(archive)],
                                check=True, capture_output=True, text=True)
        if _local_image(reference) is None:
            references = [line.split(': ', 1)[1] for line in (loaded.stdout or '').splitlines()
                          if line.startswith(('Loaded image: ', 'Loaded image ID: '))]
            if len(references) != 1:
                raise RuntimeError(f'归档未提供所需环境标签: {reference}')
            image = contract.inspect(references[0], role, variant)
            subprocess.run(['docker', 'tag', image['Id'], reference], check=True)
    image = contract.inspect(reference, role, variant)
    print(f"复用 {role} 镜像: {reference} ({image['Id']})", flush=True)
    return image
