"""Reuse the published GPU environment locally or import its verified archives."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from environment import common


def _local_image(reference):
    result = subprocess.run(['docker', 'image', 'inspect', reference], capture_output=True, text=True)
    if result.returncode:
        if 'No such image:' in result.stderr or 'No such object:' in result.stderr:
            return None
        raise RuntimeError(f'无法检查 Docker 镜像 {reference}: {result.stderr.strip()}')
    return json.loads(result.stdout)[0]


def verified_archive(role, variant, published, directory):
    """Check archive identity and bytes before allowing docker load."""
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
    if role == 'runtime':
        expected.update({key: published['runtime'][key] for key in ('filename', 'sha256', 'size_bytes')})
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
    return archive


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
            archive = verified_archive(role, variant, published, directory)
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
