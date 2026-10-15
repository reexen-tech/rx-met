"""Verify runtime archive contents using only the Python standard library.

Embedded in install.sh and run inside the exact pinned runtime, never an arbitrary image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile


def verify_runtime_payload(archive, image_id):
    """Verify classic config IDs and Docker 29 OCI manifest/index IDs without loading."""
    files, documents = {}, {}
    try:
        with tarfile.open(archive, 'r|gz') as saved:
            for member in saved:
                if member.isdir():
                    continue
                if not member.isfile() or member.name in files:
                    raise RuntimeError(f'无效或重复的 Docker 镜像成员: {archive}: {member.name}')
                digest = hashlib.sha256()
                data = bytearray()
                with saved.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b''):
                        digest.update(block)
                        if member.size <= 16 * 1024**2:
                            data.extend(block)
                files[member.name] = (digest.hexdigest(), member.size)
                if data.startswith((b'{', b'[')):
                    try:
                        documents[member.name] = json.loads(data)
                    except (ValueError, UnicodeError):
                        pass
        if 'index.json' in files:
            # containerd-backed Docker uses the OCI index/manifest digest as Id.
            index = documents.get('index.json', {})
            roots = index.get('manifests', []) if isinstance(index, dict) else []
            if (not isinstance(roots, list) or len(roots) != 1 or not isinstance(roots[0], dict)
                    or roots[0].get('digest') != image_id):
                raise RuntimeError(f'归档必须且仅能包含指定的 runtime 镜像: {archive}')

            def verify_descriptor(descriptor):
                if not isinstance(descriptor, dict) or not re.fullmatch(r'sha256:[0-9a-f]{64}', str(descriptor.get('digest', ''))):
                    raise RuntimeError(f'无效的 OCI 镜像描述: {archive}')
                digest = descriptor['digest'][7:]
                name = 'blobs/sha256/' + digest
                if files.get(name) != (digest, descriptor.get('size')):
                    raise RuntimeError(f'OCI 镜像内容缺失或摘要不匹配: {archive}: {name}')
                return documents.get(name)

            def verify_image(descriptor):
                document = verify_descriptor(descriptor)
                if not isinstance(document, dict):
                    raise RuntimeError(f'无效的 OCI 镜像元数据: {archive}')
                if 'manifests' in document:
                    for child in document['manifests']:
                        verify_image(child)
                else:
                    verify_descriptor(document.get('config'))
                    for layer in document.get('layers', []):
                        verify_descriptor(layer)

            verify_image(roots[0])
        else:
            # Classic docker-save archives identify an image by its config digest.
            manifest = documents.get('manifest.json')
            if not isinstance(manifest, list) or len(manifest) != 1 or not isinstance(manifest[0], dict):
                raise RuntimeError(f'归档必须且仅能包含指定的 runtime 镜像: {archive}')
            name = manifest[0].get('Config')
            if not isinstance(name, str) or files.get(name, (None,))[0] != image_id.removeprefix('sha256:'):
                raise RuntimeError(f'归档内的 runtime 镜像身份不匹配: {archive}')
            config = documents.get(name, {})
            expected_layers = config.get('rootfs', {}).get('diff_ids', [])
            layers = manifest[0].get('Layers', [])
            if (not isinstance(layers, list) or len(layers) != len(expected_layers)
                    or any('sha256:' + files.get(layer, ('missing',))[0] != expected
                           for layer, expected in zip(layers, expected_layers))):
                raise RuntimeError(f'归档内的 runtime 镜像层不匹配: {archive}')
    except (tarfile.TarError, EOFError, ValueError, TypeError) as error:
        raise RuntimeError(f'无效的 runtime 镜像归档: {archive}: {error}') from error



if __name__ == '__main__':
    try:
        verify_runtime_payload(Path(sys.argv[1]), sys.argv[2])
        print('Runtime archive identity and contents verified')
    except (RuntimeError, OSError) as error:
        raise SystemExit(str(error)) from None
