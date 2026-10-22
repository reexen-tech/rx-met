#!/usr/bin/env python3
"""Export and load published runtime images without freezing their archive bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from environment import common
from release import environment_archives


def export_runtime(variant, directory):
    publication = common.ROOT / 'docker/environment-images.json'
    published_variants = json.loads(publication.read_text()).get('variants', {}) if publication.is_file() else {}
    if variant not in published_variants:
        # First publication creates the immutable image identities using the
        # original environment implementation, whose bytes are part of v1's digest.
        common.export(variant, directory)
    published = common.load_record(variant)
    archive, record = environment_archives.runtime_archive(variant, published, directory, export=True)
    environment_archives.write_checksum(archive, record)
    matrix = common.config()
    manifest_path = directory / 'manifest.json'
    document = {'schema_version': 1, 'environment_name': common.NAME,
                'environment_version': matrix['environment_version'],
                'config_sha256': common.config_digest(), 'variants': {}}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != document[key] for key in ('environment_name', 'environment_version', 'config_sha256')):
            raise RuntimeError('归档目录包含其他公共环境的发布清单')
        document = existing
    document['variants'][variant] = {**published, 'runtime': record}
    common.write_json(manifest_path, document)
    (directory / 'README.md').write_text(
        '# rx-quant 公共 runtime\n\n'
        '将归档目录按环境名和版本命名，与产品目录平级放置；在产品目录内解压应用包并运行 install.sh。\n'
        '应用包固定镜像 ID；同一镜像重新导出的归档可以复用，文件校验值单独记录。\n')
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith('.') and p.name != 'SHA256SUMS')
    (directory / 'SHA256SUMS').write_text(''.join(f'{common.sha256(p)}  {p.name}\n' for p in files))
    print('Runtime archive:', archive, flush=True)


def load_runtime(variant, directory):
    published = common.load_record(variant)
    archive, _ = environment_archives.verified_archive('runtime', variant, published, directory)
    reference = common.image_ref('runtime', variant)
    local = environment_archives._local_image(reference)
    if local is not None and local['Id'] != published['runtime_image_id']:
        raise RuntimeError(f'{reference}: 同名镜像与公共环境发布记录不一致；不会自动覆盖')
    subprocess.run(['docker', 'load', '--input', str(archive)], check=True)
    environment_archives.ensure_image('runtime', variant, published)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'load'])
    parser.add_argument('variants', nargs='*', choices=['cu118', 'cu126', 'cu130'])
    parser.add_argument('--output-dir', '--export-dir', dest='output_dir', type=Path)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_intermixed_args()
    if args.action == 'load' and args.directory is None:
        parser.error('load requires --directory')
    directory = (args.directory if args.action == 'load' else
                 args.output_dir or ROOT / '.release/runtime-images' / common.config()['environment_version']).resolve()
    for variant in args.variants or common.config()['variants']:
        if args.action == 'export':
            export_runtime(variant, directory)
        else:
            load_runtime(variant, directory)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'public runtime failed: {error}') from None
