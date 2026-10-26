#!/usr/bin/env python3
"""Export and load versioned runtime tags independently of repository records."""
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


def export_runtime(variant, directory, *, contract=common):
    archive, record = environment_archives.image_archive(
        'runtime', variant, directory, export=True, contract=contract)
    environment_archives.write_checksum(archive, record)
    matrix = contract.config()
    manifest_path = directory / 'manifest.json'
    document = {'schema_version': 1, 'environment_name': contract.NAME,
                'environment_version': matrix['environment_version'],
                'variants': {}}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != document[key] for key in ('environment_name', 'environment_version')):
            raise RuntimeError('归档目录包含其他公共环境的发布清单')
        document = existing
    document.pop('config_sha256', None)
    document['variants'][variant] = {'runtime': record}
    common.write_json(manifest_path, document)
    (directory / 'README.md').write_text(
        f'# {contract.NAME} runtime {matrix["environment_version"]}\n\n'
        '将归档目录按环境名和版本命名，与产品目录平级放置；在产品目录内解压应用包并运行 install.sh。\n'
        '应用包按版本标签复用兼容环境；归档校验值独立记录，镜像 ID 仅供追踪。\n')
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith('.') and p.name != 'SHA256SUMS')
    (directory / 'SHA256SUMS').write_text(''.join(f'{common.sha256(p)}  {p.name}\n' for p in files))
    print('Runtime archive:', archive, flush=True)


def load_runtime(variant, directory, *, contract=common):
    environment_archives.ensure_image('runtime', variant, directory, contract=contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'load'])
    parser.add_argument('variants', nargs='*', choices=list(common.config()['variants']))
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
