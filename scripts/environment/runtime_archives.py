#!/usr/bin/env python3
"""Export and load versioned environment tags independently of repository records."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from environment import common
from release import delivery, environment_archives


def export_environment(variant, directory, role='runtime', *, contract=common):
    directory = Path(directory)
    archive, record = environment_archives.image_archive(
        role, variant, directory, export=True, contract=contract)
    delivery.publish_environment(archive, record, directory)
    print(f'{role.capitalize()} archive:', archive, flush=True)


def export_runtime(variant, directory, *, contract=common):
    export_environment(variant, directory, contract=contract)


def load_environment(variant, directory, role='runtime', *, contract=common):
    environment_archives.ensure_image(role, variant, directory, contract=contract)


def load_runtime(variant, directory, *, contract=common):
    load_environment(variant, directory, contract=contract)


def main():
    matrix = common.config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'load'])
    parser.add_argument('variants', nargs='*', choices=list(matrix['variants']),
                        help='Environment variants (default: CUDA variants; CPU requires cpu)')
    parser.add_argument('--output-dir', '--export-dir', dest='output_dir', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--role', choices=['runtime', 'build', 'all'], default='runtime',
                        help='Environment image role (default: runtime)')
    args = parser.parse_intermixed_args()
    if args.action == 'load' and args.directory is None:
        parser.error('load requires --directory')
    directory = (args.directory if args.action == 'load' else
                 args.output_dir or ROOT / '.release/runtime-images' / matrix['environment_version']).resolve()
    for variant in args.variants or common.default_variants(matrix):
        for role in ('build', 'runtime') if args.role == 'all' else (args.role,):
            if args.action == 'export':
                export_environment(variant, directory, role)
            else:
                load_environment(variant, directory, role)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'public environment archive failed: {error}') from None
