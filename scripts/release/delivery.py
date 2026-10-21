"""Assemble product deliveries and independently reusable environment attachments."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from environment import common


def check_destination(directory, replace):
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise RuntimeError(f'产品交付路径不是独立目录: {directory}')
    if directory.exists() and not replace:
        raise RuntimeError(f'产品交付目录已存在: {directory}；开发时可显式 --replace')


@contextmanager
def stage_release(destination, *, replace):
    """Publish all selected variants together, retaining the previous release on failure."""
    check_destination(destination, replace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f'.{destination.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_destination(destination, replace)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
        staged = temporary / destination.name
        previous = temporary / 'previous'
        staged.mkdir()
        try:
            yield staged
            check_destination(destination, replace)
            if destination.exists():
                destination.replace(previous)
            try:
                staged.replace(destination)
            except BaseException:
                if previous.exists():
                    previous.replace(destination)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        finally:
            # Preserve the backup if restoring or removing it failed.
            if not previous.exists():
                shutil.rmtree(temporary)


def publish_environment(archive, record, directory):
    """Attach a verified archive without replacing a published environment."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = directory / 'manifest.json'
        document = {key: record[key] for key in
                    ('schema_version', 'environment_name', 'environment_version', 'config_sha256')}
        document['variants'] = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if any(existing.get(key) != document[key] for key in
                   ('schema_version', 'environment_name', 'environment_version', 'config_sha256')):
                raise RuntimeError(f'环境交付目录包含其他环境的发布清单: {directory}')
            document = existing
        verified = {}
        for entry in document['variants'].values():
            previous_record = entry['runtime']
            previous_archive = directory / previous_record['filename']
            if not previous_archive.is_file() or common.sha256(previous_archive) != previous_record['sha256']:
                raise RuntimeError(f'环境交付目录的归档冲突: {previous_archive}')
            verified[previous_archive.name] = previous_record['sha256']
        old = document['variants'].get(record['variant'])
        if old and old['runtime']['image_id'] != record['image_id']:
            raise RuntimeError(f'环境交付目录的镜像身份冲突: {directory}')
        destination = directory / archive.name
        if destination.exists():
            if (verified.get(destination.name) or common.sha256(destination)) != record['sha256']:
                raise RuntimeError(f'环境交付目录的归档冲突: {destination}')
        else:
            try:
                os.link(archive, destination)
            except OSError:
                shutil.copy2(archive, destination)
        verified[destination.name] = record['sha256']
        common.write_json(destination.with_name(destination.name + '.json'), record)
        destination.with_name(destination.name + '.sha256').write_text(
            f"{record['sha256']}  {destination.name}\n")
        document['variants'][record['variant']] = {'runtime': record}
        common.write_json(manifest_path, document)
        (directory / 'README.md').write_text(
            f"# {record['environment_name']} {record['environment_version']} 公共运行环境\n\n"
            '将本目录与产品交付目录平级放置，在产品目录内解压所选应用包并运行 install.sh。\n'
            '这里只交付运行环境镜像；应用包按精确镜像 ID 校验环境，不包含构建环境镜像。\n')
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and not path.name.startswith('.') and path.name != 'SHA256SUMS')
        (directory / 'SHA256SUMS').write_text(''.join(
            f'{verified.get(path.name) or common.sha256(path)}  {path.name}\n' for path in files))


def write_release_index(directory, product, version, changelog, bundles, *, with_env):
    lines = [f'# {product} v{version} 产品交付', '',
             '在本目录内解压所选应用包，然后运行包内 ./install.sh。',
             '首次离线部署时，将所需公共环境目录与本产品目录平级放置。',
             '安装和运行参数见应用包 README。', '',
             '| 变体 | 应用包 | 应用大小（MiB） | 所需公共环境 | 环境附件 |',
             '| --- | --- | ---: | --- | --- |']
    entries = []
    for manifest in bundles:
        filename = manifest['bundle_name'] + '.tar.gz'
        archive = directory / filename
        runtime = manifest['runtime']
        environment = f"{runtime['environment_name']}-{runtime['environment_version']}"
        runtime_path = f'../{environment}/{runtime["filename"]}'
        attachment = f'[已附带]({runtime_path})' if with_env else '未附带，需另行准备'
        lines.append(f"| {manifest['variant']} | [{filename}]({filename}) | "
                     f"{archive.stat().st_size / 1024**2:.1f} | {environment} | {attachment} |")
        entries.append({'variant': manifest['variant'], 'filename': filename,
                        'sha256': common.sha256(archive), 'size_bytes': archive.stat().st_size,
                        'runtime': runtime, 'environment_directory': f'../{environment}',
                        'environment_included': with_env})
    lines += ['', '本目录 SHA256SUMS 只校验产品文件；公共环境目录维护独立的 manifest.json 和 SHA256SUMS。',
              '每个应用包另有自己的 SHA256SUMS，install.sh 会自动校验。', '']
    (directory / 'README.md').write_text('\n'.join(lines))
    (directory / 'CHANGELOG.md').write_text(changelog, encoding='utf-8')
    common.write_json(directory / 'release.json',
                      {'schema_version': 1, 'product': product, 'version': version, 'bundles': entries})
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != 'SHA256SUMS')
    (directory / 'SHA256SUMS').write_text(''.join(
        f'{common.sha256(path)}  {path.name}\n' for path in files))
