"""Assemble product deliveries and independently reusable environment attachments."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from environment import common


LEGACY_PRODUCT_FILES = {'README.md', 'CHANGELOG.md', 'release.json', 'SHA256SUMS'}


def lock_path(directory):
    """Share the same lock across repositories publishing to the same directory."""
    directory = directory.resolve()
    workspace = next((parent for parent in directory.parents if parent.name == '.release'), None)
    if workspace is None:
        export = next((parent for parent in directory.parents if parent.name == 'export'), None)
        workspace = (export.parent if export is not None else directory.parent) / '.release'
    identity = hashlib.sha256(os.fsencode(directory)).hexdigest()
    return workspace / 'locks' / f'{directory.name}-{identity}.lock'


@contextmanager
def _publication_lock(directory):
    path = lock_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def check_destination(directory, archives, replace):
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise RuntimeError(f'产品交付路径不是独立目录: {directory}')
    conflicts = []
    for name in archives:
        if Path(name).name != name or not name.endswith('.tar.gz'):
            raise ValueError(f'无效的应用归档文件名: {name}')
        archive = directory / name
        for path in (archive, archive.with_name(name + '.sha256')):
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise RuntimeError(f'应用产物路径不是普通文件: {path}')
        if archive.exists() and not replace:
            conflicts.append(str(archive))
    if conflicts:
        raise RuntimeError('应用产物已存在: ' + ', '.join(conflicts) + '；可显式 --replace 替换同名归档')


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _retain_release_files(destination, staged, archives):
    replaced = set(archives) | {name + '.sha256' for name in archives}
    if not destination.exists():
        return
    for source in destination.iterdir():
        if source.name in replaced or source.name in LEGACY_PRODUCT_FILES:
            continue
        target = staged / source.name
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            shutil.copytree(source, target, symlinks=True, copy_function=_link_or_copy)
        else:
            _link_or_copy(source, target)


def _legacy_archive_checksums(directory):
    manifest = directory / 'SHA256SUMS'
    if manifest.is_symlink() or not manifest.is_file():
        return {}
    checksums = {}
    for line in manifest.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, name = parts
        name = name.removeprefix('*').removeprefix('./')
        if Path(name).name != name or not name.endswith('.tar.gz'):
            continue
        if len(digest) != 64 or any(character not in '0123456789abcdefABCDEF' for character in digest):
            raise RuntimeError(f'无效的旧应用归档校验值: {manifest}: {name}')
        digest = digest.lower()
        if name in checksums and checksums[name] != digest:
            raise RuntimeError(f'冲突的旧应用归档校验值: {manifest}: {name}')
        checksums[name] = digest
    return checksums


def _write_archive_checksums(staged, archives, legacy_checksums):
    for name in archives:
        archive = staged / name
        if archive.is_symlink() or not archive.is_file():
            raise RuntimeError(f'缺少已验收的应用归档: {archive}')
        (staged / (name + '.sha256')).write_text(f'{common.sha256(archive)}  {name}\n')
    # Old deliveries only had a combined checksum file; migrate retained archives as well.
    for archive in staged.glob('*.tar.gz'):
        checksum = archive.with_name(archive.name + '.sha256')
        if archive.is_symlink() or not archive.is_file() or checksum.exists() or checksum.is_symlink():
            continue
        actual = common.sha256(archive)
        expected = legacy_checksums.get(archive.name)
        if expected is not None and actual != expected:
            raise RuntimeError(f'旧应用归档 SHA-256 不匹配: {archive.name}')
        checksum.write_text(f'{actual}  {archive.name}\n')


@contextmanager
def stage_release(destination, archives, *, replace):
    """Add or replace selected archives together, retaining other files and failure recovery."""
    archives = tuple(archives)
    check_destination(destination, archives, replace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(destination):
        check_destination(destination, archives, replace)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
        staged = temporary / destination.name
        previous = temporary / 'previous'
        staged.mkdir()
        try:
            legacy_checksums = _legacy_archive_checksums(destination)
            _retain_release_files(destination, staged, archives)
            yield staged
            _write_archive_checksums(staged, archives, legacy_checksums)
            check_destination(destination, archives, replace)
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
    with _publication_lock(directory):
        manifest_path = directory / 'manifest.json'
        document = {key: record[key] for key in
                    ('schema_version', 'environment_name', 'environment_version')}
        document['variants'] = {}
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if any(existing.get(key) != document[key] for key in
                   ('schema_version', 'environment_name', 'environment_version')):
                raise RuntimeError(f'环境交付目录包含其他环境的发布清单: {directory}')
            document = existing
        verified = {}
        for entry in document['variants'].values():
            for previous_record in entry.values():
                previous_archive = directory / previous_record['filename']
                if not previous_archive.is_file() or common.sha256(previous_archive) != previous_record['sha256']:
                    raise RuntimeError(f'环境交付目录的归档冲突: {previous_archive}')
                verified[previous_archive.name] = previous_record['sha256']
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
        document['variants'].setdefault(record['variant'], {})[record['role']] = record
        common.write_json(manifest_path, document)
        (directory / 'README.md').write_text(
            f"# {record['environment_name']} {record['environment_version']} 公共环境\n\n"
            '将本目录与产品交付目录平级放置，在产品目录内解压所选应用包并运行 install.sh。\n'
            '应用发布只附带 runtime 镜像；build 镜像可通过独立环境导出命令添加，供开发构建使用。\n'
            '使用 docker load 加载 .docker.tar.gz 镜像归档。\n'
            '应用包按版本标签复用兼容环境。\n')
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and not path.name.startswith('.') and path.name != 'SHA256SUMS')
        (directory / 'SHA256SUMS').write_text(''.join(
            f'{verified.get(path.name) or common.sha256(path)}  {path.name}\n' for path in files))
