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
ENVIRONMENT_README_TEMPLATE = Path(__file__).resolve().parents[2] / 'packaging/environment/README.md.in'
PRODUCT_README_TEMPLATE = Path(__file__).resolve().parents[2] / 'packaging/product/README.md.in'


def environment_readme(record):
    text = ENVIRONMENT_README_TEMPLATE.read_text(encoding='utf-8')
    for name, key in (('ENV_NAME', 'environment_name'), ('ENV_VERSION', 'environment_version'),
                      ('ARCHIVE_FILENAME', 'filename')):
        text = text.replace('@' + name + '@', record[key])
    return text


def product_readme(product, version, matrix, directory):
    available = []
    rows = []
    for variant, spec in matrix['variants'].items():
        bundle = f'{product}-v{version}-{variant}-linux-amd64'
        if not (Path(directory) / (bundle + '.tar.gz')).is_file():
            continue
        available.append((variant, bundle))
        architectures = ', '.join(f'sm_{value}' for value in spec.get('cuda_architectures', [])) or '-'
        rows.append('| ' + ' | '.join((variant, spec.get('python_version', '-'),
                    spec.get('cuda_version') or '-', spec.get('minimum_driver_version') or '-',
                    architectures, f'`{bundle}.tar.gz`')) + ' |')
    if not available:
        raise RuntimeError(f'缺少 {product} {version} 的应用归档，无法生成产品目录说明')
    variant, bundle = available[0]
    env_version = matrix['environment_version']
    values = {
        'PRODUCT': product, 'VERSION': version, 'DIRECTORY': Path(directory).name,
        'VARIANT_TABLE': '\n'.join(rows), 'EXAMPLE_BUNDLE': bundle,
        'ENV_DIRECTORY': f'{common.NAME}-{env_version}',
        'RUNTIME_ARCHIVE': f'{common.NAME}-runtime-{env_version}-{variant}-linux-amd64.docker.tar.gz',
    }
    text = PRODUCT_README_TEMPLATE.read_text(encoding='utf-8')
    for name, value in values.items():
        text = text.replace('@' + name + '@', value)
    return text


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


def _environment_archive_checksums(directory, record):
    verified = {}
    for archive in directory.glob('*.docker.tar.gz'):
        if archive.is_symlink() or not archive.is_file():
            raise RuntimeError(f'环境产物路径不是普通文件: {archive}')
        actual = common.sha256(archive)
        metadata = archive.with_name(archive.name + '.json')
        checksum = archive.with_name(archive.name + '.sha256')
        for path in (metadata, checksum):
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise RuntimeError(f'环境产物路径不是普通文件: {path}')
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            if not isinstance(previous, dict):
                raise RuntimeError(f'无效的环境归档说明: {metadata}')
            if any(key in previous and previous[key] != record[key]
                   for key in ('environment_name', 'environment_version')):
                raise RuntimeError(f'环境交付目录包含其他环境的归档: {archive}')
            if previous.get('filename', archive.name) != archive.name or previous.get('sha256', actual) != actual:
                raise RuntimeError(f'环境交付目录的归档冲突: {archive}')
        if checksum.exists():
            parts = checksum.read_text().strip().split(maxsplit=1)
            if (len(parts) != 2 or parts[0].lower() != actual
                    or parts[1].removeprefix('*') != archive.name):
                raise RuntimeError(f'环境交付目录的归档冲突: {archive}')
        verified[archive.name] = actual
    return verified


def publish_environment(archive, record, directory):
    """Attach a verified archive without replacing a published environment."""
    directory.mkdir(parents=True, exist_ok=True)
    with _publication_lock(directory):
        verified = _environment_archive_checksums(directory, record)
        destination = directory / archive.name
        if destination.exists():
            if (verified.get(destination.name) or common.sha256(destination)) != record['sha256']:
                raise RuntimeError(f'环境交付目录的归档冲突: {destination}')
        else:
            try:
                os.link(archive, destination)
            except OSError:
                shutil.copy2(archive, destination)
        common.write_json(destination.with_name(destination.name + '.json'), record)
        destination.with_name(destination.name + '.sha256').write_text(
            f"{record['sha256']}  {destination.name}\n")
        (directory / 'README.md').write_text(environment_readme(record), encoding='utf-8')
        (directory / 'manifest.json').unlink(missing_ok=True)
        (directory / 'SHA256SUMS').unlink(missing_ok=True)
