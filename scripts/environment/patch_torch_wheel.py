"""Patch reviewed staged Torch wheels to bound FX generated-source lifetime.

Only reviewed source hashes are changed. Other Torch versions stay untouched
and are reported as uncovered. Native members are copied in bounded chunks;
only graph_module.py, the new helper, WHEEL and RECORD change.
"""
from __future__ import annotations

import ast
import base64
import copy
import csv
from email.parser import BytesParser
import hashlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile


BUILD_TAG = '1rxmet'
SOURCE = 'torch/fx/graph_module.py'
HELPER = 'torch/fx/_rx_source_cache.py'
HELPER_PATH = Path(__file__).with_name('patches') / 'torch_fx_source_cache.py'
HELPER_SHA256 = '9f2ee566fbdc4c94732a8794b7995b95b354a1ff0d8f5b62a0cc3d9f81b63257'
# Each value is (upstream graph_module.py hash, patched graph_module.py hash).
REVIEWED_SOURCES = {
    '2.10.0': ('a681378dbc62b36ef7527c7f5254632a62ff536435b8b5b12130d464353410af',
               '170a42e0700d440b54993cd5f164751785081597fb0b42af27455772ad5fc01f'),
    '2.10.0+cu130': ('a681378dbc62b36ef7527c7f5254632a62ff536435b8b5b12130d464353410af',
                     '170a42e0700d440b54993cd5f164751785081597fb0b42af27455772ad5fc01f'),
    '2.8.0': ('e37eac2ee7214a78e06eb7e0aca5793f10b7f921cbd47d6a94392ba23fc24d70',
              '0825a5df58a16c197321cfa2b911e104fde0ec74badf95506019380902431fe9'),
    '2.8.0+cu126': ('e37eac2ee7214a78e06eb7e0aca5793f10b7f921cbd47d6a94392ba23fc24d70',
                    '0825a5df58a16c197321cfa2b911e104fde0ec74badf95506019380902431fe9'),
}
COPY_CHUNK_SIZE = 1024 * 1024
SMALL_MEMBER_LIMIT = 16 * 1024 * 1024


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_values(data: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=')
    return 'sha256=' + digest.decode('ascii'), str(len(data))


def patch_source(source: bytes) -> bytes:
    """Replace only the reviewed loader and its compile entry point."""
    text = source.decode('utf-8')
    module = ast.parse(text)
    loaders = [node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == '_EvalCacheLoader']
    executors = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == '_exec_with_source']
    if len(loaders) != 1 or len(executors) != 1:
        raise ValueError('expected one FX loader and one source execution entry point')
    loader, executor = loaders[0], executors[0]
    if loader.decorator_list or executor.decorator_list:
        raise ValueError('unexpected decorated FX loader or execution entry point')
    lines = text.splitlines(keepends=True)
    newline = '\r\n' if '\r\n' in text else '\n'
    replacements = [
        (loader.lineno - 1, loader.end_lineno,
         'from ._rx_source_cache import OwnedEvalCacheLoader as _EvalCacheLoader' + newline),
        (executor.body[0].lineno - 1, executor.end_lineno,
         '    _loader.exec_with_source(src, globals, co_fields)' + newline),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    patched = ''.join(lines).encode('utf-8')
    ast.parse(patched)
    return patched


def _read_small(archive: zipfile.ZipFile, name: str) -> bytes:
    if archive.getinfo(name).file_size > SMALL_MEMBER_LIMIT:
        raise ValueError(f'{name}: metadata/source member exceeds size limit')
    return archive.read(name)


def _record_rows(data: bytes, files: set[str], record_name: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(data.decode('utf-8'), newline='')))
    names = [row[0] for row in rows if len(row) == 3]
    if len(names) != len(rows) or len(set(names)) != len(names):
        raise ValueError('invalid or duplicate RECORD rows')
    if set(names) != files:
        raise ValueError('RECORD members disagree with wheel contents')
    if next(row[1:] for row in rows if row[0] == record_name) != ['', '']:
        raise ValueError('RECORD must not hash itself')
    return rows


def patch_wheel(path: Path, *, helper_path: Path | None = None) -> Path:
    """Atomically replace a covered staged wheel; leave other versions intact."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError(f'{path.name}: duplicate wheel members')
        metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
        if len(metadata_names) != 1:
            raise ValueError(f'{path.name}: expected one METADATA member')
        metadata_name = metadata_names[0]
        metadata = BytesParser().parsebytes(_read_small(archive, metadata_name))
        if metadata.get_all('Name') != ['torch'] or len(metadata.get_all('Version', [])) != 1:
            raise ValueError(f'{path.name}: expected Torch distribution metadata')
        version = metadata['Version']
        reviewed = REVIEWED_SOURCES.get(version)
        if reviewed is None:
            print(f'[rx-met] FX source-cache fix is not applied to torch=={version}; '
                  f'this version is not covered by reviewed sources', file=sys.stderr)
            return path
        prefix = metadata_name.rsplit('/', 1)[0]
        record_name, wheel_name = prefix + '/RECORD', prefix + '/WHEEL'
        if any(name in names for name in (record_name + '.jws', record_name + '.p7s')):
            raise ValueError(f'{path.name}: cannot modify a signed wheel')
        if not {SOURCE, record_name, wheel_name}.issubset(names):
            raise ValueError(f'{path.name}: missing source, WHEEL or RECORD')
        source = _read_small(archive, SOURCE)
        digest = _digest(source)
        if digest not in reviewed:
            raise ValueError(f'{path.name}: unreviewed {SOURCE} SHA256 {digest}')
        fields = path.name.removesuffix('.whl').split('-')
        if (not path.name.endswith('.whl') or len(fields) not in (5, 6)
                or fields[:2] != ['torch', version]
                or (len(fields) == 6 and fields[2] != BUILD_TAG)):
            raise ValueError(f'{path.name}: unexpected Torch wheel filename/build tag')
        target = path.with_name('-'.join(fields[:2] + [BUILD_TAG] + fields[-3:]) + '.whl')
        helper = Path(helper_path or HELPER_PATH).read_bytes()
        if _digest(helper) != HELPER_SHA256:
            raise ValueError(f'{path.name}: unreviewed source-cache helper SHA256 {_digest(helper)}')
        ast.parse(helper)
        wheel_data = _read_small(archive, wheel_name)
        wheel_metadata = BytesParser().parsebytes(wheel_data)
        builds = wheel_metadata.get_all('Build', [])
        files = {entry.filename for entry in entries if not entry.is_dir()}
        rows = _record_rows(_read_small(archive, record_name), files, record_name)
        if digest == reviewed[1] and path == target and builds == [BUILD_TAG]:
            if HELPER not in names or _read_small(archive, HELPER) != helper:
                raise ValueError(f'{path.name}: patched source-cache helper disagrees')
            recorded = {row[0]: tuple(row[1:]) for row in rows}
            for name, data in ((SOURCE, source), (HELPER, helper), (wheel_name, wheel_data)):
                if recorded[name] != _record_values(data):
                    raise ValueError(f'{path.name}: patched RECORD disagrees for {name}')
            return path
        if path == target or builds or digest == reviewed[1]:
            raise ValueError(f'{path.name}: source and build marker disagree')
        if HELPER in names:
            raise ValueError(f'{path.name}: source-cache helper already exists')
        if target.exists():
            raise ValueError(f'{path.name}: patched target already exists: {target.name}')
        patched = patch_source(source)
        if _digest(patched) != reviewed[1]:
            raise ValueError(f'{path.name}: patched source checksum mismatch')
        newline = b'\r\n' if b'\r\n' in wheel_data else b'\n'
        replacements = {SOURCE: patched, HELPER: helper,
                        wheel_name: b'Build: ' + BUILD_TAG.encode('ascii') + newline + wheel_data}
        output = io.StringIO(newline='')
        writer = csv.writer(output, lineterminator='\n')
        for name, digest_value, size in rows:
            if name in replacements:
                digest_value, size = _record_values(replacements[name])
            writer.writerow((name, digest_value, size))
        writer.writerow((HELPER, *_record_values(helper)))
        replacements[record_name] = output.getvalue().encode('utf-8')
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.torch-fx-patch-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with zipfile.ZipFile(temporary, 'w', allowZip64=True) as result:
                result.comment = archive.comment
                for entry in entries:
                    destination = copy.copy(entry)
                    if entry.filename in replacements:
                        result.writestr(destination, replacements[entry.filename])
                    elif entry.is_dir():
                        result.writestr(destination, b'')
                    else:
                        with archive.open(entry, 'r') as original, result.open(
                                destination, 'w', force_zip64=entry.file_size >= zipfile.ZIP64_LIMIT) as copied:
                            shutil.copyfileobj(original, copied, length=COPY_CHUNK_SIZE)
                helper_info = zipfile.ZipInfo(HELPER, date_time=archive.getinfo(SOURCE).date_time)
                helper_info.compress_type = archive.getinfo(SOURCE).compress_type
                helper_info.external_attr = 0o100644 << 16
                result.writestr(helper_info, helper)
            temporary.chmod(path.stat().st_mode & 0o777)
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise ValueError(f'{path.name}: patched target already exists: {target.name}') from error
        finally:
            temporary.unlink(missing_ok=True)
    path.unlink()
    return target
