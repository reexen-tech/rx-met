"""Apply the reviewed onnxsim 0.7.0 shutdown fix to staged dependency wheels.

Eager Optional[PyModelExecutor] evaluation lets typing's global cache retain the
native executor through interpreter shutdown (CPython issue #151728). Quoting
only this annotation preserves the executor and simplification behavior.

The upstream version and entire Python source must match the reviewed input.
The patched wheel has build tag 1rxmet, a matching WHEEL header and fresh RECORD;
its native extension and all other implementation files remain unchanged.
"""
from __future__ import annotations

import base64
import csv
from email.parser import BytesParser
import hashlib
import io
from pathlib import Path
import tempfile
import zipfile


VERSION = '0.7.0'
BUILD_TAG = '1rxmet'
SOURCE = 'onnxsim/onnx_simplifier.py'
SOURCE_SHA256 = '483ca95482fdf3c397d1566ddf21aecb08456ec13735736681d7c6dbbdf29c14'
PATCHED_SHA256 = 'c588cc242494c7a77221f6233894d21002bbeb9e6c66d3528e306d9d18bdcbf8'
BEFORE = b'_model_executor: Optional[PyModelExecutor] = None'
AFTER = b'_model_executor: "Optional[PyModelExecutor]" = None'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_wheel(path: Path) -> Path:
    """Atomically replace a staged wheel; reject unreviewed or ambiguous inputs."""
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
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if (metadata.get('Name'), metadata.get('Version')) != ('onnxsim', VERSION):
            raise ValueError(f'{path.name}: shutdown fix only supports onnxsim=={VERSION}')
        prefix = metadata_name.rsplit('/', 1)[0]
        record_name = prefix + '/RECORD'
        wheel_name = prefix + '/WHEEL'
        if any(name in names for name in (record_name + '.jws', record_name + '.p7s')):
            raise ValueError(f'{path.name}: cannot modify a signed wheel')
        if not {SOURCE, record_name, wheel_name}.issubset(names):
            raise ValueError(f'{path.name}: missing source, WHEEL or RECORD')
        source = archive.read(SOURCE)
        digest = _digest(source)
        if digest not in (SOURCE_SHA256, PATCHED_SHA256):
            raise ValueError(f'{path.name}: unreviewed {SOURCE} SHA256 {digest}')
        fields = path.name.removesuffix('.whl').split('-')
        if (not path.name.endswith('.whl') or len(fields) not in (5, 6)
                or fields[:2] != ['onnxsim', VERSION]
                or (len(fields) == 6 and fields[2] != BUILD_TAG)):
            raise ValueError(f'{path.name}: unexpected onnxsim wheel filename/build tag')
        target = path.with_name('-'.join(fields[:2] + [BUILD_TAG] + fields[-3:]) + '.whl')
        wheel_data = archive.read(wheel_name)
        wheel_metadata = BytesParser().parsebytes(wheel_data)
        builds = wheel_metadata.get_all('Build', [])
        if digest == PATCHED_SHA256 and path == target and builds == [BUILD_TAG]:
            return path
        if path == target or builds:
            raise ValueError(f'{path.name}: source and build marker disagree')
        if target.exists():
            raise ValueError(f'{path.name}: patched target already exists: {target.name}')
        if source.count(BEFORE) != 1:
            raise ValueError(f'{path.name}: expected exactly one executor annotation')
        patched = source.replace(BEFORE, AFTER)
        if _digest(patched) != PATCHED_SHA256:
            raise ValueError(f'{path.name}: patched source checksum mismatch')
        newline = b'\r\n' if b'\r\n' in wheel_data else b'\n'
        patched_wheel = b'Build: ' + BUILD_TAG.encode('ascii') + newline + wheel_data
        replacements = {SOURCE: patched, wheel_name: patched_wheel}
        output = io.StringIO(newline='')
        writer = csv.writer(output, lineterminator='\n')
        for entry in entries:
            if entry.filename == record_name or entry.is_dir():
                continue
            data = replacements.get(entry.filename)
            if data is None:
                data = archive.read(entry)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=')
            writer.writerow((entry.filename, 'sha256=' + encoded.decode('ascii'), str(len(data))))
        writer.writerow((record_name, '', ''))
        replacements[record_name] = output.getvalue().encode('utf-8')
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.onnxsim-patch-', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with zipfile.ZipFile(temporary, 'w') as result:
                for entry in entries:
                    data = replacements.get(entry.filename)
                    result.writestr(entry, archive.read(entry) if data is None else data)
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    path.unlink()
    return target
