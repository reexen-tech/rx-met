#!/usr/bin/env python3
"""验证 wheelhouse 是否精确满足锁定的 requirements。"""

from __future__ import annotations

import argparse
import zipfile
from email.parser import BytesParser
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def load_requirements(paths: list[Path]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for path in paths:
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("--"):
                continue
            requirement = Requirement(line)
            specs = list(requirement.specifier)
            if (
                requirement.url is not None
                or requirement.marker is not None
                or len(specs) != 1
                or specs[0].operator != "=="
            ):
                raise ValueError(f"{path}:{line_number}: 依赖必须使用无条件精确版本: {line}")
            name = canonicalize_name(requirement.name)
            version = specs[0].version
            previous = expected.setdefault(name, version)
            if previous != version:
                raise ValueError(
                    f"依赖版本冲突: {name} 同时要求 {previous} 和 {version}"
                )
    return expected


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"{path.name}: METADATA 数量不是 1")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"{path.name}: METADATA 缺少 Name 或 Version")
    return canonicalize_name(name), version


def verify(wheel_dir: Path, requirement_paths: list[Path]) -> None:
    expected = load_requirements(requirement_paths)
    compatible_tags = set(sys_tags())
    found: dict[str, tuple[str, Path]] = {}

    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise ValueError(f"wheelhouse 为空: {wheel_dir}")
    for wheel in wheels:
        distribution, filename_version, _, tags = parse_wheel_filename(wheel.name)
        if compatible_tags.isdisjoint(tags):
            raise ValueError(f"wheel 与当前 Python/平台不兼容: {wheel.name}")
        metadata_name, metadata_version = wheel_metadata(wheel)
        filename_name = canonicalize_name(distribution)
        if filename_name != metadata_name or str(filename_version) != metadata_version:
            raise ValueError(f"wheel 文件名与 METADATA 不一致: {wheel.name}")
        if metadata_name in found:
            raise ValueError(
                f"wheelhouse 包含重复包: {found[metadata_name][1].name}, {wheel.name}"
            )
        found[metadata_name] = (metadata_version, wheel)

    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(found)
        if expected[name] != found[name][0]
    )
    errors = []
    if missing:
        errors.append(f"缺少依赖: {', '.join(missing)}")
    if extra:
        errors.append(f"存在 lock 外依赖: {', '.join(extra)}")
    if mismatched:
        details = ", ".join(
            f"{name}={found[name][0]} (需要 {expected[name]})" for name in mismatched
        )
        errors.append(f"版本不一致: {details}")
    if errors:
        raise ValueError("; ".join(errors))
    print(f"wheelhouse 校验通过: {len(found)} 个锁定 wheel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("requirements", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        verify(args.wheel_dir, args.requirements)
    except (ValueError, zipfile.BadZipFile) as error:
        raise SystemExit(f"wheelhouse 校验失败: {error}") from None


if __name__ == "__main__":
    main()
