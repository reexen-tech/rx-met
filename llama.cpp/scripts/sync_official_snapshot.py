#!/usr/bin/env python3
"""Compare and selectively sync an official llama.cpp snapshot.

The default mode is read-only. Mutating operations require --apply and never
delete files from the internal tree.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Iterable


DEFAULT_OFFICIAL = Path("/mnt/data8t/zcx/llama.cpp-official")
DEFAULT_INTERNAL = Path(__file__).resolve().parent.parent
DEFAULT_EXCLUDES = (
    ".git/",
    ".git/**",
    ".build/",
    ".build/**",
    ".cache/",
    ".cache/**",
    ".venv/",
    ".venv/**",
    "build*/",
    "build*/**",
    "models/",
    "models/**",
    "out/",
    "out/**",
    "output/",
    "output/**",
    "tmp/",
    "tmp/**",
    "**/__pycache__/",
    "**/__pycache__/**",
    "**/node_modules/",
    "**/node_modules/**",
    "**/output/",
    "**/output/**",
)
STATUSES = (
    "different",
    "type_mismatch",
    "official_only",
    "internal_only",
    "same",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "比较官方与内部 llama.cpp 目录；默认只报告，不修改任何文件。"
        )
    )
    parser.add_argument(
        "--official",
        type=Path,
        default=DEFAULT_OFFICIAL,
        help=f"官方仓库目录（默认：{DEFAULT_OFFICIAL}）",
    )
    parser.add_argument(
        "--internal",
        type=Path,
        default=DEFAULT_INTERNAL,
        help=f"内部 llama.cpp 目录（默认：{DEFAULT_INTERNAL}）",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="写入完整报告；扩展名为 .json 时输出 JSON，否则输出 Markdown",
    )
    parser.add_argument(
        "--diff-output",
        type=Path,
        help="将内容不同的文件逐个输出为统一 diff，目录结构与源码一致",
    )
    parser.add_argument(
        "--max-diff-bytes",
        type=int,
        default=2 * 1024 * 1024,
        metavar="N",
        help="生成行级 diff 的单文件大小上限（默认：2097152 字节）",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="附加排除规则（POSIX glob，可重复）",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="不使用构建目录、输出目录等默认排除规则",
    )
    parser.add_argument(
        "--copy-new",
        action="store_true",
        help="选择所有“仅官方存在”的文件进行复制",
    )
    parser.add_argument(
        "--sync-list",
        type=Path,
        help="选择列表中的官方文件进行覆盖/复制，每行一个相对路径",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行复制；不指定时仅预览将执行的操作",
    )
    return parser.parse_args()


def normalize_root(path: Path, label: str) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{label}目录不存在或不是目录：{root}")
    return root


def is_excluded(
    relative: str,
    patterns: tuple[str, ...],
    is_directory: bool,
) -> bool:
    candidate = f"{relative}/" if is_directory else relative
    return any(fnmatch.fnmatchcase(candidate, pattern) for pattern in patterns)


def scan_tree(root: Path, excludes: tuple[str, ...]) -> dict[str, str]:
    entries: dict[str, str] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                is_directory = entry.is_dir(follow_symlinks=False)
                if is_excluded(relative, excludes, is_directory):
                    continue
                if entry.is_symlink():
                    entries[relative] = "symlink"
                elif is_directory:
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    entries[relative] = "file"
                else:
                    entries[relative] = "special"

    visit(root)
    return entries


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries_equal(
    official: Path,
    internal: Path,
    official_type: str,
    internal_type: str,
) -> bool:
    if official_type != internal_type:
        return False
    if official_type == "symlink":
        return os.readlink(official) == os.readlink(internal)
    if official_type != "file":
        return False
    official_stat = official.stat()
    internal_stat = internal.stat()
    if official_stat.st_size != internal_stat.st_size:
        return False
    return sha256(official) == sha256(internal)


def compare_trees(
    official_root: Path,
    internal_root: Path,
    excludes: tuple[str, ...],
) -> dict[str, list[str]]:
    official_entries = scan_tree(official_root, excludes)
    internal_entries = scan_tree(internal_root, excludes)
    result = {status: [] for status in STATUSES}

    for relative in sorted(official_entries.keys() | internal_entries.keys()):
        official_type = official_entries.get(relative)
        internal_type = internal_entries.get(relative)
        if official_type is None:
            result["internal_only"].append(relative)
        elif internal_type is None:
            result["official_only"].append(relative)
        elif official_type != internal_type:
            result["type_mismatch"].append(relative)
        elif entries_equal(
            official_root / relative,
            internal_root / relative,
            official_type,
            internal_type,
        ):
            result["same"].append(relative)
        else:
            result["different"].append(relative)

    return result


def markdown_report(
    official: Path,
    internal: Path,
    excludes: tuple[str, ...],
    result: dict[str, list[str]],
) -> str:
    labels = {
        "different": "双方存在但内容不同",
        "type_mismatch": "类型不一致",
        "official_only": "仅官方存在",
        "internal_only": "仅内部存在",
        "same": "内容相同",
    }
    lines = [
        "# llama.cpp 快照差异报告",
        "",
        f"- 官方目录：`{official}`",
        f"- 内部目录：`{internal}`",
        f"- 排除规则数：{len(excludes)}",
        "",
        "## 汇总",
        "",
    ]
    for status in STATUSES:
        lines.append(f"- {labels[status]}：{len(result[status])}")
    for status in STATUSES:
        lines.extend(("", f"## {labels[status]}", ""))
        paths = result[status]
        if paths:
            lines.extend(f"- `{path}`" for path in paths)
        else:
            lines.append("- 无")
    return "\n".join(lines) + "\n"


def write_report(
    report_path: Path,
    official: Path,
    internal: Path,
    excludes: tuple[str, ...],
    result: dict[str, list[str]],
) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.suffix.lower() == ".json":
        payload = {
            "official": str(official),
            "internal": str(internal),
            "excludes": list(excludes),
            "counts": {status: len(result[status]) for status in STATUSES},
            "files": result,
        }
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        report_path.write_text(
            markdown_report(official, internal, excludes, result),
            encoding="utf-8",
        )


def is_binary(path: Path) -> bool:
    with path.open("rb") as stream:
        sample = stream.read(8192)
    if b"\0" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def describe_entry(path: Path) -> list[str]:
    if path.is_symlink():
        return [f"type: symlink", f"target: {os.readlink(path)}"]
    if path.is_file():
        stat = path.stat()
        return [
            "type: file",
            f"size: {stat.st_size}",
            f"sha256: {sha256(path)}",
        ]
    if path.is_dir():
        return ["type: directory"]
    if path.exists():
        return ["type: special"]
    return ["type: missing"]


def metadata_diff(
    relative: str,
    official: Path,
    internal: Path,
    reason: str,
) -> str:
    lines = [
        f"# {relative}",
        "",
        f"无法生成行级 diff：{reason}",
        "",
        "## official",
        *describe_entry(official),
        "",
        "## internal",
        *describe_entry(internal),
        "",
    ]
    return "\n".join(lines)


def unified_diff(
    relative: str,
    official: Path,
    internal: Path,
    max_diff_bytes: int,
) -> tuple[str, str]:
    if official.is_symlink() or internal.is_symlink():
        return metadata_diff(
            relative, official, internal, "至少一侧是符号链接"
        ), "metadata"
    if not official.is_file() or not internal.is_file():
        return metadata_diff(
            relative, official, internal, "至少一侧不是普通文件"
        ), "metadata"

    official_size = official.stat().st_size
    internal_size = internal.stat().st_size
    if official_size > max_diff_bytes or internal_size > max_diff_bytes:
        reason = (
            "文件超过大小限制 "
            f"{max_diff_bytes} 字节（official={official_size}, "
            f"internal={internal_size}）"
        )
        return metadata_diff(relative, official, internal, reason), "metadata"
    if is_binary(official) or is_binary(internal):
        return metadata_diff(
            relative, official, internal, "检测为二进制或非 UTF-8 文件"
        ), "metadata"

    official_lines = official.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    internal_lines = internal.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    content = "".join(
        difflib.unified_diff(
            official_lines,
            internal_lines,
            fromfile=f"official/{relative}",
            tofile=f"internal/{relative}",
        )
    )
    return content, "text"


def write_diffs(
    output_root: Path,
    official_root: Path,
    internal_root: Path,
    result: dict[str, list[str]],
    max_diff_bytes: int,
) -> tuple[int, int]:
    if max_diff_bytes < 0:
        raise ValueError("--max-diff-bytes 不能为负数")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    text_count = 0
    metadata_count = 0

    for status in ("different", "type_mismatch"):
        for relative in result[status]:
            content, output_type = unified_diff(
                relative,
                official_root / relative,
                internal_root / relative,
                max_diff_bytes,
            )
            destination = output_root / f"{relative}.diff"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            if output_type == "text":
                text_count += 1
            else:
                metadata_count += 1

    summary = {
        "official": str(official_root),
        "internal": str(internal_root),
        "text_diff_count": text_count,
        "metadata_diff_count": metadata_count,
        "max_diff_bytes": max_diff_bytes,
    }
    (output_root / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return text_count, metadata_count


def validate_relative_path(raw: str) -> str:
    value = raw.strip()
    path = PurePosixPath(value)
    if not value or value.startswith("/") or ".." in path.parts:
        raise ValueError(f"非法相对路径：{raw!r}")
    return path.as_posix()


def read_sync_list(path: Path) -> list[str]:
    selected: list[str] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        try:
            selected.append(validate_relative_path(value))
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return selected


def select_paths(
    args: argparse.Namespace,
    result: dict[str, list[str]],
) -> list[str]:
    selected: set[str] = set()
    if args.copy_new:
        selected.update(result["official_only"])
    if args.sync_list:
        selected.update(read_sync_list(args.sync_list.expanduser().resolve()))
    return sorted(selected)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.is_dir():
        raise ValueError(f"目标是目录，拒绝覆盖：{destination}")

    if source.is_symlink():
        temporary = destination.parent / (
            f".{destination.name}.sync-{os.getpid()}"
        )
        try:
            temporary.symlink_to(os.readlink(source))
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return

    if not source.is_file():
        raise ValueError(f"仅支持同步普通文件或符号链接：{source}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.sync-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary, follow_symlinks=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def preview_or_apply(
    official_root: Path,
    internal_root: Path,
    selected: Iterable[str],
    apply: bool,
) -> tuple[int, int]:
    planned = 0
    completed = 0
    for relative in selected:
        source = official_root / relative
        destination = internal_root / relative
        if not source.is_file() and not source.is_symlink():
            print(f"[跳过] 官方文件不存在或不是普通文件：{relative}", file=sys.stderr)
            continue
        planned += 1
        if apply:
            atomic_copy(source, destination)
            completed += 1
            print(f"[已同步] {relative}")
        else:
            print(f"[预览] {relative}")
    return planned, completed


def main() -> int:
    args = parse_args()
    try:
        official = normalize_root(args.official, "官方")
        internal = normalize_root(args.internal, "内部")
        if official == internal:
            raise ValueError("官方目录和内部目录不能相同")

        default_excludes = () if args.no_default_excludes else DEFAULT_EXCLUDES
        excludes = tuple(default_excludes) + tuple(args.exclude)
        result = compare_trees(official, internal, excludes)

        print("差异汇总：")
        for status in STATUSES:
            print(f"  {status:14s} {len(result[status])}")

        if args.report:
            write_report(args.report, official, internal, excludes, result)
            print(f"完整报告：{args.report.expanduser().resolve()}")

        if args.diff_output:
            text_count, metadata_count = write_diffs(
                args.diff_output,
                official,
                internal,
                result,
                args.max_diff_bytes,
            )
            print(
                f"diff 输出：{args.diff_output.expanduser().resolve()} "
                f"（文本 {text_count}，元数据 {metadata_count}）"
            )

        selected = select_paths(args, result)
        if args.apply and not selected:
            raise ValueError("--apply 必须与 --copy-new 或 --sync-list 一起使用")
        if selected:
            planned, completed = preview_or_apply(
                official, internal, selected, args.apply
            )
            if args.apply:
                print(f"同步完成：计划 {planned}，成功 {completed}")
            else:
                print(f"仅预览：共 {planned} 个文件；添加 --apply 后才会修改")
        elif not args.report and not args.diff_output:
            print("提示：使用 --report FILE 输出完整文件清单。")
        return 0
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
