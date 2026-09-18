#!/usr/bin/env python3
"""Generate and verify rx-met release dependency artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ImportError:  # Python 3.10 build images
    from pip._vendor import tomli as tomllib

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docker" / "variants.json"
BAKE_PATH = ROOT / "docker" / "docker-bake.hcl"
LOCK_DIR = ROOT / "docker" / "requirements"
BUILD_LOCK = LOCK_DIR / "build.lock"


class DependencyError(ValueError):
    """Raised when the dependency manifest and generated artifacts disagree."""


def _canonical_packages(packages: Mapping[str, str], location: str) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_name, raw_version in packages.items():
        name = canonicalize_name(raw_name)
        if name != raw_name:
            raise DependencyError(
                f"{location}: package name must be canonical: {raw_name} -> {name}"
            )
        version = str(raw_version).strip()
        if not version:
            raise DependencyError(f"{location}: empty version for {name}")
        try:
            Version(version)
        except ValueError as error:
            raise DependencyError(f"{location}: invalid version {name}={version}") from error
        previous = normalized.setdefault(name, version)
        if previous != version:
            raise DependencyError(
                f"{location}: conflicting versions for {name}: {previous}, {version}"
            )
    return normalized


def load_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    try:
        matrix = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DependencyError(f"cannot read dependency matrix {path}: {error}") from error
    if matrix.get("schema_version") != 1:
        raise DependencyError(f"{path}: unsupported schema_version")
    if not str(matrix.get("environment_version", "")).strip():
        raise DependencyError(f"{path}: missing environment_version")

    matrix["build_packages"] = _canonical_packages(
        matrix.get("build_packages", {}), "build_packages"
    )
    matrix["common_packages"] = _canonical_packages(
        matrix.get("common_packages", {}), "common_packages"
    )
    if not matrix["build_packages"] or not matrix["common_packages"]:
        raise DependencyError(f"{path}: build_packages and common_packages are required")
    variants = matrix.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise DependencyError(f"{path}: variants must be a non-empty object")
    invalid_names = [name for name in variants if not name.startswith("cu")]
    if invalid_names:
        raise DependencyError(f"{path}: invalid variant names: {', '.join(invalid_names)}")

    required_fields = {
        "python_version",
        "cuda_version",
        "devel_image",
        "base_image",
        "cuda_architectures",
        "torch_cuda_arch_list",
        "minimum_driver_version",
        "torch",
        "onnxruntime",
    }
    for name, variant in variants.items():
        missing = required_fields - set(variant)
        if missing:
            raise DependencyError(f"{name}: missing fields: {', '.join(sorted(missing))}")
        for field in ("cuda_architectures", "torch_cuda_arch_list"):
            values = variant[field]
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value for value in values
            ):
                raise DependencyError(f"{name}.{field}: expected a non-empty string list")
        for group in ("torch", "onnxruntime"):
            config = variant[group]
            if not str(config.get("index_url", "")).strip():
                raise DependencyError(f"{name}.{group}: missing index_url")
            config["packages"] = _canonical_packages(
                config.get("packages", {}), f"{name}.{group}.packages"
            )
        variant["torch"].setdefault("extra_index_urls", [])
        if not isinstance(variant["torch"]["extra_index_urls"], list):
            raise DependencyError(f"{name}.torch.extra_index_urls: expected a list")
        required_torch = {"torch", "torchvision", "torchaudio"}
        missing_torch = required_torch - set(variant["torch"]["packages"])
        if missing_torch:
            raise DependencyError(
                f"{name}.torch.packages: missing {', '.join(sorted(missing_torch))}"
            )
        if set(variant["onnxruntime"]["packages"]) != {"onnxruntime-gpu"}:
            raise DependencyError(
                f"{name}.onnxruntime.packages: expected only onnxruntime-gpu"
            )
        source_sha256 = str(variant["onnxruntime"].get("source_sha256", ""))
        if source_sha256 != "vendored" and not (
            len(source_sha256) == 64
            and all(character in "0123456789abcdef" for character in source_sha256)
        ):
            raise DependencyError(f"{name}.onnxruntime: invalid source_sha256")
        overlap = set(matrix["common_packages"]) & (
            set(variant["torch"]["packages"])
            | set(variant["onnxruntime"]["packages"])
        )
        if overlap:
            raise DependencyError(
                f"{name}: packages declared in common and variant groups: "
                + ", ".join(sorted(overlap))
            )
    return matrix


def load_project_requirements() -> list[Requirement]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"].get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise DependencyError("pyproject.toml must define project.dependencies")
    return [Requirement(line) for line in dependencies]


def runtime_packages(matrix: Mapping[str, Any], variant_name: str) -> dict[str, str]:
    variant = matrix["variants"][variant_name]
    result = dict(matrix["common_packages"])
    for group in ("torch", "onnxruntime"):
        for name, version in variant[group]["packages"].items():
            previous = result.setdefault(name, version)
            if previous != version:
                raise DependencyError(
                    f"{variant_name}: conflicting versions for {name}: "
                    f"{previous}, {version}"
                )
    return result


def variant_lock_path(matrix: Mapping[str, Any], variant_name: str) -> Path:
    python_tag = str(matrix["variants"][variant_name]["python_version"]).replace(".", "")
    return LOCK_DIR / f"{variant_name}-py{python_tag}.lock"


def validate_project_compatibility(
    matrix: Mapping[str, Any], requirements: list[Requirement]
) -> None:
    for requirement in requirements:
        if requirement.marker is not None or requirement.url is not None:
            raise DependencyError(
                f"project dependency must be an unconditional version range: {requirement}"
            )
        name = canonicalize_name(requirement.name)
        for variant_name in matrix["variants"]:
            packages = runtime_packages(matrix, variant_name)
            if name not in packages:
                raise DependencyError(
                    f"{variant_name}: project dependency is not locked: {name}"
                )
            version = Version(packages[name])
            if requirement.specifier and not requirement.specifier.contains(
                version, prereleases=True
            ):
                raise DependencyError(
                    f"{variant_name}: {name}=={version} is outside "
                    f"{requirement.specifier}"
                )


def render_lock(packages: Mapping[str, str], description: str) -> str:
    lines = [
        "# Generated by 'python scripts/dependencies.py lock'; do not edit.",
        f"# {description}",
    ]
    lines.extend(
        f"{name}=={version}"
        for name, version in sorted(packages.items(), key=lambda item: item[0])
    )
    return "\n".join(lines) + "\n"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def render_bake(matrix: Mapping[str, Any]) -> str:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    environment_version = matrix["environment_version"]
    lines = [
        "# Generated by 'python scripts/dependencies.py lock'; do not edit.",
        "",
        'variable "VERSION" {',
        f'  default = "{version}"',
        "}",
        "",
        'variable "IMAGE_REPOSITORY" {',
        '  default = "rx-met"',
        "}",
        "",
        'variable "SOURCE_REVISION" {',
        '  default = "unknown"',
        "}",
        "",
        'variable "ENV_VERSION" {',
        f'  default = "{environment_version}"',
        "}",
        "",
        'variable "BUILD_ENV_REPOSITORY" {',
        '  default = "rx-met-build-env"',
        "}",
        "",
        'variable "RUNTIME_ENV_REPOSITORY" {',
        '  default = "rx-met-runtime-env"',
        "}",
        "",
        'group "default" {',
        "  targets = ["
        + ", ".join(f'"{name}"' for name in matrix["variants"])
        + "]",
        "}",
        "",
        'group "environments" {',
        "  targets = [",
    ]
    for name in matrix["variants"]:
        lines.extend(
            [
                f'    "environment-build-{name}",',
                f'    "environment-runtime-{name}",',
            ]
        )
    lines.extend(["  ]", "}", ""])

    for name, variant in matrix["variants"].items():
        torch = variant["torch"]["packages"]
        ort = variant["onnxruntime"]["packages"]
        lines.extend(
            [
                f'target "variant-{name}" {{',
                "  args = {",
                f'    CUDA_VARIANT              = "{name}"',
                f'    RX_MET_CUDA_VERSION       = "{variant["cuda_version"]}"',
                f'    DEVEL_IMAGE               = "{variant["devel_image"]}"',
                f'    BASE_IMAGE                = "{variant["base_image"]}"',
                '    RX_MET_CUDA_ARCHITECTURES = "'
                + ";".join(variant["cuda_architectures"])
                + '"',
                '    TORCH_CUDA_ARCH_LIST      = "'
                + ";".join(variant["torch_cuda_arch_list"])
                + '"',
                f'    TORCH_VERSION             = "{_base_version(torch["torch"])}"',
                f'    TORCHVISION_VERSION       = "{_base_version(torch["torchvision"])}"',
                f'    TORCHAUDIO_VERSION        = "{_base_version(torch["torchaudio"])}"',
                f'    ONNXRUNTIME_VERSION       = "{ort["onnxruntime-gpu"]}"',
                f'    ONNXRUNTIME_SOURCE_SHA256 = "{variant["onnxruntime"]["source_sha256"]}"',
                f'    PYTHON_VERSION            = "{variant["python_version"]}"',
                f'    MIN_DRIVER_VERSION        = "{variant["minimum_driver_version"]}"',
                "  }",
                "}",
                "",
            ]
        )

    lines.extend(
        [
            'target "environment-build" {',
            '  context    = "."',
            '  dockerfile = "docker/Dockerfile.environment"',
            '  target     = "build-env"',
            '  platforms  = ["linux/amd64"]',
            "  pull       = false",
            "  args = {",
            "    ENV_VERSION = ENV_VERSION",
            "  }",
            "  labels = {",
            '    "org.opencontainers.image.title"   = "rx-met build environment"',
            '    "org.opencontainers.image.version" = ENV_VERSION',
            "  }",
            "}",
            "",
            'target "environment-runtime" {',
            '  context    = "."',
            '  dockerfile = "docker/Dockerfile.environment"',
            '  target     = "runtime-env"',
            '  platforms  = ["linux/amd64"]',
            "  pull       = false",
            "  args = {",
            "    ENV_VERSION = ENV_VERSION",
            "  }",
            "  labels = {",
            '    "org.opencontainers.image.title"   = "rx-met runtime environment"',
            '    "org.opencontainers.image.version" = ENV_VERSION',
            "  }",
            "}",
            "",
        ]
    )
    for name in matrix["variants"]:
        lines.extend(
            [
                f'target "environment-build-{name}" {{',
                f'  inherits = ["environment-build", "variant-{name}"]',
                f'  tags     = ["${{BUILD_ENV_REPOSITORY}}:${{ENV_VERSION}}-{name}"]',
                "}",
                "",
                f'target "environment-runtime-{name}" {{',
                f'  inherits = ["environment-runtime", "variant-{name}"]',
                f'  tags     = ["${{RUNTIME_ENV_REPOSITORY}}:${{ENV_VERSION}}-{name}"]',
                "}",
                "",
            ]
        )
    lines.extend(
        [
            'target "release" {',
            '  context    = "."',
            '  dockerfile = "docker/Dockerfile"',
            '  target     = "runtime"',
            '  platforms  = ["linux/amd64"]',
            "  pull       = false",
            "  args = {",
            "    ENV_VERSION     = ENV_VERSION",
            "    RX_MET_VERSION  = VERSION",
            "    SOURCE_REVISION = SOURCE_REVISION",
            "  }",
            "  labels = {",
            '    "org.opencontainers.image.title"    = "rx-met"',
            '    "org.opencontainers.image.version"  = VERSION',
            '    "org.opencontainers.image.revision" = SOURCE_REVISION',
            "  }",
            "}",
            "",
        ]
    )
    for name in matrix["variants"]:
        lines.extend(
            [
                f'target "{name}" {{',
                f'  inherits = ["release", "variant-{name}"]',
                f'  tags     = ["${{IMAGE_REPOSITORY}}:${{VERSION}}-{name}"]',
                "  args = {",
                f'    BUILD_ENV_IMAGE   = "${{BUILD_ENV_REPOSITORY}}:${{ENV_VERSION}}-{name}"',
                f'    RUNTIME_ENV_IMAGE = "${{RUNTIME_ENV_REPOSITORY}}:${{ENV_VERSION}}-{name}"',
                "  }",
                "}",
                "",
            ]
        )
    return "\n".join(lines)


def write_generated_artifacts(matrix: Mapping[str, Any]) -> None:
    outputs = {
        BAKE_PATH: render_bake(matrix),
        BUILD_LOCK: render_lock(matrix["build_packages"], "Python build tools."),
    }
    for variant_name in matrix["variants"]:
        variant = matrix["variants"][variant_name]
        outputs[variant_lock_path(matrix, variant_name)] = render_lock(
            runtime_packages(matrix, variant_name),
            f"Complete runtime environment for {variant_name} / "
            f"Python {variant['python_version']}.",
        )
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") == content:
            print(f"unchanged {path.relative_to(ROOT)}")
            continue
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


def check_generated_artifacts(matrix: Mapping[str, Any]) -> None:
    validate_project_compatibility(matrix, load_project_requirements())
    expected = {
        BAKE_PATH: render_bake(matrix),
        BUILD_LOCK: render_lock(matrix["build_packages"], "Python build tools."),
    }
    for variant_name in matrix["variants"]:
        variant = matrix["variants"][variant_name]
        expected[variant_lock_path(matrix, variant_name)] = render_lock(
            runtime_packages(matrix, variant_name),
            f"Complete runtime environment for {variant_name} / "
            f"Python {variant['python_version']}.",
        )
    stale = [
        str(path.relative_to(ROOT))
        for path, content in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    unexpected_locks = sorted(
        str(path.relative_to(ROOT))
        for path in LOCK_DIR.glob("*.lock")
        if path not in expected
    )
    if stale:
        raise DependencyError(
            "generated dependency artifacts are stale: "
            + ", ".join(stale)
            + "; run 'python scripts/dependencies.py lock'"
        )
    if unexpected_locks:
        raise DependencyError(
            "unexpected generated locks: " + ", ".join(unexpected_locks)
        )
    print("dependency configuration check passed")


def _write_requirement_file(path: Path, packages: Mapping[str, str]) -> None:
    path.write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in sorted(packages.items(), key=lambda item: item[0])
        )
        + "\n",
        encoding="utf-8",
    )


def _pip_download(
    requirement_file: Path,
    output_dir: Path,
    *,
    index_url: str | None = None,
    extra_index_urls: list[str] | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(output_dir),
        "--only-binary=:all:",
        "--no-deps",
    ]
    if index_url:
        command.extend(["--index-url", index_url])
    for url in extra_index_urls or []:
        command.extend(["--extra-index-url", url])
    command.extend(["-r", str(requirement_file)])
    subprocess.run(command, check=True)


def _load_wheelhouse_verifier() -> Any:
    path = ROOT / "scripts" / "internal" / "verify_dependency_wheelhouse.py"
    spec = importlib.util.spec_from_file_location("rx_met_wheelhouse_verifier", path)
    if spec is None or spec.loader is None:
        raise DependencyError(f"cannot load wheelhouse verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_wheelhouse(output_dir: Path, lock_paths: list[Path]) -> bool:
    try:
        _load_wheelhouse_verifier().verify(output_dir, lock_paths)
    except (ValueError, OSError, zipfile.BadZipFile):
        return False
    return True


def _verify_checksum_manifest(output_dir: Path) -> bool:
    checksum_file = output_dir / "SHA256SUMS"
    if not checksum_file.is_file():
        return False
    expected: dict[str, str] = {}
    try:
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            filename = filename.lstrip("*")
            if filename.startswith("./"):
                filename = filename[2:]
            if len(digest) != 64 or filename in expected:
                return False
            expected[filename] = digest
    except (OSError, ValueError):
        return False
    wheels = {path.name: path for path in output_dir.glob("*.whl")}
    if set(expected) != set(wheels):
        return False
    return all(
        hashlib.sha256(wheels[name].read_bytes()).hexdigest() == digest
        for name, digest in expected.items()
    )


def download_wheelhouse(
    matrix: Mapping[str, Any], variant_name: str, output_dir: Path
) -> None:
    variant = matrix["variants"][variant_name]
    lock_paths = [BUILD_LOCK, variant_lock_path(matrix, variant_name)]
    output_dir.mkdir(parents=True, exist_ok=True)
    if _verify_wheelhouse(output_dir, lock_paths) and _verify_checksum_manifest(
        output_dir
    ):
        print(f"wheelhouse already complete: {output_dir}")
        return
    for path in output_dir.glob("*.whl"):
        path.unlink()
    checksum_file = output_dir / "SHA256SUMS"
    checksum_file.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="rx-met-requirements-") as temporary:
        temp = Path(temporary)
        groups = {
            "build": matrix["build_packages"],
            "common": matrix["common_packages"],
            "torch": variant["torch"]["packages"],
            "onnxruntime": variant["onnxruntime"]["packages"],
        }
        requirement_files = {}
        for group, packages in groups.items():
            requirement_file = temp / f"{group}.txt"
            _write_requirement_file(requirement_file, packages)
            requirement_files[group] = requirement_file
        _pip_download(requirement_files["build"], output_dir)
        _pip_download(requirement_files["common"], output_dir)
        _pip_download(
            requirement_files["torch"],
            output_dir,
            index_url=variant["torch"]["index_url"],
            extra_index_urls=variant["torch"]["extra_index_urls"],
        )
        _pip_download(
            requirement_files["onnxruntime"],
            output_dir,
            index_url=variant["onnxruntime"]["index_url"],
        )

    verifier = _load_wheelhouse_verifier()
    try:
        verifier.verify(output_dir, lock_paths)
    except (ValueError, zipfile.BadZipFile) as error:
        raise DependencyError(f"downloaded wheelhouse is invalid: {error}") from error
    checksum_file.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(output_dir.glob("*.whl"))
        ),
        encoding="utf-8",
    )
    print(f"wheelhouse ready: {output_dir}")


def main() -> None:
    try:
        matrix = load_matrix()
    except (DependencyError, OSError) as error:
        raise SystemExit(f"dependency operation failed: {error}") from None

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("lock", help="regenerate locks and Docker Bake configuration")
    subparsers.add_parser("check", help="verify manifest and generated artifacts")
    download_parser = subparsers.add_parser(
        "download", help="download one variant's complete offline wheelhouse"
    )
    download_parser.add_argument(
        "--variant", choices=tuple(matrix["variants"]), required=True
    )
    download_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "lock":
            validate_project_compatibility(matrix, load_project_requirements())
            write_generated_artifacts(matrix)
        elif args.command == "check":
            check_generated_artifacts(matrix)
        else:
            check_generated_artifacts(matrix)
            download_wheelhouse(matrix, args.variant, args.output)
    except (DependencyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"dependency operation failed: {error}") from None


if __name__ == "__main__":
    main()
