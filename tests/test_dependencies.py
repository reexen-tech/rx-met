"""Tests for the generated release dependency configuration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dependencies = _load_module("rx_met_dependencies", ROOT / "scripts/dependencies.py")
wheelhouse = _load_module(
    "rx_met_wheelhouse",
    ROOT / "scripts/internal/verify_dependency_wheelhouse.py",
)


class DependencyConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = dependencies.load_matrix()

    def test_project_ranges_accept_every_release_variant(self) -> None:
        dependencies.validate_project_compatibility(
            self.matrix, dependencies.load_project_requirements()
        )

    def test_generated_artifacts_are_current(self) -> None:
        dependencies.check_generated_artifacts(self.matrix)

    def test_variant_locks_are_complete_and_self_contained(self) -> None:
        for variant_name in self.matrix["variants"]:
            expected = dependencies.runtime_packages(self.matrix, variant_name)
            actual = wheelhouse.load_requirements(
                [dependencies.variant_lock_path(self.matrix, variant_name)]
            )
            self.assertEqual(expected, actual)

    def test_build_lock_matches_manifest(self) -> None:
        actual = wheelhouse.load_requirements([dependencies.BUILD_LOCK])
        self.assertEqual(self.matrix["build_packages"], actual)

    def test_manifest_rejects_incomplete_torch_group(self) -> None:
        invalid = json.loads(dependencies.MATRIX_PATH.read_text(encoding="utf-8"))
        del invalid["variants"]["cu126"]["torch"]["packages"]["torchvision"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "variants.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(dependencies.DependencyError):
                dependencies.load_matrix(path)

    def test_checksum_manifest_accepts_sha256sum_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            wheel = directory / "example-1.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            (directory / "SHA256SUMS").write_text(
                f"{digest}  ./{wheel.name}\n", encoding="utf-8"
            )
            self.assertTrue(dependencies._verify_checksum_manifest(directory))


if __name__ == "__main__":
    unittest.main()
