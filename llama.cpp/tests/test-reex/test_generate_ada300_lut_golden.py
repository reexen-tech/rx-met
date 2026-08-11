"""Regression tests for the ADA300 bit-golden generator."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


GENERATOR_PATH = Path(__file__).with_name("generate_ada300_lut_golden.py")
SPEC = importlib.util.spec_from_file_location("generate_ada300_lut_golden", GENERATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {GENERATOR_PATH}")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class Ada300GoldenGeneratorTest(unittest.TestCase):
    def test_segment_seeds_are_stable(self) -> None:
        self.assertEqual(GENERATOR.seed_for_segments(16), 0xADA30016)
        self.assertEqual(GENERATOR.seed_for_segments(31), 0xADA3001F)
        self.assertEqual(GENERATOR.seed_for_segments(63), 0xADA3003F)

    def test_unsupported_segment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GENERATOR.seed_for_segments(64)


if __name__ == "__main__":
    unittest.main()
