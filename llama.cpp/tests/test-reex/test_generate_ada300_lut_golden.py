"""Regression tests for the ADA300 bit-golden generator."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


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

    def test_supported_precisions_are_stable(self) -> None:
        self.assertEqual(GENERATOR.SUPPORTED_PRECISIONS, ("mixed_fp16", "fp32"))

    def test_sigmoid_special_values_use_composed_path(self) -> None:
        class Runner:
            @staticmethod
            def exp_eval(_bundle, x, _precision):
                return np.zeros_like(x, dtype=np.float32)

            @staticmethod
            def normalized_eval(_bundle, _op, x, _precision):
                return np.full_like(x, np.float32(0.75))

        inputs = np.array([0.0, -0.0, np.inf], dtype=np.float32)
        outputs = GENERATOR.eval_op(
            "sigmoid", inputs, Runner(), {"exp2": {}, "reciprocal": {}}, "fp32")
        np.testing.assert_array_equal(outputs, np.full_like(inputs, np.float32(0.75)))


if __name__ == "__main__":
    unittest.main()
