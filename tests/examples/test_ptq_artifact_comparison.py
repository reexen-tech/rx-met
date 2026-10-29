"""The artifact comparison CLI reports every difference without tolerances."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from tests.support.paths import REPO_ROOT


ROOT = REPO_ROOT
COMMAND = ROOT / "scripts/validation/compare_ptq_artifacts.py"


class PtqArtifactComparisonTest(unittest.TestCase):
    def test_cli_preserves_a_report_when_outputs_contain_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = helper.make_graph(
                [helper.make_node("Identity", ["input"], ["output"])], "nonfinite",
                [helper.make_tensor_value_info("input", 1, [1])],
                [helper.make_tensor_value_info("output", 1, [1])],
            )
            onnx.save(helper.make_model(graph), root / "model.onnx")
            (root / "encodings.json").write_text('{}')
            np.savez(root / "left.npz", ptq_output0=np.array([np.inf], dtype=np.float32))
            np.savez(root / "right.npz", ptq_output0=np.array([1.0], dtype=np.float32))
            report = root / "report.json"
            result = subprocess.run([
                sys.executable, str(COMMAND), "--left-onnx", str(root / "model.onnx"),
                "--right-onnx", str(root / "model.onnx"), "--left-encodings", str(root / "encodings.json"),
                "--right-encodings", str(root / "encodings.json"), "--left-outputs", str(root / "left.npz"),
                "--right-outputs", str(root / "right.npz"), "--report", str(report),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            comparison = json.loads(report.read_text())
            self.assertFalse(comparison["equal"])
            self.assertEqual(comparison["outputs"]["differences"][0]["nonfinite_elements"], 1)
            self.assertIsNone(comparison["outputs"]["differences"][0]["max_abs"])

    def test_cli_reports_initializer_and_nested_encoding_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("cpu", 1.0), ("cuda", 1.125)):
                graph = helper.make_graph(
                    [helper.make_node("Add", ["input", "weight"], ["output"])],
                    "comparison", [helper.make_tensor_value_info("input", 1, [1])],
                    [helper.make_tensor_value_info("output", 1, [1])],
                    [numpy_helper.from_array(np.array([value], dtype=np.float32), "weight")],
                )
                onnx.save(helper.make_model(graph), root / f"{name}.onnx")
                (root / f"{name}.json").write_text(json.dumps({
                    "activation_encodings": {"rnn": {"output": [{"scale": [value]}]}}
                }))
            report = root / "report.json"
            result = subprocess.run([
                sys.executable, str(COMMAND), "--left-onnx", str(root / "cpu.onnx"),
                "--right-onnx", str(root / "cuda.onnx"), "--left-encodings", str(root / "cpu.json"),
                "--right-encodings", str(root / "cuda.json"), "--report", str(report),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            comparison = json.loads(report.read_text())
            self.assertFalse(comparison["equal"])
            self.assertTrue(comparison["onnx"]["structure"]["equal"])
            weight = comparison["onnx"]["initializers"]["differences"][0]
            self.assertEqual(weight["name"], "weight")
            self.assertEqual(weight["max_abs"], 0.125)
            self.assertEqual(weight["element_differences"], [
                {"index": [0], "left": 1.0, "right": 1.125},
            ])
            self.assertEqual(comparison["encodings"]["differences"][0]["path"],
                             "/activation_encodings/rnn/output/0/scale/0")


if __name__ == "__main__":
    unittest.main()
