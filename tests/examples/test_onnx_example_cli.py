"""The direct ONNX example can consume a KWS PTQ export and shared feeds."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper

from examples import quick_start_onnx_ptq as example


class OnnxExampleCliTest(unittest.TestCase):
    def test_scalar_onnx_outputs_are_archived_per_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = helper.make_graph(
                [helper.make_node("Relu", ["input"], ["output"], name="relu")], "scalar",
                [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [])],
                [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [])],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8
            onnx.save(model, root / "scalar.onnx")
            calibration = root / "calib"
            calibration.mkdir()
            for index, value in enumerate((1.0, 3.0)):
                np.savez(calibration / f"{index}.npz", input=np.array(value, dtype=np.float32))
            output = root / "output"
            result = subprocess.run([
                sys.executable, example.__file__, "--device", "cpu", "--model", str(root / "scalar.onnx"),
                "--calib-dir", str(calibration), "--output-dir", str(output), "--prefix", "scalar",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archive, = output.glob("*/exports/outputs.npz")
            with np.load(archive) as outputs:
                np.testing.assert_array_equal(outputs["fp32_output0"], np.array([1.0, 3.0], dtype=np.float32))
                self.assertEqual(outputs["ptq_output0"].shape, (2,))
                self.assertEqual(outputs["po2_output0"].shape, (2,))
            metadata = json.loads((archive.parent / "scalar.meta.json").read_text())
            self.assertIn("po2", metadata["evaluation"])
            self.assertNotIn("example", metadata)

    def test_cpu_accepts_explicit_kws_model_and_feed_paths(self):
        args = example.parse_args([
            "--model", "/models/att_mh_rnn_kws_ptq.onnx", "--device", "cpu",
            "--calib-dir", "/shared/calib", "--validation-dir", "/shared/val",
            "--output-dir", "/output/cpu", "--prefix", "kws_ptq",
            "--calib-limit", "2", "--validation-limit", "3",
        ])
        self.assertEqual(args.model, Path("/models/att_mh_rnn_kws_ptq.onnx"))
        self.assertEqual(args.calib_dir, Path("/shared/calib"))
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.prefix, "kws_ptq")
        self.assertEqual(args.calib_limit, 2)
        self.assertEqual(args.validation_limit, 3)

    def test_default_model_is_latest_successful_export_of_selected_type(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            os.environ["RX_MET_KWS_OUTPUT_DIR"] = str(root)
            for kind in ("gru", "lstm"):
                for index, status in ((1, "success"), (2, "success"), (3, "failed")):
                    run = root / kind / f"20261118_12000{index}_000000"
                    exports = run / "exports"
                    exports.mkdir(parents=True)
                    (exports / "att_mh_rnn_kws.onnx").touch()
                    (run / "run.json").write_text(json.dumps({
                        "status": status, "started_at": f"2026-11-18T12:00:0{index}+08:00",
                    }))
                model = example.resolve_model_path(example.parse_args(["--rnn-type", kind]))
                self.assertEqual(model, root / kind / "20261118_120002_000000/exports/att_mh_rnn_kws.onnx")
            explicit = example.parse_args(["--model", "/models/custom.onnx"])
            self.assertEqual(example.resolve_model_path(explicit), Path("/models/custom.onnx"))
            os.environ["RX_MET_ONNX_PTQ_MODEL"] = "/models/env.onnx"
            self.assertEqual(example.resolve_model_path(example.parse_args([])), Path("/models/env.onnx"))

    def test_missing_default_model_reports_explicit_model_option(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"RX_MET_KWS_OUTPUT_DIR": directory}, clear=True,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "--model"):
                example.resolve_model_path(example.parse_args([]))


if __name__ == "__main__":
    unittest.main()
