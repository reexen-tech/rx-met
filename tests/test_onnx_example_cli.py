"""The direct ONNX example can consume a KWS PTQ export and shared feeds."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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
            result = subprocess.run([
                sys.executable, example.__file__, "--device", "cpu", "--model", str(root / "scalar.onnx"),
                "--calib-dir", str(calibration), "--output-dir", str(root / "output"), "--prefix", "scalar",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            archive, = (root / "output").glob("*/exports/outputs.npz")
            with np.load(archive) as outputs:
                np.testing.assert_array_equal(outputs["fp32_output0"], np.array([1.0, 3.0], dtype=np.float32))
                self.assertEqual(outputs["ptq_output0"].shape, (2,))
                self.assertEqual(outputs["po2_output0"].shape, (2,))

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


if __name__ == "__main__":
    unittest.main()
