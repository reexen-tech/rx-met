"""The ONNX example accepts explicit models and discovers successful KWS exports."""

from pathlib import Path
import ast
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper

from examples import quick_start_onnx_ptq as example
from tests.support import onnx as fixtures


class OnnxExampleCliTest(unittest.TestCase):
    def test_removing_entire_po2_stage_preserves_affine_bias_export_and_reload(self):
        fixture = fixtures.OnnxFixture(self)
        root = fixture.path
        model_path = root / "linear.onnx"
        onnx.save(fixtures.make_graph(), model_path)
        calibration = root / "calib"
        calibration.mkdir()
        for index, offset in enumerate((0.0, 1.0)):
            np.save(calibration / f"{index}.npy", fixture.feed["x"] + offset)
        mixed = root / "mixed.json"
        mixed.write_text("{}")
        source = Path(example.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        stage, = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                  and isinstance(node.items[0].context_expr, ast.Call)
                  and node.items[0].context_expr.args
                  and isinstance(node.items[0].context_expr.args[0], ast.Constant)
                  and str(node.items[0].context_expr.args[0].value).startswith("步骤 5:")]
        lines = source.splitlines(keepends=True)
        del lines[stage.lineno - 1:stage.end_lineno]
        source = "".join(lines)
        command = (f"exec(compile({source!r}, {example.__file__!r}, 'exec'), "
                   f"{{'__name__': '__main__', '__file__': {example.__file__!r}}})")
        result = subprocess.run([
            sys.executable, "-c", command, "--device", "cpu", "--model", str(model_path),
            "--calib-dir", str(calibration), "--output-dir", str(root / "output"),
            "--prefix", "affine", "--config-file", str(fixture.config), "--quant-config", str(mixed),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metadata_path, = (root / "output").glob("*/exports/affine.meta.json")
        self.assertFalse(list(metadata_path.parent.glob("*.npz")))
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(set(metadata["evaluation"]), {"fp32", "ptq"})
        self.assertNotIn("po2_vs_fp32", metadata)
        self.assertNotIn("all_power_of_2", metadata)
        self.assertEqual(metadata["bias_align"]["modified"], 1)
        self.assertTrue(all(item["max_abs"] == 0 for item in metadata["restored_quantized_vs_pre_export"]))
        payload = json.loads((metadata_path.parent / "affine.json").read_text())
        bias, weight = (payload["param_encodings"][name] for name in ("b", "w"))
        self.assertEqual(bias["bitwidth"], 32)
        input_scale = payload["activation_encodings"]["fc"]["input"]["0"]["scale"]
        np.testing.assert_array_equal(bias["scale"], np.asarray(weight["scale"]) * input_scale)

    def test_scalar_onnx_outputs_are_evaluated_and_restored(self):
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
            environment = dict(os.environ, RX_MET_ONNX_PTQ_MODEL="/removed/model.onnx",
                               RX_MET_ONNX_PTQ_CALIB="/removed/calib", RX_MET_ONNX_PTQ_VAL="/removed/val",
                               RX_MET_ONNX_PTQ_OUTPUT_DIR="/removed/output", RX_MET_ONNX_PTQ_DEVICE="invalid")
            result = subprocess.run([
                sys.executable, example.__file__, "--device", "cpu", "--model", str(root / "scalar.onnx"),
                "--calib-dir", str(calibration), "--output-dir", str(output), "--prefix", "scalar",
            ], capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metadata_path, = output.glob("*/exports/scalar.meta.json")
            self.assertFalse(list(metadata_path.parent.glob("*.npz")))
            metadata = json.loads(metadata_path.read_text())
            for stage in ("fp32", "ptq", "po2"):
                self.assertEqual(metadata["evaluation"][stage]["feed_count"], 2)
            restored = metadata["restored_quantized_vs_pre_export"][0]
            self.assertEqual(restored["value_count"], 2)
            self.assertEqual(restored["max_abs"], 0)
            self.assertEqual(onnx.load(metadata_path.parent / "scalar.onnx").graph.output[0].type.tensor_type.shape.dim, [])
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "output" / "quick_start_kws"
            for kind in ("gru", "lstm"):
                for index, status in ((1, "success"), (2, "success"), (3, "failed"), (4, "running")):
                    run = outputs / kind / f"20261118_12000{index}_000000"
                    exports = run / "exports"
                    exports.mkdir(parents=True)
                    (exports / "att_mh_rnn_kws.onnx").touch()
                    (run / "run.json").write_text(json.dumps({
                        "status": status, "started_at": f"2026-11-18T12:00:0{index}+08:00",
                    }))
                stale = outputs / kind / "20261118_120005_000000"
                (stale / "exports").mkdir(parents=True)
                (stale / "exports/att_mh_rnn_kws.onnx").touch()
                (stale / "run.json").write_text(json.dumps({
                    "status": "success", "started_at": "2026-11-18T12:00:05+08:00",
                    "outputs": {"exports/att_mh_rnn_kws_ptq.onnx": {}},
                }))
                with patch.object(Path, "cwd", return_value=root):
                    args = example.parse_args([] if kind == "gru" else ["--rnn-type", kind])
                    model = example.resolve_model_path(args)
                self.assertEqual(model, outputs / kind / "20261118_120002_000000/exports/att_mh_rnn_kws.onnx")

    def test_model_discovery_handles_direct_exports_and_bad_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output/quick_start_kws/gru"
            for name, record in (("invalid", "{"), ("null", "null"), ("missing", None),
                                 ("bad-start", '{"status":"success","started_at":null}')):
                run = output / name
                (run / "exports").mkdir(parents=True)
                (run / "exports/att_mh_rnn_kws.onnx").touch()
                if record is not None:
                    (run / "run.json").write_text(record)
            with patch.object(Path, "cwd", return_value=root):
                with self.assertRaisesRegex(FileNotFoundError, "--model"):
                    example.resolve_model_path(example.parse_args([]))
                explicit = example.parse_args(["--model", "/models/custom.onnx"])
                self.assertEqual(example.resolve_model_path(explicit), Path("/models/custom.onnx"))
                (output / "exports").mkdir()
                expected = output / "exports/att_mh_rnn_kws.onnx"
                expected.touch()
                (output / "run.json").write_text(json.dumps({"status": "success"}))
                self.assertEqual(example.resolve_model_path(example.parse_args([])), expected)

    def test_defaults_and_explicit_options_ignore_removed_environment(self):
        environment = {
            "RX_MET_SPEECH_COMMANDS_ROOT": "/removed/data",
            "RX_MET_ONNX_PTQ_OUTPUT_DIR": "/removed/output",
            "RX_MET_ONNX_PTQ_DEVICE": "invalid",
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment):
            root = Path(directory)
            with patch.object(Path, "cwd", return_value=root):
                namespace = runpy.run_path(example.__file__, run_name="onnx_defaults_test")
                args = namespace["parse_args"]([])
                self.assertIsNone(args.model)
                self.assertEqual(args.rnn_type, "gru")
                self.assertEqual(args.data_dir, Path("/datasets/speech_commands_v0.02"))
                self.assertEqual(args.output_dir, root / "output/quick_start_onnx_ptq")
                self.assertEqual(args.device, "cuda")
                args = namespace["parse_args"]([
                    "--model", "/models/custom.onnx",
                    "--data-dir", "/custom/data", "--output-dir", "/custom/output", "--device", "cpu",
                ])
                self.assertEqual(args.data_dir, Path("/custom/data"))
                self.assertEqual(args.output_dir, Path("/custom/output"))
                self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
