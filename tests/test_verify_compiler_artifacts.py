"""Standalone acceptance reads the original files and never repairs them."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_compiler_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_compiler_artifacts", SCRIPT)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class CompilerArtifactCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.model_path = self.path / "model.onnx"
        self.enc_path = self.path / "model.encodings"
        self.report = self.path / "report.json"
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"], name="identity")],
            "acceptance", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8),
                  self.model_path)
        record = dict(bitwidth=8, dtype="INT8", enc_type="PER_TENSOR", is_symmetric="True",
                      scale=.125, n=3, zero_point=0, real_min=-16., real_max=15.875)
        self.document = {"schema_version": 4, "activation_encodings": {
            "identity": {"input": {"0": copy.deepcopy(record)}, "output": {"0": record}}},
            "param_encodings": {}, "excluded_layers": [], "version": "1.0.0"}
        self.write()

    def write(self):
        self.enc_path.write_text(json.dumps(self.document))

    def run_cli(self, *options):
        before = self.model_path.read_bytes(), self.enc_path.read_bytes()
        code = cli.main(["--onnx", str(self.model_path), "--encodings", str(self.enc_path),
                         "--report", str(self.report), *options])
        self.assertEqual(before, (self.model_path.read_bytes(), self.enc_path.read_bytes()))
        return code, json.loads(self.report.read_text())

    def test_valid_pair_and_real_command_exit_zero(self):
        code, report = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(report["schema_version"], 4)
        result = subprocess.run([sys.executable, str(SCRIPT), "--onnx", str(self.model_path),
            "--encodings", str(self.enc_path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_invalid_schema_fields_are_rejected_without_repair(self):
        for label in ("old_schema", "missing_n", "bad_max", "bad_port"):
            with self.subTest(label=label):
                original = copy.deepcopy(self.document)
                if label == "old_schema":
                    self.document["schema_version"] = 3
                elif label == "missing_n":
                    self.document["activation_encodings"]["identity"]["input"]["0"].pop("n")
                elif label == "bad_max":
                    self.document["activation_encodings"]["identity"]["output"]["0"]["real_max"] = 16.
                else:
                    self.document["activation_encodings"]["identity"]["input"]["999"] = (
                        copy.deepcopy(self.document["activation_encodings"]["identity"]["input"]["0"]))
                self.write()
                code, report = self.run_cli()
                self.assertEqual(code, 1)
                self.assertEqual(report["status"], "failed")
                self.document = original

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for text in ('{"schema_version":4,"schema_version":3}', '{"schema_version":NaN}',
                     '{"schema_version":1e999}'):
            with self.subTest(text=text):
                self.enc_path.write_text(text)
                code, report = self.run_cli()
                self.assertEqual(code, 1)
                self.assertIn("JSON", report["error"])

    def test_affine_requires_explicit_mode(self):
        for ports in self.document["activation_encodings"]["identity"].values():
            record = ports["0"]
            record.update(scale=.1, n=3, real_min=-12.8, real_max=12.7)
        self.write()
        self.assertEqual(self.run_cli()[0], 1)
        self.assertEqual(self.run_cli("--allow-affine")[0], 0)

    def test_missing_compute_input_requires_partial_mode(self):
        model = onnx.load(self.model_path)
        model.graph.node[0].op_type = 'Neg'
        onnx.save(model, self.model_path)
        self.document['activation_encodings'].clear()
        self.write()
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn('missing compiler quantization grid', report['error'])
        code, report = self.run_cli('--allow-partial')
        self.assertEqual(code, 0)
        self.assertFalse(report['require_complete'])

    def test_unverifiable_virtual_container_is_rejected(self):
        model = onnx.load(self.model_path)
        model.graph.node[0].name = 'block.identity'
        onnx.save(model, self.model_path)
        self.document['activation_encodings'] = {'block': self.document['activation_encodings']['identity']}
        self.write()
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn('container', report['error'])

    def test_malformed_onnx_is_reported_without_modifying_inputs(self):
        self.model_path.write_bytes(b'not an ONNX model')
        self.assertEqual(self.run_cli()[0], 1)

    def test_parameter_channel_count_matches_onnx(self):
        graph = helper.make_graph([helper.make_node('Conv', ['x', 'w'], ['y'], name='conv')],
            'channels', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 1, 1])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 1, 1])],
            [helper.make_tensor('w', TensorProto.FLOAT, [2, 1, 1, 1], [.125, .25])])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8),
                  self.model_path)
        self.document['activation_encodings'] = {'conv': self.document['activation_encodings']['identity']}
        record = copy.deepcopy(self.document['activation_encodings']['conv']['input']['0'])
        record.update(enc_type='PER_CHANNEL', scale=[.125, .25], n=[3, 2], zero_point=[0, 0],
                      real_min=[-16., -32.], real_max=[15.875, 31.75])
        self.document['param_encodings']['w'] = record
        self.write()
        self.assertEqual(self.run_cli()[0], 0)
        for field, value in dict(scale=.5, n=1, zero_point=0, real_min=-64., real_max=63.5).items():
            record[field].append(value)
        self.write()
        code, report = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn('channel', report['error'])


if __name__ == "__main__":
    unittest.main()
