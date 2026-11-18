"""JSON2 disable rules take precedence over automatic model IO quantization."""

import copy
import json
import unittest

import numpy as np
import torch
from onnx import helper

from aimet_common.compiler_artifacts import validate_compiler_artifacts
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.input_quantization import input_quantizer
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
from tests.support import onnx as fixtures


class SigmoidGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Sigmoid()

    def forward(self, x):
        return self.gate(x)


class IoConfigurationPrecedenceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxGraphFixture(self)
        self.index = 0

    def config(self, payload):
        self.index += 1
        path = self.fixture.path / f"mixed-{self.index}.json"
        path.write_text(json.dumps(payload))
        return path

    def gate(self, sibling=None):
        nodes = [helper.make_node("Sigmoid", ["x"], ["gate_out"], name="gate")]
        outputs = {"gate_out": [1, 4]}
        constants = {}
        if sibling == "Relu":
            nodes.append(helper.make_node("Relu", ["x"], ["other_out"], name="other"))
            outputs["other_out"] = [1, 4]
        elif sibling == "Concat":
            nodes.append(helper.make_node("Concat", ["x", "x"], ["other_out"], name="other", axis=1))
            outputs["other_out"] = [1, 8]
        elif sibling == "MatMul":
            constants["w"] = np.eye(4, dtype=np.float32)
            nodes.append(helper.make_node("MatMul", ["x", "w"], ["other_out"], name="other"))
            outputs["other_out"] = [1, 4]
        return self.fixture.model(nodes, {"x": [1, 4]}, outputs, constants)

    def apply(self, sim, path):
        return apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)

    def assert_gate_disabled(self, sim):
        op = sim.connected_graph.get_all_ops()["gate"]
        quantizers = [input_quantizer(sim, op, 0), sim.qc_quantize_op_dict["gate_out"]]
        for quantizer in quantizers:
            self.assertFalse(quantizer.enabled)
            self.assertIn(id(quantizer), sim._rx_explicit_disabled_quantizers)

    def test_disable_scope_matches_torch_and_survives_repeated_io_forcing(self):
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        from aimet_torch.v2.quantsim import QuantizationSimModel

        cases = (
            {"layer_type_config": {"QuantizedSigmoid": {"disable_quantization": True}}},
            {"layer_name_config": {"gate": {"disable_quantization": True}}},
            {"layer_name_config": {"g*": {"disable_quantization": True}}},
            {"default_config": {"disable_quantization": True}},
        )
        for settings in cases:
            with self.subTest(settings=settings):
                path = self.config(settings)
                sim, model, feed = self.gate()
                reference = QuantizationSimModel(prepare_model(SigmoidGate().eval()),
                    torch.from_numpy(feed["x"]), config_file=str(self.fixture.config), quant_scheme="tf")
                apply_mixed_precision_bitwidth(reference.model, str(path), verbose=False)
                self.assertTrue(all(q is None for q in reference.model.gate.input_quantizers))
                self.assertTrue(all(q is None for q in reference.model.gate.output_quantizers))
                for _ in range(2):
                    self.apply(sim, path)
                    pipeline.force_model_io_quantizers(sim, model)
                    self.assert_gate_disabled(sim)

    def test_disabled_sigmoid_does_not_disable_enabled_shared_consumer(self):
        aliases = {"Relu": "QuantizedReLU", "Concat": "QuantizedConcat", "MatMul": "QuantizedMatMul"}
        for kind, alias in aliases.items():
            with self.subTest(kind=kind):
                sim, model, feed = self.gate(kind)
                path = self.config({"layer_type_config": {
                    "QuantizedSigmoid": {"disable_quantization": True},
                    alias: {"input_bitwidth": 8, "output_bitwidth": 16},
                }})
                self.apply(sim, path)
                pipeline.force_model_io_quantizers(sim, model)
                self.assert_gate_disabled(sim)
                other = sim.connected_graph.get_all_ops()["other"]
                enabled = input_quantizer(sim, other, 0)
                self.assertTrue(enabled.enabled)
                self.assertNotIn(id(enabled), sim._rx_explicit_disabled_quantizers)
                self.assertTrue(sim.qc_quantize_op_dict["other_out"].enabled)
                self.assertFalse(sim.qc_quantize_op_dict["x"].enabled)
                sim.compute_encodings([feed])
                apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
                expected = pipeline.run_onnx_feed(sim, feed)
                artifacts = pipeline.export_onnx_json(sim, model,
                    self.fixture.path / f"shared-export-{kind}", "model")
                self.assertEqual(validate_compiler_artifacts(artifacts["onnx"], artifacts["encodings"],
                    require_complete=True)["status"], "passed")
                report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
                    config_file=str(self.fixture.config), quant_scheme="tf")
                self.assertTrue(all(item["max_abs"] == 0 for item in report))

    def test_disabling_local_concat_or_matmul_keeps_sigmoid_enabled(self):
        for kind, alias in (("Concat", "QuantizedConcat"), ("MatMul", "QuantizedMatMul")):
            with self.subTest(kind=kind):
                sim, model, _ = self.gate(kind)
                self.apply(sim, self.config({"layer_type_config": {
                    "QuantizedSigmoid": {"input_bitwidth": 8, "output_bitwidth": 16},
                    alias: {"disable_quantization": True},
                }}))
                pipeline.force_model_io_quantizers(sim, model)
                other = sim.connected_graph.get_all_ops()["other"]
                for q in (input_quantizer(sim, other, 0), sim.qc_quantize_op_dict["other_out"]):
                    self.assertFalse(q.enabled)
                    self.assertIn(id(q), sim._rx_explicit_disabled_quantizers)
                gate = sim.connected_graph.get_all_ops()["gate"]
                self.assertTrue(input_quantizer(sim, gate, 0).enabled)
                self.assertTrue(sim.qc_quantize_op_dict["gate_out"].enabled)
                self.assertFalse(sim.qc_quantize_op_dict["x"].enabled)

    def test_exact_type_and_pattern_enabling_exceptions_beat_lower_priority_disable(self):
        cases = (
            {"layer_type_config": {"QuantizedSigmoid": {"disable_quantization": True}},
             "layer_name_config": {"gate": {"disable_quantization": False}}},
            {"layer_type_config": {"QuantizedSigmoid": {"disable_quantization": False}},
             "layer_name_config": {"g*": {"disable_quantization": True}}},
            {"default_config": {"disable_quantization": True},
             "layer_name_config": {"g*": {"disable_quantization": False}}},
        )
        for settings in cases:
            with self.subTest(settings=settings):
                sim, model, _ = self.gate()
                self.apply(sim, self.config(settings))
                pipeline.force_model_io_quantizers(sim, model)
                op = sim.connected_graph.get_all_ops()["gate"]
                for q in (input_quantizer(sim, op, 0), sim.qc_quantize_op_dict["gate_out"]):
                    self.assertTrue(q.enabled)
                    self.assertNotIn(id(q), sim._rx_explicit_disabled_quantizers)

    def test_explicit_activation_widths_restore_disabled_io_and_clear_markers(self):
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        from aimet_torch.v2.quantsim import QuantizationSimModel

        sim, model, feed = self.gate()
        reference = QuantizationSimModel(prepare_model(SigmoidGate().eval()),
            torch.from_numpy(feed["x"]), config_file=str(self.fixture.config), quant_scheme="tf")
        disabled = self.config({"layer_type_config": {
            "QuantizedSigmoid": {"disable_quantization": True}}})
        self.apply(sim, disabled)
        apply_mixed_precision_bitwidth(reference.model, str(disabled), verbose=False)
        pipeline.force_model_io_quantizers(sim, model)
        self.assert_gate_disabled(sim)
        enabled = self.config({"layer_type_config": {
            "QuantizedSigmoid": {"input_bitwidth": 8, "output_bitwidth": 16}}})
        self.apply(sim, enabled)
        apply_mixed_precision_bitwidth(reference.model, str(enabled), verbose=False)
        op = sim.connected_graph.get_all_ops()["gate"]
        for quantizer, expected in ((input_quantizer(sim, op, 0), 8),
                                   (sim.qc_quantize_op_dict["gate_out"], 16)):
            self.assertTrue(quantizer.enabled)
            self.assertEqual(quantizer.bitwidth, expected)
            self.assertNotIn(id(quantizer), sim._rx_explicit_disabled_quantizers)
        self.assertEqual(reference.model.gate.input_quantizers[0].bitwidth, 8)
        self.assertEqual(reference.model.gate.output_quantizers[0].bitwidth, 16)
        pipeline.force_model_io_quantizers(sim, model)
        self.assertTrue(input_quantizer(sim, op, 0).enabled)
        self.assertTrue(sim.qc_quantize_op_dict["gate_out"].enabled)

    def test_disabled_layer_exports_complete_artifacts_and_reloads(self):
        sim, model, feed = self.gate()
        self.apply(sim, self.config({"layer_type_config": {
            "QuantizedSigmoid": {"disable_quantization": True}}}))
        pipeline.force_model_io_quantizers(sim, model)
        self.assert_gate_disabled(sim)
        sim.compute_encodings([feed])
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        expected = pipeline.run_onnx_feed(sim, feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.fixture.path / "disabled-export", "model")
        document = json.loads(artifacts["encodings"].read_text())
        self.assertIn("gate", document["excluded_layers"])
        self.assertEqual(document["activation_encodings"], {})
        validation = validate_compiler_artifacts(artifacts["onnx"], artifacts["encodings"],
            require_po2=True, require_complete=True)
        self.assertEqual(validation["status"], "passed")
        report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
            config_file=str(self.fixture.config), quant_scheme="tf")
        self.assertTrue(all(item["max_abs"] == 0 for item in report))

    def test_invalid_saved_overrides_fail_before_quantizer_mutation(self):
        from aimet_onnx.quantsim import load_encodings_to_sim

        sim, model, feed = self.gate()
        sim.compute_encodings([feed])
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        active_path = self.fixture.path / "active.encodings"
        sim._export_encodings(str(active_path))
        active = json.loads(active_path.read_text())
        output_record = next(entry for entry in active["activation_encodings"] if entry["name"] == "gate_out")
        sim, model, feed = self.gate()
        self.apply(sim, self.config({"layer_type_config": {
            "QuantizedSigmoid": {"disable_quantization": True}}}))
        pipeline.force_model_io_quantizers(sim, model)
        sim.compute_encodings([feed])
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        artifacts = pipeline.export_onnx_json(sim, model, self.fixture.path / "tamper-source", "model")
        saved = json.loads(artifacts["aimet_encodings"].read_text())
        self.assertEqual(saved["rx_quantization_overrides"]["schema_version"], 1)

        def state(target):
            quantizers = {name: (q.enabled, q.bitwidth, q.signed, q.use_symmetric_encodings, q.is_initialized())
                          for name, q in target.qc_quantize_op_dict.items()}
            return (quantizers, set(getattr(target, "_rx_explicit_disabled_quantizers", ())),
                    tuple(getattr(target, "_excluded_layer_names", ()) or ()))

        for case in ("unknown_quantizer", "unknown_layer", "encoded_disabled_output"):
            with self.subTest(case=case):
                changed = copy.deepcopy(saved)
                metadata = changed["rx_quantization_overrides"]
                if case == "unknown_quantizer":
                    metadata["disabled_quantizers"].append("absent_quantizer")
                elif case == "unknown_layer":
                    metadata["excluded_layers"].append("absent_layer")
                else:
                    changed["activation_encodings"].append(copy.deepcopy(output_record))
                path = self.fixture.path / f"tampered-{case}.encodings"
                path.write_text(json.dumps(changed))
                target, _, _ = self.gate()
                before = state(target)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_encodings_to_sim(target, str(path), strict=False,
                        allow_overwrite=True, disable_missing_quantizers=True)
                self.assertEqual(state(target), before)

    def test_active_legacy_raw_clears_stale_disabled_markers_and_exclusions(self):
        from aimet_onnx.quantsim import load_encodings_to_sim
        from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter

        source, model, feed = self.gate()
        source.compute_encodings([feed])
        apply_power_of_2_workflow(source, align_bias_scale=False, verbose=False)
        path = self.fixture.path / "active-legacy.encodings"
        source._export_encodings(str(path))
        raw = json.loads(path.read_text())
        raw.pop("rx_quantization_overrides", None)
        path.write_text(json.dumps(raw))
        expected, _ = CompilerEncodingConverter(source, model).convert()

        target, _, _ = self.gate()
        self.apply(target, self.config({"layer_type_config": {
            "QuantizedSigmoid": {"disable_quantization": True}}}))
        pipeline.force_model_io_quantizers(target, model)
        self.assert_gate_disabled(target)
        self.assertIn("gate", target._excluded_layer_names)
        load_encodings_to_sim(target, str(path), strict=False,
            allow_overwrite=True, disable_missing_quantizers=True)
        op = target.connected_graph.get_all_ops()["gate"]
        for quantizer in (input_quantizer(target, op, 0), target.qc_quantize_op_dict["gate_out"]):
            self.assertTrue(quantizer.enabled)
            self.assertNotIn(id(quantizer), target._rx_explicit_disabled_quantizers)
        self.assertNotIn("gate", target._excluded_layer_names)
        actual, _ = CompilerEncodingConverter(target, model).convert()
        self.assertEqual(actual, expected)
        for left, right in zip(pipeline.run_onnx_feed(source, feed), pipeline.run_onnx_feed(target, feed)):
            np.testing.assert_array_equal(left, right)

    def test_shared_alias_identity_conflict_is_rejected_before_loading(self):
        from aimet_onnx.quantsim import load_encodings_to_sim

        source, _, feed = self.gate("Concat")
        source.compute_encodings([feed])
        apply_power_of_2_workflow(source, align_bias_scale=False, verbose=False)
        first = source._rx_input_quantizers["other", 0]
        second = source._rx_input_quantizers["other", 1]
        self.assertIs(source.qc_quantize_op_dict[first], source.qc_quantize_op_dict[second])
        path = self.fixture.path / "shared-alias-conflict.encodings"
        source._export_encodings(str(path))
        raw = json.loads(path.read_text())
        raw["activation_encodings"] = [entry for entry in raw["activation_encodings"] if entry["name"] != first]
        self.assertIn(second, {entry["name"] for entry in raw["activation_encodings"]})
        raw["rx_quantization_overrides"] = {
            "schema_version": 1, "disabled_quantizers": [first], "excluded_layers": []}
        path.write_text(json.dumps(raw))
        target, _, _ = self.gate("Concat")

        def state():
            return ({name: (q.enabled, q.bitwidth, q.signed, q.use_symmetric_encodings, q.is_initialized())
                     for name, q in target.qc_quantize_op_dict.items()},
                    set(getattr(target, "_rx_explicit_disabled_quantizers", ())),
                    tuple(getattr(target, "_excluded_layer_names", ()) or ()))

        before = state()
        with self.assertRaises((ValueError, RuntimeError)):
            load_encodings_to_sim(target, str(path), strict=False,
                allow_overwrite=True, disable_missing_quantizers=True)
        self.assertEqual(state(), before)

    @unittest.skipUnless(torch.cuda.is_available(), "native GRU requires CUDA")
    def test_disabling_local_sigmoid_preserves_native_input_domain_and_reload(self):
        from tests.support.recurrent import gru_graph

        model, feed = gru_graph(directions=1, initial_state=False)
        model.graph.node.append(helper.make_node("Sigmoid", ["x"], ["gate_out"], name="gate"))
        model.graph.output.append(helper.make_tensor_value_info("gate_out", 1, feed["x"].shape))
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
        sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.fixture.config),
            quant_scheme="tf", providers=providers, work_dir=self.fixture.path / "native")
        self.apply(sim, self.config({"layer_type_config": {
            "QuantizedSigmoid": {"disable_quantization": True}}}))
        pipeline.force_model_io_quantizers(sim, model)
        self.assert_gate_disabled(sim)
        self.assertFalse(sim.qc_quantize_op_dict["x"].enabled)
        adapter = sim._rx_native_recurrent["user/gru"]
        self.assertTrue(adapter.module.use_quantization)
        sim.compute_encodings([feed, {"x": feed["x"] * 1.5}])
        apply_power_of_2_workflow(sim, verbose=False)
        expected = pipeline.run_onnx_feed(sim, feed, providers)
        artifacts = pipeline.export_onnx_json(sim, model, self.fixture.path / "native-export", "model")
        self.assertEqual(validate_compiler_artifacts(artifacts["onnx"], artifacts["encodings"],
            require_complete=True)["status"], "passed")
        report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
            config_file=str(self.fixture.config), quant_scheme="tf", providers=providers)
        self.assertTrue(all(item["max_abs"] == 0 for item in report))


if __name__ == "__main__":
    unittest.main()
