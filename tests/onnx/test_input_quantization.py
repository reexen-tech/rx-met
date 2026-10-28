"""ONNX operand precision, consumer-local shared grids and conflict handling."""

import json
import unittest
import numpy as np
from onnx import helper
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
from aimet_common.defs import QuantScheme
from aimet_onnx.quantsim import compute_encodings
import onnx
from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.rx_ptq.input_quantization import input_quantizer

from tests.support import onnx as fixtures
from tests.support.calibration import quantizer


class OnnxOperandPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxGraphFixture(self)
        self.path = self.fixture.path

    def test_input_precision_reaches_quantizer_before_transpose(self):
        sim, model, feeds = self.fixture.moved_sim()
        self.assertNotIn("moved", sim.qc_quantize_op_dict)
        self.fixture.apply_config(sim, {"layer_name_config": {"window": {
            "input_bitwidth": 16, "output_bitwidth": 16, "input_symmetric": True}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 16)
        sim.compute_encodings([feeds])
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        artifact = pipeline.export_onnx_json(sim, model, self.path / "export", "model")
        payload = json.loads(artifact["encodings"].read_text())
        self.assertEqual(payload["activation_encodings"]["window"]["input"]["0"]["bitwidth"], 16)
        self.assertTrue(np.isfinite(sim.session.run(None, feeds)[0]).all())

    def test_shared_precision_conflict_through_transpose_is_atomic(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                sim, _, _ = self.fixture.model([
                    helper.make_node("Relu", ["x"], ["before"], name="producer"),
                    helper.make_node("Transpose", ["before"], ["moved"], name="transpose", perm=[1, 0]),
                    helper.make_node("Relu", ["moved"], ["y"], name="consumer"),
                ], {"x": [2, 2]}, {"y": [2, 2]})
                if reverse:
                    ops = dict(reversed(list(sim.connected_graph.get_all_ops().items())))
                    sim.connected_graph.get_all_ops = lambda: ops
                original = {name: q.bitwidth for name, q in sim.qc_quantize_op_dict.items()}
                with self.assertRaisesRegex(ValueError, "Conflicting shared quantizer bitwidth"):
                    self.fixture.apply_config(sim, {"layer_name_config": {
                        "producer": {"output_bitwidth": 16}, "consumer": {"input_bitwidth": 8}}})
                self.assertEqual(original, {name: q.bitwidth for name, q in sim.qc_quantize_op_dict.items()})

    def test_matmul_constant_uses_input_precision_and_per_tensor_grid(self):
        sim, model = self.fixture.sim(bias=False)
        self.fixture.apply_config(sim, {"layer_type_config": {"QuantizedMatMul": {
            "input_bitwidth": 16, "input_symmetric": True, "output_bitwidth": 8}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 16)
        weight = sim.qc_quantize_op_dict["w"]
        self.assertEqual(weight.bitwidth, 16)
        self.assertFalse(weight.quant_info.usePerChannelMode)
        self.fixture.calibrate(sim, bias=False)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        artifact = pipeline.export_onnx_json(sim, model, self.path / "matmul", "model")
        payload = json.loads(artifact["encodings"].read_text())
        self.assertEqual(payload["param_encodings"]["w"]["bitwidth"], 16)
        self.assertEqual(len(payload["param_encodings"]["w"]["scale"]), 1)

    def test_matmul_input_bitwidths_keep_constant_port_indices(self):
        sim, _ = self.fixture.sim(bias=False)
        self.fixture.apply_config(sim, {"layer_type_config": {"QuantizedMatMul": {
            "input_bitwidths": [8, 16], "input_symmetric": False}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 8)
        self.assertEqual(sim.qc_quantize_op_dict["w"].bitwidth, 16)
        self.assertFalse(sim.qc_quantize_op_dict["w"].use_symmetric_encodings)

    def test_constant_matmul_operand_is_passthrough_during_calibration(self):
        feed = {"x": np.array([[1., 0., 0., 0.]], dtype=np.float32)}
        weight = np.array([[.01337, -.02], [.00456, .00789],
                           [.01234, .01029], [-.00312, .0199]], dtype=np.float32)
        for kind in ("QuantizedMatMul", "QuantizedLinear"):
            with self.subTest(kind=kind):
                sim, _, _ = self.fixture.model([helper.make_node("MatMul", ["x", "w"], ["y"], name="fc")],
                    {"x": [1, 4]}, {"y": [1, 2]}, {"w": weight})
                for q in sim.qc_quantize_op_dict.values():
                    q.set_quant_scheme(QuantScheme.post_training_percentile)
                    q.set_percentile_value(99.99)
                settings = {"input_bitwidth": 16, "output_bitwidth": 8}
                if kind == "QuantizedLinear":
                    settings.update(weight_bitwidth=16, per_channel_quantization=False)
                self.fixture.apply_config(sim, {"layer_type_config": {kind: settings}})
                with compute_encodings(sim):
                    actual = sim.session.run(None, feed)[0]
                if kind == "QuantizedMatMul":
                    np.testing.assert_array_equal(actual, feed["x"] @ weight)
                else:
                    self.assertFalse(np.array_equal(actual, feed["x"] @ weight))
                    quantized = sim.qc_quantize_op_dict["w"].quantize_dequantize(weight)
                    np.testing.assert_array_equal(actual, feed["x"] @ quantized)
                self.assertTrue(sim.qc_quantize_op_dict["w"].is_initialized())

    def test_shared_constant_cannot_mix_parameter_and_operand_calibration(self):
        sim, _, _ = self.fixture.model([
            helper.make_node("MatMul", ["x1", "w"], ["y1"], name="left"),
            helper.make_node("MatMul", ["x2", "w"], ["y2"], name="right"),
        ], {"x1": [1, 4], "x2": [1, 4]}, {"y1": [1, 2], "y2": [1, 2]},
            {"w": np.ones((4, 2), dtype=np.float32)})
        with self.assertRaisesRegex(ValueError, "Conflicting shared quantizer calibration_passthrough"):
            self.fixture.apply_config(sim, {"layer_type_config": {"QuantizedMatMul": {"input_bitwidth": 8}},
                             "layer_name_config": {"right": {"weight_bitwidth": 8,
                                                              "input_bitwidth": 8}}})

    def test_named_matmul_input_settings_override_defaults(self):
        sim, _ = self.fixture.sim(bias=False)
        self.fixture.apply_config(sim, {"default_bitwidth": {"weight": 8, "output": 8},
                         "layer_type_config": {"QuantizedMatMul": {"input_bitwidth": 8}},
                         "layer_name_config": {"fc": {"input_bitwidth": 16}}})
        self.assertEqual(sim.qc_quantize_op_dict["w"].bitwidth, 16)
        self.assertFalse(sim.qc_quantize_op_dict["w"].quant_info.usePerChannelMode)

    def test_explicit_matmul_weight_policy_takes_precedence(self):
        sim, _ = self.fixture.sim(bias=False)
        self.fixture.apply_config(sim, {"layer_type_config": {"QuantizedMatMul": {
            "input_bitwidth": 16, "input_symmetric": True}},
            "layer_name_config": {"fc": {"input_bitwidth": 16, "input_symmetric": True,
                                          "weight_bitwidth": 4, "weight_symmetric": False,
                                          "per_channel_quantization": True}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 16)
        weight = sim.qc_quantize_op_dict["w"]
        self.assertEqual(weight.bitwidth, 4)
        self.assertFalse(weight.use_symmetric_encodings)
        self.assertTrue(weight.quant_info.usePerChannelMode)

    def test_linear_weight_policy_is_separate_from_inputs(self):
        sim, _ = self.fixture.sim(bias=False)
        self.fixture.apply_config(sim, {"layer_type_config": {"QuantizedLinear": {
            "input_bitwidth": 16, "input_symmetric": False, "weight_bitwidth": 8}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 16)
        weight = sim.qc_quantize_op_dict["w"]
        self.assertEqual(weight.bitwidth, 8)
        self.assertTrue(weight.use_symmetric_encodings)
        self.assertTrue(weight.quant_info.usePerChannelMode)

    def test_shared_constant_conflict_is_rejected_before_writes(self):
        sim, _, _ = self.fixture.model([
            helper.make_node("MatMul", ["x1", "w"], ["y1"], name="left"),
            helper.make_node("MatMul", ["x2", "w"], ["y2"], name="right"),
        ], {"x1": [1, 4], "x2": [1, 4]}, {"y1": [1, 2], "y2": [1, 2]},
            {"w": np.ones((4, 2), dtype=np.float32)})
        original = {name: (q.bitwidth, q.quant_info.usePerChannelMode)
                    for name, q in sim.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(ValueError, "Conflicting shared quantizer bitwidth"):
            self.fixture.apply_config(sim, {"layer_type_config": {"QuantizedMatMul": {"input_bitwidth": 8}},
                             "layer_name_config": {"left": {"input_bitwidths": [8, 16]},
                                                   "right": {"input_bitwidths": [8, 8]}}})
        self.assertEqual(original, {name: (q.bitwidth, q.quant_info.usePerChannelMode)
                                    for name, q in sim.qc_quantize_op_dict.items()})


class OnnxInputQuantizationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxGraphFixture(self)
        self.path = self.fixture.path

    def concat(self):
        sim, model, _ = self.fixture.model([
            helper.make_node("Relu", ["x"], ["left"], name="left"),
            helper.make_node("Neg", ["x"], ["right"], name="right"),
            helper.make_node("Concat", ["left", "right"], ["y"], name="join", axis=1),
        ], {"x": [1, 4]}, {"y": [1, 8]}, quant_scheme="percentile")
        sim.set_percentile_value(99.99)
        self.fixture.apply(sim, {"layer_name_config": {"join": {
            "input_bitwidth": 4, "input_symmetric": True, "output_bitwidth": 8}}})
        return sim, model

    def test_concat_has_independent_shared_input_grid(self):
        sim, model = self.concat()
        op = sim.connected_graph.get_all_ops()["join"]
        shared = input_quantizer(sim, op, 0)
        self.assertIs(shared, input_quantizer(sim, op, 1))
        self.assertIsNot(shared, sim.qc_quantize_op_dict["left"])
        self.assertIsNot(shared, sim.qc_quantize_op_dict["right"])
        self.assertEqual(sim.qc_quantize_op_dict["left"].bitwidth, 8)
        expected = quantizer()
        expected.set_bitwidth(4)
        feeds = [{"x": (np.array([[-2., -.1, .03, .7]], dtype=np.float32) * factor)}
                 for factor in (1, 10, .5)]
        for feed in feeds:
            expected.update_encoding_stats(np.maximum(feed["x"], 0))
            expected.update_encoding_stats(-feed["x"])
        sim.compute_encodings(feeds)
        self.assertEqual(shared.get_encodings()[0].delta, expected.compute_encodings()[0].delta)
        self.assertEqual(shared._percentile_calibrator.observer.stats[0].histogram.sum().item(), 24)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "export", "model")
        payload = json.loads(artifacts["encodings"].read_text())
        self.assertEqual(payload["activation_encodings"]["join"]["input"]["0"],
                         payload["activation_encodings"]["join"]["input"]["1"])
        self.assertEqual(payload["activation_encodings"]["join"]["input"]["0"]["bitwidth"], 4)
        self.assertEqual(payload["activation_encodings"]["left"]["output"]["0"]["bitwidth"], 8)
        self.assertEqual(list(onnx.load(artifacts["onnx"]).graph.node), list(model.graph.node))
        output = sim.session.run(None, feeds[0])
        report = pipeline.verify_onnx_ptq_artifacts(artifacts, feeds[0], output,
            config_file=str(self.fixture.config), quant_scheme="percentile")
        self.assertEqual(report[0]["max_abs"], 0)

    def test_conflicting_shared_concat_inputs_fail_before_writes(self):
        sim, _ = self.concat()
        op = sim.connected_graph.get_all_ops()["join"]
        shared = input_quantizer(sim, op, 0)
        with self.assertRaisesRegex(ValueError, "Conflicting shared quantizer bitwidth"):
            self.fixture.apply(sim, {"layer_name_config": {"join": {
                "input_bitwidths": [8, 16], "output_bitwidth": 16}}})
        self.assertEqual(shared.bitwidth, 4)
        self.assertEqual(sim.qc_quantize_op_dict["y"].bitwidth, 8)

    def test_disabled_concat_preserves_producer_grids(self):
        sim, _ = self.concat()
        self.fixture.apply(sim, {"layer_name_config": {"join": {"disable_quantization": True}}})
        self.assertTrue(sim.qc_quantize_op_dict["left"].enabled)
        self.assertTrue(sim.qc_quantize_op_dict["right"].enabled)
        op = sim.connected_graph.get_all_ops()["join"]
        self.assertFalse(input_quantizer(sim, op, 0).enabled)

    def test_concat_graph_inputs_have_no_hidden_upstream_quantization(self):
        sim, model, _ = self.fixture.model([
            helper.make_node("Concat", ["x", "x", "z"], ["y"], name="join", axis=1),
            helper.make_node("Relu", ["x"], ["other"], name="branch"),
        ], {"x": [1, 4], "z": [1, 4]}, {"y": [1, 12], "other": [1, 4]}, quant_scheme="percentile")
        self.fixture.apply(sim, {"layer_name_config": {
            "join": {"input_bitwidth": 4, "input_symmetric": True},
            "branch": {"input_bitwidth": 16}}})
        for name in ("x", "z"):
            self.assertFalse(sim.qc_quantize_op_dict[name].enabled)
        ops = sim.connected_graph.get_all_ops()
        q = input_quantizer(sim, ops["join"], 0)
        self.assertIs(q, input_quantizer(sim, ops["join"], 1))
        self.assertIs(q, input_quantizer(sim, ops["join"], 2))
        self.assertEqual(input_quantizer(sim, ops["branch"], 0).bitwidth, 16)
        sim.set_percentile_value(99.99)
        feed = {"x": np.array([[-1.234, -.01, .01, 1.23]], dtype=np.float32),
                "z": np.array([[-3.14, .2, .5, 2.7]], dtype=np.float32)}
        expected = quantizer()
        expected.set_bitwidth(4)
        for name in ("x", "x", "z"):
            expected.update_encoding_stats(feed[name])
        sim.compute_encodings([feed])
        self.assertEqual(q.get_encodings()[0].delta, expected.compute_encodings()[0].delta)
        self.assertEqual(q._percentile_calibrator.observer.stats[0].histogram.sum().item(), 12)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "direct", "model")
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
            config_file=str(self.fixture.config), quant_scheme="percentile")

    def test_conflicting_saved_shared_grids_are_rejected_before_loading(self):
        sim, _ = self.concat()
        sim.compute_encodings([{"x": np.array([[-2., -.1, .03, .7]], dtype=np.float32)}])
        saved = self.path / "raw.encodings"
        sim._export_encodings(str(saved))
        payload = json.loads(saved.read_text())
        alias = sim._rx_input_quantizers["join", 1]
        entry = next(value for value in payload["activation_encodings"] if value["name"] == alias)
        entry["scale"][0] *= 2
        saved.write_text(json.dumps(payload))
        restored, _ = self.concat()
        with self.assertRaisesRegex(ValueError, "Conflicting encodings for shared quantizer"):
            load_encodings_to_sim(restored, str(saved), strict=False, allow_overwrite=False)
        op = restored.connected_graph.get_all_ops()["join"]
        self.assertFalse(input_quantizer(restored, op, 0).is_initialized())


if __name__ == "__main__":
    unittest.main()
