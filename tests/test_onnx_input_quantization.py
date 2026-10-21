"""Consumer-local native/Concat grids, calibration order, and clean export."""

import json
import unittest

import numpy as np
import onnx
from onnx import helper

from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.input_quantization import input_quantizer
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
import test_onnx_operand_precision as operand
from test_percentile_calibration import quantizer


class OnnxInputQuantizationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = operand.OnnxOperandPrecisionTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
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
            config_file=str(self.fixture.fixture.config), quant_scheme="percentile")
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
            config_file=str(self.fixture.fixture.config), quant_scheme="percentile")

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
