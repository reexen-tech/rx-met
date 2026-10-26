"""Exercise ONNX GRU through the native operator, including saved artifacts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch

from aimet_onnx.quantsim import load_encodings_to_sim
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow

CONFIG = Path(__file__).resolve().parents[1] / "examples" / "config"
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def graph(directions=2, layout=0, initial_state=True):
    random = np.random.default_rng(42)
    weights = [numpy_helper.from_array(random.normal(0, .1, shape).astype(np.float32), name)
               for name, shape in (("custom_W", (directions, 6, 3)), ("custom_R", (directions, 6, 2)),
                                   ("custom_B", (directions, 12)))]
    xshape = [2, 4, 3] if layout else [4, 2, 3]
    hshape = [2, directions, 2] if layout else [directions, 2, 2]
    yshape = [2, 4, directions, 2] if layout else [4, directions, 2, 2]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, xshape)]
    feed = {"x": random.normal(0, .2, xshape).astype(np.float32)}
    if initial_state:
        inputs.append(helper.make_tensor_value_info("h0", TensorProto.FLOAT, hshape))
        feed["h0"] = random.normal(0, .1, hshape).astype(np.float32)
    node = helper.make_node("GRU", ["x", "custom_W", "custom_R", "custom_B", "", "h0" if initial_state else ""],
                            ["y", "state"], name="user/gru", hidden_size=2, linear_before_reset=1,
                            direction="bidirectional" if directions == 2 else "forward", layout=layout)
    model = helper.make_model(helper.make_graph([node], "native-gru", inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, yshape),
         helper.make_tensor_value_info("state", TensorProto.FLOAT, hshape)], weights),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    return model, feed


@unittest.skipUnless(torch.cuda.is_available(), "native GRU requires CUDA")
class NativeOnnxGruTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-onnx-gru-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.index = 0

    def sim(self, model, feed):
        self.index += 1
        return pipeline.create_onnx_ptq_sim(model, feed,
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"),
            providers=PROVIDERS, work_dir=self.path / str(self.index))

    def calibrate(self, sim, feed):
        apply_onnx_mixed_precision_bitwidth(sim, str(CONFIG / "quick_start_full_quant.json"), verbose=False)
        sim.compute_encodings([feed, {name: value * 1.5 for name, value in feed.items()}])
        apply_power_of_2_workflow(sim, verbose=False)

    def test_cpu_provider_requires_cpu_wheel(self):
        model, feed = graph()
        with self.assertRaisesRegex(RuntimeError, "CPU QuantGRU wheel"):
            pipeline.create_onnx_ptq_sim(
                model, feed,
                config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"),
                providers=("CPUExecutionProvider",), work_dir=self.path / "cpu",
            )

    def test_float_layout_and_gate_mapping_matches_original_ort(self):
        for directions in (1, 2):
            with self.subTest(directions=directions):
                model, feed = graph(directions)
                sim = self.sim(model, feed)
                sim._rx_native_grus["user/gru"].module.use_quantization = False
                expected = ort.InferenceSession(model.SerializeToString(), providers=list(PROVIDERS)).run(None, feed)
                actual = sim.session.run(None, feed)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
                self.assertFalse(sim.qc_quantize_op_dict)

    def test_native_calibration_export_and_restore(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        adapter = sim._rx_native_grus["user/gru"]
        expected = adapter.run(feed["x"], feed["h0"])
        actual = sim.session.run(None, feed)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "export", "gru")
        enc = json.loads(artifacts["encodings"].read_text())
        self.assertEqual(len(enc["param_encodings"]["custom_W"]["scale"]), 12)
        self.assertEqual(len(enc["param_encodings"]["custom_R"]["scale"]), 12)
        self.assertEqual(len(enc["param_encodings"]["custom_B"]["scale"]), 24)
        self.assertEqual(len(enc["activation_encodings"]["user/gru"]["internal_ops"]), 13)
        self.assertEqual(len(enc["activation_encodings"]["user/gru"]["internal_ops_reverse"]), 13)
        saved = onnx.load(str(artifacts["onnx"]))
        self.assertEqual(list(saved.graph.node), list(model.graph.node))
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, actual,
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)

        def legacy_records(value):
            if isinstance(value, dict):
                return {key: legacy_records(child) for key, child in value.items() if key != "n"}
            if isinstance(value, list):
                return [legacy_records(child) for child in value]
            return value

        raw = json.loads(artifacts["aimet_encodings"].read_text())
        raw["native_recurrent_encodings"] = legacy_records(raw["native_recurrent_encodings"])
        for layer in raw["native_recurrent_encodings"]["activation_encodings"].values():
            for kind in ("input", "output"):
                layer[kind] = [layer[kind]["0"]]
        legacy_path = self.path / "legacy.encodings"
        legacy_path.write_text(json.dumps(raw))
        restored = self.sim(model, feed)
        load_encodings_to_sim(restored, str(legacy_path), strict=False)
        for a, b in zip(restored.session.run(None, feed), actual):
            np.testing.assert_array_equal(a, b)

    def test_missing_native_encodings_cannot_restore_generic_grids(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        path = self.path / "raw.encodings"
        sim._export_encodings(str(path))
        payload = json.loads(path.read_text())
        del payload["native_recurrent_encodings"]
        path.write_text(json.dumps(payload))
        restored = self.sim(model, feed)
        with self.assertRaisesRegex(ValueError, "Missing native recurrent"):
            load_encodings_to_sim(restored, str(path), strict=False)

    def test_missing_gate_encoding_is_rejected(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        path = self.path / "raw.encodings"
        sim._export_encodings(str(path))
        payload = json.loads(path.read_text())
        del payload["native_recurrent_encodings"]["activation_encodings"]["user/gru"]["internal_ops"]["update_gate_input"]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "internal encodings"):
            load_encodings_to_sim(self.sim(model, feed), str(path), strict=False)

    def test_native_output_keeps_linear_input_grid_for_bias_alignment(self):
        model, feed = graph()
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "flat_shape"),
            numpy_helper.from_array(np.full((2, 8), .1, dtype=np.float32), "fc_weight"),
            numpy_helper.from_array(np.array([.1, -.1], dtype=np.float32), "fc_bias"),
        ])
        model.graph.node.extend([
            helper.make_node("Reshape", ["state", "flat_shape"], ["flat"], name="flatten"),
            helper.make_node("Gemm", ["flat", "fc_weight", "fc_bias"], ["classification"], name="fc", transB=1),
        ])
        model.graph.output.append(helper.make_tensor_value_info("classification", TensorProto.FLOAT, [1, 2]))
        sim = self.sim(model, feed)
        from aimet_onnx.rx_ptq.po2 import disable_onnx_bias_quantizers
        disable_onnx_bias_quantizers(sim)
        self.calibrate(sim, feed)
        from aimet_onnx.rx_ptq.input_quantization import input_quantizer
        self.assertFalse(sim.qc_quantize_op_dict["state"].enabled)
        self.assertTrue(input_quantizer(sim, sim.connected_graph.get_all_ops()["fc"], 0).is_initialized())
        self.assertTrue(sim.qc_quantize_op_dict["fc_bias"].enabled)

    def test_selected_native_frame_is_calibrated_at_its_consumer(self):
        from aimet_onnx.quantsim import compute_encodings
        from aimet_onnx.rx_ptq.input_quantization import input_quantizer
        from test_percentile_calibration import quantizer

        model, feed = graph()
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array(1, dtype=np.int64), "frame_index"),
            numpy_helper.from_array(np.array([1, 8], dtype=np.int64), "flat_shape"),
            numpy_helper.from_array(np.full((2, 8), .1, dtype=np.float32), "fc_weight"),
        ])
        model.graph.node.extend([
            helper.make_node("Gather", ["y", "frame_index"], ["frame"], name="select", axis=0),
            helper.make_node("Reshape", ["frame", "flat_shape"], ["flat"], name="flatten"),
            helper.make_node("Gemm", ["flat", "fc_weight"], ["scores"], name="fc", transB=1),
            helper.make_node("Neg", ["y"], ["negated"], name="full_sequence"),
        ])
        model.graph.output.extend([
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("negated", TensorProto.FLOAT, [4, 2, 2, 2]),
        ])
        sim = self.sim(model, feed)
        apply_onnx_mixed_precision_bitwidth(sim, str(CONFIG / "quick_start_full_quant.json"), verbose=False)
        ops = sim.connected_graph.get_all_ops()
        selected = input_quantizer(sim, ops["fc"], 0)
        full = input_quantizer(sim, ops["full_sequence"], 0)
        self.assertIsNot(selected, full)
        self.assertFalse(sim.qc_quantize_op_dict["y"].enabled)
        expected = quantizer()
        expected._rx_calibration_device = "cuda:0"
        alias = sim._rx_input_quantizers["fc", 0]
        with compute_encodings(sim):
            for factor in (1, 3):
                values = sim.session.run([alias], {name: array * factor for name, array in feed.items()})[0]
                expected.update_encoding_stats(values)
        self.assertEqual(selected.get_encodings()[0].delta, expected.compute_encodings()[0].delta)
        self.assertNotEqual(selected.get_encodings()[0].delta, full.get_encodings()[0].delta)
        apply_power_of_2_workflow(sim, verbose=False)
        actual = sim.session.run(None, feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "selected", "gru")
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, actual,
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "consumer", "model")
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)

    def test_batch_first_layout_and_optional_initial_state(self):
        model, feed = graph(layout=1, initial_state=False)
        sim = self.sim(model, feed)
        sim._rx_native_grus["user/gru"].module.use_quantization = False
        reference, _ = graph(layout=0, initial_state=False)
        expected = ort.InferenceSession(reference.SerializeToString(), providers=list(PROVIDERS)).run(
            None, {"x": feed["x"].transpose(1, 0, 2).copy()})
        actual = sim.session.run(None, feed)
        np.testing.assert_allclose(actual[0], expected[0].transpose(2, 0, 1, 3), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual[1], expected[1].transpose(1, 0, 2), atol=1e-6, rtol=1e-5)

    def test_explicit_disable_exports_and_restores_without_calibration(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        path = self.path / "disabled.json"
        path.write_text(json.dumps({"GRU_config": {"default_config": {"disable_quantization": True}}}))
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)
        apply_power_of_2_workflow(sim, verbose=False)
        expected = sim.session.run(None, feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "disabled", "gru")
        payload = json.loads(artifacts["encodings"].read_text())
        self.assertEqual(payload["excluded_layers"], ["user/gru"])
        self.assertFalse(payload["activation_encodings"])
        self.assertFalse(payload["param_encodings"])
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)

    def test_explicit_native_bitwidth_and_granularity_are_used(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        path = self.path / "precision.json"
        path.write_text(json.dumps({"GRU_config": {"default_config": {"use_pot2_scale": False},
            "operator_config": {"input": {"bitwidth": 16}, "output": {"bitwidth": 16},
                                "weight_ih": {"quantization_granularity": "PER_GATE"}}}}))
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)
        sim.compute_encodings([feed])
        apply_power_of_2_workflow(sim, verbose=False)
        enc = sim._rx_native_grus["user/gru"].export_encodings()
        self.assertEqual(enc["activation_encodings"]["user/gru"]["input"]["0"]["bitwidth"], 16)
        self.assertEqual(enc["activation_encodings"]["user/gru"]["output"]["0"]["bitwidth"], 16)
        self.assertEqual(sim._rx_native_grus["user/gru"].module._bitwidth_config.W_granularity_, 1)
        scales = enc["param_encodings"]["custom_W"]["scale"]
        # Native public encodings expand gate grids to the packed channel view.
        self.assertEqual(len(scales), 12)
        self.assertTrue(all(scales[index] == scales[index + 1] for index in range(0, 12, 2)))
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "precision", "gru")
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)

    def test_failure_restores_native_calibration_flags(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        def fail(session):
            session.run(None, feed)
            raise RuntimeError("callback failure")
        with self.assertRaisesRegex(RuntimeError, "callback failure"):
            sim.compute_encodings(fail)
        self.assertFalse(sim._rx_native_grus["user/gru"].module.calibrating)

    def test_unsupported_equation_is_rejected(self):
        model, feed = graph()
        for attr in model.graph.node[0].attribute:
            if attr.name == "linear_before_reset":
                attr.i = 0
        with self.assertRaisesRegex(NotImplementedError, "linear_before_reset=1"):
            self.sim(model, feed)


if __name__ == "__main__":
    unittest.main()
