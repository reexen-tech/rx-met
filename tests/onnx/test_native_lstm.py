"""ONNX LSTM PTQ delegates execution and grids to native QuantLSTM."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import torch
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow

from tests.support.paths import REPO_ROOT
from tests.support.recurrent import lstm_graph as graph


CONFIG = REPO_ROOT / "examples" / "config"
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


@unittest.skipUnless(torch.cuda.is_available(), "native LSTM requires CUDA")
class NativeOnnxLstmTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-onnx-lstm-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.index = 0

    def sim(self, model, feed):
        self.index += 1
        return pipeline.create_onnx_ptq_sim(model, feed,
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"),
            providers=PROVIDERS, work_dir=self.path / str(self.index))

    def calibrate(self, sim, feed, config=None):
        apply_onnx_mixed_precision_bitwidth(sim, str(config or CONFIG / "quick_start_full_quant.json"),
                                           verbose=False)
        sim.compute_encodings([feed, {name: value * 1.5 for name, value in feed.items()}])
        apply_power_of_2_workflow(sim, verbose=False)

    def test_float_gate_layout_and_independent_optional_states(self):
        for directions, layout, states, bias in ((1, 0, (True, True), True),
                (2, 0, (True, True), True), (2, 1, (False, False), False),
                (2, 0, (True, False), True), (1, 0, (False, True), True)):
            with self.subTest(directions=directions, layout=layout, states=states, bias=bias):
                model, feed = graph(directions, layout, states, bias)
                sim = self.sim(model, feed)
                adapter = sim._rx_native_recurrent["user/lstm"]
                adapter.module.use_quantization = False
                # ORT does not implement layout=1. Convert only the reference layout.
                reference = onnx.ModelProto.FromString(model.SerializeToString())
                for attr in reference.graph.node[0].attribute:
                    if attr.name == "layout":
                        attr.i = 0
                del reference.graph.value_info[:]
                for item in list(reference.graph.input) + list(reference.graph.output):
                    item.type.tensor_type.ClearField("shape")
                ref_feed = {name: value.transpose(1, 0, 2).copy() if layout else value
                            for name, value in feed.items()}
                expected = ort.InferenceSession(reference.SerializeToString(),
                    providers=pipeline.ptq_execution_providers(PROVIDERS)).run(None, ref_feed)
                if layout:
                    expected = [expected[0].transpose(2, 0, 1, 3), expected[1].transpose(1, 0, 2),
                                expected[2].transpose(1, 0, 2)]
                for actual, wanted in zip(sim.session.run(None, feed), expected):
                    np.testing.assert_allclose(actual, wanted, atol=1e-6, rtol=1e-5)
                self.assertFalse(sim.qc_quantize_op_dict)

    def test_calibration_po2_export_restore_and_session_lifetime(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        adapter = sim._rx_native_recurrent["user/lstm"]
        expected = adapter.run(feed["x"], feed["h0"], feed["c0"])
        actual = sim.session.run(None, feed)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "export", "lstm")
        enc = json.loads(artifacts["encodings"].read_text())
        self.assertEqual(len(enc["param_encodings"]["custom_W"]["scale"]), 16)
        self.assertEqual(len(enc["param_encodings"]["custom_B"]["scale"]), 32)
        entry = enc["activation_encodings"]["user/lstm"]
        self.assertEqual(len(entry["internal_ops"]), 14)
        self.assertEqual(len(entry["internal_ops_reverse"]), 14)
        clean = onnx.load(artifacts["onnx"])
        self.assertEqual(clean.graph.node[0], model.graph.node[0])
        self.assertEqual(list(clean.graph.initializer), list(model.graph.initializer))
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, actual,
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)
        session = sim.session
        del sim, adapter
        for a, b in zip(session.run(None, feed), actual):
            np.testing.assert_array_equal(a, b)

    def test_lstm_config_is_used_and_disabled_model_restores(self):
        model, feed = graph(bias=False)
        sim = self.sim(model, feed)
        path = self.path / "precision.json"
        path.write_text(json.dumps({"LSTM_config": {"use_quantization": True, "quant_config": {
            "schema_version": 1, "operators": {"input": {"bitwidth": 16},
            "weight_ih": {"granularity": "per_gate"}}}}}))
        self.calibrate(sim, feed, path)
        adapter = sim._rx_native_recurrent["user/lstm"]
        enc = adapter.export_encodings()
        self.assertEqual(enc["activation_encodings"]["user/lstm"]["input"]["0"]["bitwidth"], 16)
        self.assertEqual(adapter.module.get_quant_config("weight_ih")["granularity"], "per_gate")
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "precision", "lstm")
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)
        sim = self.sim(model, feed)
        path.write_text(json.dumps({"LSTM_config": {"use_quantization": False}}))
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)
        apply_power_of_2_workflow(sim, verbose=False)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "disabled", "lstm")
        self.assertEqual(json.loads(artifacts["encodings"].read_text())["excluded_layers"], ["user/lstm"])
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)

    def test_failure_restores_calibration_flags_and_unsupported_attributes_fail(self):
        model, feed = graph()
        sim = self.sim(model, feed)
        def fail(session):
            session.run(None, feed)
            raise RuntimeError("callback failure")
        with self.assertRaisesRegex(RuntimeError, "callback failure"):
            sim.compute_encodings(fail)
        self.assertFalse(sim._rx_native_recurrent["user/lstm"].module.calibrating)
        for key, value in (("input_forget", 1), ("clip", 1.0), ("direction", "reverse")):
            bad = onnx.ModelProto.FromString(model.SerializeToString())
            for attr in list(bad.graph.node[0].attribute):
                if attr.name == key:
                    bad.graph.node[0].attribute.remove(attr)
            bad.graph.node[0].attribute.append(helper.make_attribute(key, value))
            with self.assertRaises(NotImplementedError):
                self.sim(bad, feed)

    def test_missing_internal_grids_and_required_native_state_are_rejected(self):
        from copy import deepcopy
        from aimet_onnx.rx_ptq.recurrent import export_native_encodings, restore_native_encodings
        from aimet_onnx.rx_ptq.validation import validate_quantizer_state

        model, feed = graph()
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        saved = export_native_encodings(sim)
        for kind in ("internal_ops", "internal_ops_reverse"):
            broken = deepcopy(saved)
            del broken["activation_encodings"]["user/lstm"][kind]["cell_state"]
            with self.assertRaisesRegex(ValueError, "internal encodings"):
                restore_native_encodings(sim, {"native_recurrent_encodings": broken}, allow_overwrite=False)
        sim._rx_native_recurrent["user/lstm"].module.use_quantization = False
        with self.assertRaisesRegex(RuntimeError, "missing or disabled"):
            validate_quantizer_state(sim)

    def test_optional_outputs_and_mixed_gru_lstm_graph(self):
        from tests.support.recurrent import gru_graph

        model, feed = graph(directions=1)
        del model.graph.node[0].output[:]
        model.graph.node[0].output.extend(["", "", "cell"])
        del model.graph.output[:2]
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        adapter = sim._rx_native_recurrent["user/lstm"]
        np.testing.assert_array_equal(sim.session.run(None, feed)[0], adapter.run(
            feed["x"], feed["h0"], feed["c0"])[2])

        model, feed = graph()
        other, other_feed = gru_graph()
        # A parallel GRU branch checks both runtime libraries and context registries.
        rename = {name: "gru_" + name for name in other_feed}
        rename.update({item.name: "gru_" + item.name for item in other.graph.initializer})
        rename.update({item.name: "gru_" + item.name for item in other.graph.output})
        for node in other.graph.node:
            for names in (node.input, node.output):
                for i, name in enumerate(names):
                    names[i] = rename.get(name, name)
        for item in list(other.graph.input) + list(other.graph.output) + list(other.graph.initializer):
            item.name = rename[item.name]
        model.graph.node.extend(other.graph.node)
        model.graph.initializer.extend(other.graph.initializer)
        model.graph.input.extend(other.graph.input)
        model.graph.output.extend(other.graph.output)
        feed.update({rename[name]: value for name, value in other_feed.items()})
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        self.assertEqual(set(sim._rx_native_recurrent), {"user/lstm", "user/gru"})
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "mixed", "model")
        pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
            config_file=str(CONFIG / "mrnn_quantsim_config_custom_mixed_precision_v2.json"), providers=PROVIDERS)

    def test_cell_state_output_propagates_its_own_grid(self):
        model, feed = graph(directions=1)
        model.graph.node.append(helper.make_node("Identity", ["cell"], ["cell_copy"], name="cell_copy"))
        model.graph.output.append(helper.make_tensor_value_info("cell_copy", TensorProto.FLOAT, [1, 2, 2]))
        sim = self.sim(model, feed)
        # Test propagation through an unquantized view of the native state.
        sim.qc_quantize_op_dict["cell_copy"].enabled = False
        self.calibrate(sim, feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / "cell", "lstm")
        enc = json.loads(artifacts["encodings"].read_text())["activation_encodings"]
        self.assertEqual(enc["cell_copy"]["output"]["0"],
                         enc["user/lstm"]["internal_ops"]["cell_state"]["output"][0])
        self.assertEqual(enc["cell_copy"]["output"]["0"]["bitwidth"], 16)


if __name__ == "__main__":
    unittest.main()
