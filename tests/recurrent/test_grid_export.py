"""Native recurrent export preserves directional grids and consumer boundaries."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from onnx import helper, numpy_helper
from aimet_common.compiler_graph_validation import validate_compiler_encoding_graph

from tests.support.paths import REPO_ROOT


class NativeRecurrentGridExportTest(unittest.TestCase):
    def setUp(self):
        for dependency in ("torch", "onnxruntime", "quant_gru_onnx_runtime", "quant_lstm_onnx_runtime"):
            if importlib.util.find_spec(dependency) is None:
                self.skipTest(f"native recurrent export requires {dependency}")
        import torch
        if not torch.cuda.is_available():
            self.skipTest("native recurrent export requires CUDA")
        self.temporary = tempfile.TemporaryDirectory(prefix="recurrent-compiler-grid-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def sim(self, kind, layout=0, consumer=None, output_index=0):
        from aimet_onnx.rx_ptq import pipeline
        from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
        from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
        from tests.support.recurrent import gru_graph
        from tests.support.recurrent import lstm_graph

        model, feed = (gru_graph if kind == "GRU" else lstm_graph)(layout=layout)
        for tensor in model.graph.initializer:
            value = numpy_helper.to_array(tensor).copy()
            value[1] *= .001
            tensor.CopyFrom(numpy_helper.from_array(value, tensor.name))
        for name in ("h0", "c0"):
            if name in feed:
                if layout:
                    feed[name][:, 1] *= .001
                else:
                    feed[name][1] *= .001
        if consumer:
            source = model.graph.node[0].output[output_index]
            model.graph.node.append(helper.make_node(consumer, [source], ["after"], name="after"))
            output = copy.deepcopy(model.graph.output[output_index])
            output.name = "after"
            model.graph.output.append(output)
        config = REPO_ROOT / "examples/config"
        sim = pipeline.create_onnx_ptq_sim(model, feed,
            config_file=str(config / "mrnn_quantsim_config_custom_mixed_precision_v2.json"),
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"), work_dir=self.path / kind / str(layout))
        apply_onnx_mixed_precision_bitwidth(sim, str(config / "quick_start_full_quant.json"), verbose=False)
        sim.compute_encodings([feed, {name: value * 1.5 for name, value in feed.items()}])
        apply_power_of_2_workflow(sim, verbose=False)
        return model, feed, sim

    def test_terminal_mixed_grids_export_without_invented_scalar_state_ports(self):
        from aimet_onnx.rx_ptq import pipeline
        from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
        for kind in ("GRU", "LSTM"):
            for layout in (0, 1):
                with self.subTest(kind=kind, layout=layout):
                    model, feed, sim = self.sim(kind, layout)
                    document, _ = CompilerEncodingConverter(sim, model).convert()
                    entry = document["activation_encodings"][model.graph.node[0].name]
                    key = "add_final_hidden" if kind == "GRU" else "output"
                    self.assertNotEqual(entry["internal_ops"][key], entry["internal_ops_reverse"][key])
                    self.assertEqual(set(entry["output"]), {"0"})
                    validate_compiler_encoding_graph(model, document)
                    artifacts = pipeline.export_onnx_json(sim, model, self.path / f"terminal-{kind}-{layout}", "valid")
                    config = REPO_ROOT / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"
                    pipeline.verify_onnx_ptq_artifacts(artifacts, feed, sim.session.run(None, feed),
                        config_file=str(config), providers=("CUDAExecutionProvider", "CPUExecutionProvider"))

    def test_mixed_identity_keeps_native_grids_without_false_scalar_input(self):
        from aimet_onnx.rx_ptq import pipeline
        for kind in ("GRU", "LSTM"):
            for index in range(2 if kind == "GRU" else 3):
                with self.subTest(kind=kind, index=index):
                    model, feed, sim = self.sim(kind, consumer="Identity", output_index=index)
                    adapter = next(iter(sim._rx_native_recurrent.values()))
                    native = adapter.run(feed["x"], *[feed[name] for name in ("h0", "c0") if name in feed])
                    expected = sim.session.run(None, feed)
                    for actual, reference in zip(expected, native):
                        np.testing.assert_array_equal(actual, reference)
                    destination = self.path / f"mixed-{kind}-{index}"
                    artifacts = pipeline.export_onnx_json(sim, model, destination, "valid")
                    import json
                    document = json.loads(artifacts["encodings"].read_text())
                    self.assertNotIn("input", document["activation_encodings"].get("after", {}))
                    config = REPO_ROOT / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"
                    pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
                        config_file=str(config), providers=("CUDAExecutionProvider", "CPUExecutionProvider"))

    def test_compute_input_quantizer_can_follow_mixed_native_output(self):
        from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
        for kind in ("GRU", "LSTM"):
            with self.subTest(kind=kind):
                model, _, sim = self.sim(kind, consumer="Relu")
                alias = sim._rx_input_quantizers["after", 0]
                self.assertTrue(sim.qc_quantize_op_dict[alias].enabled)
                document, _ = CompilerEncodingConverter(sim, model).convert()
                self.assertIn("0", document["activation_encodings"]["after"]["input"])
                validate_compiler_encoding_graph(model, document)

    def test_compute_output_quantizer_does_not_replace_missing_input_boundary(self):
        from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
        for kind in ("GRU", "LSTM"):
            with self.subTest(kind=kind):
                model, _, sim = self.sim(kind, consumer="Relu")
                alias = sim._rx_input_quantizers["after", 0]
                sim.qc_quantize_op_dict[alias].enabled = False
                # Exercise the converter even when the disabled grid is allowed
                # by configuration, rather than failing the live-state check.
                sim._rx_expected_quantizers = set(sim._rx_expected_quantizers) - {alias}
                self.assertTrue(sim.qc_quantize_op_dict["after"].enabled)
                with self.assertRaisesRegex(ValueError, "independent compute input quantizer"):
                    CompilerEncodingConverter(sim, model).convert()


if __name__ == "__main__":
    unittest.main()
