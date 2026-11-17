"""Constant-weight ONNX FC graphs preserve one terminal quantization boundary."""

import copy
import json
import unittest

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from aimet_common import libpymo
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow, disable_onnx_bias_quantizers
from tests.support import onnx as fixtures


def linear_graph(shape, *, split=True):
    weight = np.array([[.27, -.37], [.41, .23]], dtype=np.float32)
    bias = np.array([.041, -.063], dtype=np.float32)
    nodes = ([helper.make_node("MatMul", ["x", "w"], ["product"], name="fc"),
              helper.make_node("Add", ["product", "b"], ["y"], name="add")]
             if split else [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="fc")])
    graph = helper.make_graph(nodes, "linear-boundary",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(bias, "b")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def integer_encoding(bits, scale):
    encoding = libpymo.TfEncoding()
    encoding.bw = bits
    encoding.delta = scale
    encoding.offset = -(1 << (bits - 1))
    encoding.min = encoding.offset * scale
    encoding.max = (encoding.offset + (1 << bits) - 1) * scale
    return encoding


class OnnxLinearPtqTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.index = 0

    def sim(self, shape, *, split=True):
        self.index += 1
        graph = linear_graph(shape, split=split)
        original = graph.SerializeToString()
        feed = {"x": np.linspace(-.7, .7, np.prod(shape), dtype=np.float32).reshape(shape)}
        sim = pipeline.create_onnx_ptq_sim(graph, feed, config_file=str(self.fixture.config),
            quant_scheme="tf", work_dir=self.fixture.path / f"linear-{self.index}")
        self.assertEqual(graph.SerializeToString(), original)
        return sim, graph, feed

    def test_split_linear_uses_one_terminal_quantizer_for_2d_and_3d(self):
        for shape in ((32, 2), (2, 16, 2)):
            with self.subTest(shape=shape):
                sim, source, _ = self.sim(shape)
                canonical = sim._rx_normalized_model
                self.assertNotIn("product", sim.qc_quantize_op_dict)
                self.assertTrue(sim.qc_quantize_op_dict["y"].enabled)
                self.assertTrue(sim.qc_quantize_op_dict["b"].enabled)
                self.assertEqual([node.op_type for node in canonical.graph.node].count("Gemm"), 1)
                self.assertFalse(any(node.op_type in ("MatMul", "Add") for node in canonical.graph.node))
                for node in canonical.graph.node:
                    for output in node.output:
                        quantizer = sim.qc_quantize_op_dict.get(output)
                        if output != "y" and quantizer is not None:
                            self.assertFalse(quantizer.enabled, output)
                self.assertEqual([node.op_type for node in source.graph.node], ["MatMul", "Add"])

    def test_fixed_grids_match_torch_linear_with_one_output_rounding(self):
        for shape, split in (((32, 2), False), ((32, 2), True), ((2, 16, 2), True)):
            for po2 in (False, True):
                with self.subTest(shape=shape, split=split, po2=po2):
                    sim, source, feed = self.sim(shape, split=split)
                    sx, sw, sy = (.125, .125, .0625) if po2 else (.1, .2, .07)
                    for name, bits, scales in (("x", 16, [sx]), ("w", 8, [sw, sw]),
                                               ("b", 32, [sx * sw, sx * sw]), ("y", 16, [sy])):
                        quantizer = sim.qc_quantize_op_dict[name]
                        self.assertTrue(quantizer.enabled, name)
                        quantizer.set_bitwidth(bits)
                        quantizer.load_encodings([integer_encoding(bits, scale) for scale in scales])
                    if "product" in sim.qc_quantize_op_dict:
                        intermediate = sim.qc_quantize_op_dict["product"]
                        intermediate.set_bitwidth(16)
                        intermediate.load_encodings([integer_encoding(16, sy)])
                    quantizers = sim.qc_quantize_op_dict
                    x = quantizers["x"].quantize_dequantize(feed["x"])
                    weight = quantizers["w"].quantize_dequantize(numpy_helper.to_array(source.graph.initializer[0]))
                    bias = quantizers["b"].quantize_dequantize(numpy_helper.to_array(source.graph.initializer[1]))
                    reference = torch.nn.Linear(2, 2)
                    with torch.no_grad():
                        reference.weight.copy_(torch.from_numpy(weight.T.copy()))
                        reference.bias.copy_(torch.from_numpy(bias))
                        floating = reference(torch.from_numpy(x)).numpy()
                    expected = quantizers["y"].quantize_dequantize(floating)
                    product = np.matmul(x, weight)
                    double_rounded = quantizers["y"].quantize_dequantize(
                        quantizers["y"].quantize_dequantize(product) + bias)
                    self.assertTrue(np.any(expected != double_rounded), "fixture must distinguish two rounding boundaries")
                    actual = pipeline.run_onnx_feed(sim, feed)[0]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
                    self.assertNotIn("product", sim.qc_quantize_op_dict)

    def test_output_precision_bias_and_canonical_metadata_survive_export_reload(self):
        from aimet_onnx.rx_ptq.linear import linear_groups

        mixed = self.fixture.path / "linear-mixed.json"
        mixed.write_text(json.dumps({"layer_type_config": {"QuantizedLinear": {
            "weight_bitwidth": 8, "bias_bitwidth": 32,
            "input_bitwidth": 8, "output_bitwidth": 16}}}))
        for shape in ((32, 2), (2, 16, 2)):
            with self.subTest(shape=shape):
                sim, source, feed = self.sim(shape)
                original = source.SerializeToString()
                apply_onnx_mixed_precision_bitwidth(sim, str(mixed), verbose=False)
                self.assertEqual(sim.qc_quantize_op_dict["y"].bitwidth, 16)
                disable_onnx_bias_quantizers(sim)
                self.assertIn("b", sim._rx_calibration_bias_quantizers)
                sim.compute_encodings([feed, {"x": -feed["x"]}])
                apply_power_of_2_workflow(sim, verbose=False)
                self.assertTrue(sim.qc_quantize_op_dict["b"].enabled)
                expected = pipeline.run_onnx_feed(sim, feed)
                canonical = sim._rx_normalized_model
                groups = linear_groups(canonical)
                self.assertEqual(set(groups), {"fc"})
                self.assertEqual(groups["fc"]["add"], "add")
                self.assertEqual(groups["fc"]["output"], "y")
                artifacts = pipeline.export_onnx_json(sim, source,
                    self.fixture.path / f"linear-export-{len(shape)}", "model")
                document = json.loads(artifacts["encodings"].read_text())
                self.assertIn("b", document["param_encodings"])
                self.assertEqual(document["param_encodings"]["b"]["dtype"], "INT32")
                clean = onnx.load(artifacts["onnx"])
                self.assertEqual(linear_groups(clean), groups)
                self.assertEqual([node.op_type for node in clean.graph.node],
                                 [node.op_type for node in canonical.graph.node])
                terminal = next(node for node in clean.graph.node if "y" in node.output)
                self.assertEqual(document["activation_encodings"][terminal.name]["output"]["0"]["bitwidth"], 16)
                restored = pipeline.create_onnx_ptq_sim(clean, feed,
                    config_file=str(self.fixture.config), quant_scheme="tf",
                    work_dir=self.fixture.path / f"linear-reload-{len(shape)}")
                self.assertEqual(linear_groups(restored._rx_normalized_model), groups)
                self.assertNotIn("product", restored.qc_quantize_op_dict)
                report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
                    config_file=str(self.fixture.config), quant_scheme="tf")
                self.assertTrue(all(item["max_abs"] == 0 for item in report))
                self.assertEqual(source.SerializeToString(), original)

    @unittest.skipUnless(torch.cuda.is_available(), "native GRU requires CUDA")
    def test_normalized_linear_and_native_gru_calibrate_export_and_reload(self):
        from aimet_onnx.rx_ptq.input_quantization import input_quantizer
        from aimet_onnx.rx_ptq.linear import linear_groups
        from aimet_onnx.rx_ptq.recurrent import restore_native_encodings
        from tests.support.recurrent import gru_graph

        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
        mixed = self.fixture.path / "gru-linear-mixed.json"
        mixed.write_text(json.dumps({"layer_type_config": {"QuantizedLinear": {
            "weight_bitwidth": 8, "bias_bitwidth": 32,
            "input_bitwidth": 16, "output_bitwidth": 16}}}))
        for before_gru in (True, False):
            with self.subTest(before_gru=before_gru):
                source, feed = gru_graph(directions=1, initial_state=False)
                input_name = "x" if before_gru else "state"
                input_shape = [4, 2, 3] if before_gru else [1, 2, 2]
                weight = np.linspace(-.15, .2, input_shape[-1] * 3,
                                     dtype=np.float32).reshape(input_shape[-1], 3)
                source.graph.initializer.extend([
                    numpy_helper.from_array(weight, "fc_w"),
                    numpy_helper.from_array(np.array([.01, -.02, .03], dtype=np.float32), "fc_b"),
                ])
                nodes = [helper.make_node("MatMul", [input_name, "fc_w"], ["fc_product"], name="fc"),
                         helper.make_node("Add", ["fc_product", "fc_b"], ["fc_terminal"], name="fc_add")]
                if before_gru:
                    recurrent = onnx.NodeProto()
                    recurrent.CopyFrom(source.graph.node[0])
                    recurrent.input[0] = "fc_terminal"
                    source.graph.ClearField("node")
                    source.graph.node.extend([*nodes, recurrent])
                else:
                    source.graph.node.extend(nodes)
                    source.graph.output.append(helper.make_tensor_value_info(
                        "fc_terminal", onnx.TensorProto.FLOAT, [*input_shape[:-1], 3]))
                original = source.SerializeToString()
                sim = pipeline.create_onnx_ptq_sim(source, feed,
                    config_file=str(self.fixture.config), quant_scheme="tf", providers=providers,
                    work_dir=self.fixture.path / f"gru-linear-{before_gru}")
                apply_onnx_mixed_precision_bitwidth(sim, str(mixed), verbose=False)
                disable_onnx_bias_quantizers(sim)
                sim.compute_encodings([feed, {name: value * 1.5 for name, value in feed.items()}])
                apply_power_of_2_workflow(sim, verbose=False)
                group = linear_groups(sim._rx_normalized_model)["fc"]
                self.assertNotIn("fc_product", sim.qc_quantize_op_dict)
                for name in (group["gemm_input"], group["gemm_output"]):
                    quantizer = sim.qc_quantize_op_dict.get(name)
                    self.assertTrue(quantizer is None or not quantizer.enabled, name)
                self.assertTrue(sim.qc_quantize_op_dict["fc_terminal"].enabled)
                op = sim.connected_graph.get_all_ops()["fc"]
                input_q = input_quantizer(sim, op, 0)
                self.assertTrue(input_q.is_initialized())
                if not before_gru:
                    alias = sim._rx_input_quantizers["fc", 0]
                    self.assertIs(input_q, sim.qc_quantize_op_dict[alias])
                    self.assertFalse(sim.qc_quantize_op_dict["state"].enabled)
                sx = input_q.get_encodings()[0].delta
                sw = [entry.delta for entry in sim.qc_quantize_op_dict["fc_w"].get_encodings()]
                bias = sim.qc_quantize_op_dict["fc_b"]
                self.assertTrue(bias.enabled)
                np.testing.assert_array_equal([entry.delta for entry in bias.get_encodings()], np.asarray(sw) * sx)
                expected = pipeline.run_onnx_feed(sim, feed, providers)
                artifacts = pipeline.export_onnx_json(sim, source,
                    self.fixture.path / f"gru-linear-export-{before_gru}", "model")
                report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
                    config_file=str(self.fixture.config), quant_scheme="tf", providers=providers)
                self.assertTrue(all(item["max_abs"] == 0 for item in report))

                # Move the identical FC grid before the output reshape, as in a native Gemm graph.
                control_model = copy.deepcopy(sim._rx_normalized_model)
                properties = [entry for entry in control_model.metadata_props
                              if entry.key != "rx_met.linear_normalization"]
                control_model.ClearField("metadata_props")
                control_model.metadata_props.extend(properties)
                control = pipeline.create_onnx_ptq_sim(control_model, feed,
                    config_file=str(self.fixture.config), quant_scheme="tf", providers=providers,
                    work_dir=self.fixture.path / f"gru-gemm-control-{before_gru}")
                for quantizer in control.qc_quantize_op_dict.values():
                    quantizer.enabled = False
                for name, quantizer in sim.qc_quantize_op_dict.items():
                    if not quantizer.enabled:
                        continue
                    target = control.qc_quantize_op_dict[group["gemm_output"] if name == "fc_terminal" else name]
                    target.set_bitwidth(quantizer.bitwidth)
                    target.signed = quantizer.signed
                    target.use_symmetric_encodings = quantizer.use_symmetric_encodings
                    target.use_strict_symmetric = quantizer.use_strict_symmetric
                    target.use_unsigned_symmetric = quantizer.use_unsigned_symmetric
                    target.enabled = True
                    target.load_encodings(quantizer.get_encodings())
                restore_native_encodings(control,
                    json.loads(artifacts["aimet_encodings"].read_text()), allow_overwrite=True)
                actual = pipeline.run_onnx_feed(control, feed, providers)
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right)
                self.assertTrue(control.qc_quantize_op_dict[group["gemm_output"]].enabled)
                terminal = control.qc_quantize_op_dict.get("fc_terminal")
                self.assertTrue(terminal is None or not terminal.enabled)
                self.assertEqual(source.SerializeToString(), original)

    def test_export_rejects_reenabled_intermediate_even_with_identical_scales(self):
        from aimet_onnx.rx_ptq.compiler_encodings import CompilerEncodingConverter
        from aimet_onnx.rx_ptq.linear import linear_groups

        sim, source, feed = self.sim((2, 16, 2))
        disable_onnx_bias_quantizers(sim)
        sim.compute_encodings([feed, {"x": -feed["x"]}])
        apply_power_of_2_workflow(sim, verbose=False)
        group = linear_groups(sim._rx_normalized_model)["fc"]
        terminal = sim.qc_quantize_op_dict[group["output"]]
        if group["gemm_output"] not in sim.qc_quantize_op_dict:
            sim._insert_quantizer(group["gemm_output"], is_param=False)
        intermediate = sim.qc_quantize_op_dict[group["gemm_output"]]
        intermediate.set_bitwidth(terminal.bitwidth)
        intermediate.load_encodings(terminal.get_encodings())
        intermediate.enabled = True
        self.assertEqual(intermediate.get_encodings()[0].delta, terminal.get_encodings()[0].delta)
        with self.assertRaisesRegex(RuntimeError, "multiple.*Linear output|multiple output quantizers"):
            CompilerEncodingConverter(sim, source).convert()


if __name__ == "__main__":
    unittest.main()
