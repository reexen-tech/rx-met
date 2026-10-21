"""JSON2 operand precision through data movement and constant MatMul inputs."""

import json
import unittest

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
import test_onnx_ptq_alignment as fixtures


class OnnxOperandPrecisionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path
        self.index = 0

    def model(self, nodes, input_shapes, output_shapes, constants=None):
        def infos(shapes):
            return [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                    for name, shape in shapes.items()]
        graph = helper.make_graph(nodes, "operand-precision", infos(input_shapes), infos(output_shapes),
                                  [numpy_helper.from_array(value, name)
                                   for name, value in (constants or {}).items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        feeds = {name: np.linspace(-1, 1, np.prod(shape), dtype=np.float32).reshape(shape)
                 for name, shape in input_shapes.items()}
        self.index += 1
        sim = pipeline.create_onnx_ptq_sim(model, feeds, config_file=str(self.fixture.config),
                                          quant_scheme="tf", work_dir=self.path / f"graph{self.index}")
        return sim, model, feeds

    def apply(self, sim, config):
        self.fixture.apply_config(sim, config)

    def moved_sim(self):
        return self.model([
            helper.make_node("Transpose", ["x"], ["moved"], name="transpose", perm=[1, 0]),
            helper.make_node("Mul", ["moved", "gain"], ["y"], name="window"),
        ], {"x": [2, 4]}, {"y": [4, 2]}, {"gain": np.ones((4, 2), dtype=np.float32)})

    def test_input_precision_reaches_quantizer_before_transpose(self):
        sim, model, feeds = self.moved_sim()
        self.assertNotIn("moved", sim.qc_quantize_op_dict)
        self.apply(sim, {"layer_name_config": {"window": {
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
                sim, _, _ = self.model([
                    helper.make_node("Relu", ["x"], ["before"], name="producer"),
                    helper.make_node("Transpose", ["before"], ["moved"], name="transpose", perm=[1, 0]),
                    helper.make_node("Relu", ["moved"], ["y"], name="consumer"),
                ], {"x": [2, 2]}, {"y": [2, 2]})
                if reverse:
                    ops = dict(reversed(list(sim.connected_graph.get_all_ops().items())))
                    sim.connected_graph.get_all_ops = lambda: ops
                original = {name: q.bitwidth for name, q in sim.qc_quantize_op_dict.items()}
                with self.assertRaisesRegex(ValueError, "Conflicting shared quantizer bitwidth"):
                    self.apply(sim, {"layer_name_config": {
                        "producer": {"output_bitwidth": 16}, "consumer": {"input_bitwidth": 8}}})
                self.assertEqual(original, {name: q.bitwidth for name, q in sim.qc_quantize_op_dict.items()})

    def test_matmul_constant_uses_input_precision_and_per_tensor_grid(self):
        sim, model = self.fixture.sim(bias=False)
        self.apply(sim, {"layer_type_config": {"QuantizedMatMul": {
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
        self.apply(sim, {"layer_type_config": {"QuantizedMatMul": {
            "input_bitwidths": [8, 16], "input_symmetric": False}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 8)
        self.assertEqual(sim.qc_quantize_op_dict["w"].bitwidth, 16)
        self.assertFalse(sim.qc_quantize_op_dict["w"].use_symmetric_encodings)

    def test_named_matmul_input_settings_override_defaults(self):
        sim, _ = self.fixture.sim(bias=False)
        self.apply(sim, {"default_bitwidth": {"weight": 8, "output": 8},
                         "layer_type_config": {"QuantizedMatMul": {"input_bitwidth": 8}},
                         "layer_name_config": {"fc": {"input_bitwidth": 16}}})
        self.assertEqual(sim.qc_quantize_op_dict["w"].bitwidth, 16)
        self.assertFalse(sim.qc_quantize_op_dict["w"].quant_info.usePerChannelMode)

    def test_explicit_matmul_weight_policy_takes_precedence(self):
        sim, _ = self.fixture.sim(bias=False)
        self.apply(sim, {"layer_type_config": {"QuantizedMatMul": {
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
        self.apply(sim, {"layer_type_config": {"QuantizedLinear": {
            "input_bitwidth": 16, "input_symmetric": False, "weight_bitwidth": 8}}})
        self.assertEqual(sim.qc_quantize_op_dict["x"].bitwidth, 16)
        weight = sim.qc_quantize_op_dict["w"]
        self.assertEqual(weight.bitwidth, 8)
        self.assertTrue(weight.use_symmetric_encodings)
        self.assertTrue(weight.quant_info.usePerChannelMode)

    def test_shared_constant_conflict_is_rejected_before_writes(self):
        sim, _, _ = self.model([
            helper.make_node("MatMul", ["x1", "w"], ["y1"], name="left"),
            helper.make_node("MatMul", ["x2", "w"], ["y2"], name="right"),
        ], {"x1": [1, 4], "x2": [1, 4]}, {"y1": [1, 2], "y2": [1, 2]},
            {"w": np.ones((4, 2), dtype=np.float32)})
        original = {name: (q.bitwidth, q.quant_info.usePerChannelMode)
                    for name, q in sim.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(ValueError, "Conflicting shared quantizer bitwidth"):
            self.apply(sim, {"layer_type_config": {"QuantizedMatMul": {"input_bitwidth": 8}},
                             "layer_name_config": {"left": {"input_bitwidths": [8, 16]},
                                                   "right": {"input_bitwidths": [8, 8]}}})
        self.assertEqual(original, {name: (q.bitwidth, q.quant_info.usePerChannelMode)
                                    for name, q in sim.qc_quantize_op_dict.items()})


if __name__ == "__main__":
    unittest.main()
