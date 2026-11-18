"""Native recurrent input grids stay independent of ordinary input branches."""

import unittest

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from aimet_common import libpymo
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.input_quantization import input_quantizer
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
from tests.support import onnx as fixtures
from tests.support.recurrent import gru_graph, lstm_graph


def fanout_graph(kind, branch="relu", movement=None, exposed=False):
    model, feed = (gru_graph(directions=1, initial_state=False) if kind == "GRU" else
                   lstm_graph(directions=1, states=(False, False)))
    recurrent = onnx.NodeProto()
    recurrent.CopyFrom(model.graph.node[0])
    nodes = []
    source, values = "x", feed["x"]
    if movement == "reshape":
        source, values = "moved", values.reshape(2, 4, 3)
        model.graph.initializer.append(numpy_helper.from_array(np.array([2, 4, 3], np.int64), "shape"))
        nodes.append(helper.make_node("Reshape", ["x", "shape"], [source], name="move"))
    elif movement == "transpose":
        source, values = "moved", values.transpose(1, 0, 2).copy()
        nodes.append(helper.make_node("Transpose", ["x"], [source], name="move", perm=[1, 0, 2]))
    recurrent.input[0] = source
    nodes.append(recurrent)
    if branch == "relu":
        nodes.append(helper.make_node("Relu", [source], ["branch_output"], name="branch"))
    elif branch == "matmul":
        model.graph.initializer.append(numpy_helper.from_array(np.eye(3, dtype=np.float32), "branch_weight"))
        nodes.append(helper.make_node("MatMul", [source, "branch_weight"], ["branch_output"], name="branch"))
    else:
        nodes.append(helper.make_node("Concat", [source, source], ["branch_output"], name="branch", axis=2))
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    sequence, batch, _ = values.shape
    model.graph.output[0].CopyFrom(helper.make_tensor_value_info("y", 1, [sequence, 1, batch, 2]))
    for output in model.graph.output[1:]:
        output.CopyFrom(helper.make_tensor_value_info(output.name, 1, [1, batch, 2]))
    if source != "x":
        model.graph.value_info.append(helper.make_tensor_value_info(source, 1, values.shape))
    branch_shape = [*values.shape[:-1], 6 if branch == "concat" else 3]
    model.graph.output.append(helper.make_tensor_value_info("branch_output", 1, branch_shape))
    if exposed:
        model.graph.output.append(helper.make_tensor_value_info("x", 1, feed["x"].shape))
    return model, feed, values


def set_grid(quantizer, scale):
    count = len(quantizer.get_encodings()) if quantizer.get_encodings() else 1
    quantizer.set_bitwidth(8)
    encodings = []
    for _ in range(count):
        value = libpymo.TfEncoding()
        value.bw, value.delta, value.offset = 8, scale, -128
        value.min, value.max = -128 * scale, 127 * scale
        encodings.append(value)
    quantizer.load_encodings(encodings)


def set_torch_grid(quantizer, scale):
    quantizer.bitwidth = 8
    quantizer.set_range(torch.full(quantizer.shape, -128 * scale, device="cuda"),
                        torch.full(quantizer.shape, 127 * scale, device="cuda"))
    quantizer._reparametrize_to_scale_offset()
    with torch.no_grad():
        quantizer.scale.fill_(scale)
    quantizer.requires_grad_(False)


def producer_fanout_graph(kind, movement=None):
    model, feed, _ = fanout_graph(kind, movement=movement)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "x":
                node.input[index] = "produced"
    nodes = list(model.graph.node)
    model.graph.ClearField("node")
    model.graph.node.extend([helper.make_node("Mul", ["x", "gain"], ["produced"], name="producer"), *nodes])
    model.graph.initializer.append(numpy_helper.from_array(np.ones(3, dtype=np.float32), "gain"))
    model.graph.value_info.append(helper.make_tensor_value_info("produced", 1, feed["x"].shape))
    return model, feed


class SharedRelu(torch.nn.Module):
    def __init__(self, native, kind):
        super().__init__()
        self.native = native
        self.lstm = kind == "LSTM"
        self.branch = torch.nn.ReLU()

    def forward(self, x):
        output, state = self.native(x)
        if self.lstm:
            return output, state[0], state[1], self.branch(x)
        return output, state, self.branch(x)


class ProducerSharedRelu(SharedRelu):
    def __init__(self, native, kind, movement):
        from aimet_torch._base.nn.modules.custom import Multiply

        super().__init__(native, kind)
        self.producer = Multiply()
        self.register_buffer("gain", torch.ones(3))
        self.movement = movement

    def forward(self, x):
        x = self.producer(x, self.gain)
        if self.movement == "reshape":
            x = x.reshape(2, 4, 3)
        elif self.movement == "transpose":
            x = x.transpose(0, 1).contiguous()
        return super().forward(x)


@unittest.skipUnless(torch.cuda.is_available(), "native recurrent fanout requires CUDA")
class NativeInputFanoutTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
        self.index = 0

    def sim(self, model, feed):
        self.index += 1
        return pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.fixture.config),
            quant_scheme="tf", providers=self.providers,
            work_dir=self.fixture.path / f"native-fanout-{self.index}")

    def calibrate(self, sim, feed):
        sim.compute_encodings([feed, {"x": feed["x"] * 1.5}])
        apply_power_of_2_workflow(sim, verbose=False)

    def torch_reference(self, model, adapter, feed):
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.v2.quantsim import QuantizationSimModel

        payload = adapter.module.export_quant_params_to_aimet_format({}, module_name=adapter.name)
        adapter.module.use_quantization = False
        x = torch.from_numpy(feed["x"]).cuda()
        prepared = prepare_model(model.cuda().eval())
        reference = QuantizationSimModel(prepared, x, quant_scheme="tf",
            config_file=str(self.fixture.config))
        self.assertTrue(reference.model.native.load_quant_params_from_aimet_format(
            payload, module_name=adapter.name, verbose=False))
        reference.model.native.use_quantization = adapter.module.use_quantization = True
        return reference, x

    def test_shared_relu_matches_real_torch_quantsim(self):
        for kind in ("GRU", "LSTM"):
            with self.subTest(kind=kind):
                model, feed, _ = fanout_graph(kind)
                sim = self.sim(model, feed)
                self.calibrate(sim, feed)
                op = sim.connected_graph.get_all_ops()["branch"]
                set_grid(input_quantizer(sim, op, 0), .25)
                set_grid(sim.qc_quantize_op_dict["branch_output"], .25)
                adapter = next(iter(sim._rx_native_recurrent.values()))
                reference, x = self.torch_reference(SharedRelu(adapter.module, kind), adapter, feed)
                for quantizer in (reference.model.branch.input_quantizers[0],
                                  reference.model.branch.output_quantizers[0]):
                    set_torch_grid(quantizer, .25)
                with torch.no_grad():
                    expected = [value.cpu().numpy() for value in reference.model(x)]
                expected[0] = expected[0].reshape(4, 2, 1, 2).transpose(0, 2, 1, 3)
                for actual, target in zip(pipeline.run_onnx_feed(sim, feed, self.providers), expected):
                    np.testing.assert_array_equal(actual, target)

    def test_native_input_domains_survive_movement_export_and_reload(self):
        cases = (("GRU", "relu", "reshape"), ("LSTM", "relu", "transpose"),
                 ("GRU", "matmul", "transpose"), ("LSTM", "concat", "reshape"))
        for kind, branch, movement in cases:
            with self.subTest(kind=kind, branch=branch, movement=movement):
                model, feed, native_values = fanout_graph(kind, branch, movement)
                sim = self.sim(model, feed)
                self.calibrate(sim, feed)
                op = sim.connected_graph.get_all_ops()["branch"]
                self.assertIn(("branch", 0), sim._rx_input_quantizers)
                set_grid(input_quantizer(sim, op, 0), .25)
                set_grid(sim.qc_quantize_op_dict["branch_output"], .25)
                for name in ("x", "moved"):
                    quantizer = sim.qc_quantize_op_dict.get(name)
                    self.assertTrue(quantizer is None or not quantizer.enabled, name)
                adapter = next(iter(sim._rx_native_recurrent.values()))
                expected_native = adapter.run(native_values)
                actual = pipeline.run_onnx_feed(sim, feed, self.providers)
                for output, expected in zip(actual, expected_native):
                    np.testing.assert_array_equal(output, expected)
                artifacts = pipeline.export_onnx_json(sim, model,
                    self.fixture.path / f"export-{kind}-{branch}", "model")
                report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, actual,
                    config_file=str(self.fixture.config), quant_scheme="tf", providers=self.providers)
                self.assertTrue(all(entry["max_abs"] == 0 for entry in report))

    def test_returned_input_keeps_its_output_grid_without_rounding_native_input(self):
        for kind in ("GRU", "LSTM"):
            with self.subTest(kind=kind):
                model, feed, values = fanout_graph(kind, exposed=True)
                sim = self.sim(model, feed)
                self.calibrate(sim, feed)
                returned = sim.qc_quantize_op_dict["x"]
                self.assertTrue(returned.enabled)
                set_grid(returned, .25)
                alias = sim._rx_input_quantizers["branch", 0]
                self.assertIsNot(sim.qc_quantize_op_dict[alias], returned)
                adapter = next(iter(sim._rx_native_recurrent.values()))
                expected_native = adapter.run(values)
                actual = pipeline.run_onnx_feed(sim, feed, self.providers)
                for output, expected in zip(actual, expected_native):
                    np.testing.assert_array_equal(output, expected)
                np.testing.assert_array_equal(actual[-1], returned.quantize_dequantize(feed["x"]))

    def test_compute_producer_output_quantization_is_preserved(self):
        model, feed = producer_fanout_graph("GRU")
        sim = self.sim(model, feed)
        self.calibrate(sim, feed)
        producer_output = sim.qc_quantize_op_dict["produced"]
        self.assertTrue(producer_output.enabled)
        set_grid(producer_output, .25)
        values = sim.qc_quantize_op_dict["x"].quantize_dequantize(feed["x"])
        values = producer_output.quantize_dequantize(values)
        adapter = next(iter(sim._rx_native_recurrent.values()))
        expected = adapter.run(values)
        actual = pipeline.run_onnx_feed(sim, feed, self.providers)
        for output, target in zip(actual, expected):
            np.testing.assert_array_equal(output, target)

    def test_internal_producer_fanout_matches_torch_and_reload(self):
        from aimet_torch.v2.nn.modules.custom import QuantizedMultiply

        for kind in ("GRU", "LSTM"):
            for movement in (None, "reshape", "transpose"):
                with self.subTest(kind=kind, movement=movement):
                    model, feed = producer_fanout_graph(kind, movement)
                    sim = self.sim(model, feed)
                    self.calibrate(sim, feed)
                    producer = sim.connected_graph.get_all_ops()["producer"]
                    branch = sim.connected_graph.get_all_ops()["branch"]
                    set_grid(input_quantizer(sim, producer, 0), .015625)
                    gain = sim.qc_quantize_op_dict.get("gain")
                    if gain is not None and gain.enabled:
                        set_grid(gain, .125)
                    producer_output = sim.qc_quantize_op_dict["produced"]
                    producer_output.enabled = True
                    set_grid(producer_output, .25)
                    branch_input = input_quantizer(sim, branch, 0)
                    set_grid(branch_input, .5)
                    set_grid(sim.qc_quantize_op_dict["branch_output"], .5)
                    adapter = next(iter(sim._rx_native_recurrent.values()))
                    reference, x = self.torch_reference(
                        ProducerSharedRelu(adapter.module, kind, movement), adapter, feed)
                    producer_reference = next(module for module in reference.model.modules()
                                              if isinstance(module, QuantizedMultiply))
                    for quantizer, scale in (
                        (producer_reference.input_quantizers[0], .015625),
                        (producer_reference.input_quantizers[1], .125),
                        (producer_reference.output_quantizers[0], .25),
                        (reference.model.branch.input_quantizers[0], .5),
                        (reference.model.branch.output_quantizers[0], .5),
                    ):
                        set_torch_grid(quantizer, scale)
                    with torch.no_grad():
                        expected = [value.cpu().numpy() for value in reference.model(x)]
                    expected[0] = expected[0].reshape(*expected[0].shape[:2], 1, 2).transpose(0, 2, 1, 3)
                    actual = pipeline.run_onnx_feed(sim, feed, self.providers)
                    for output, target in zip(actual, expected):
                        np.testing.assert_array_equal(output, target)
                    self.assertIsNot(branch_input, producer_output)
                    self.assertTrue(producer_output.enabled)
                    self.assertEqual(producer_output.get_encodings()[0].delta, .25)
                    artifacts = pipeline.export_onnx_json(sim, model,
                        self.fixture.path / f"producer-{kind}-{movement}", "model")
                    report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, actual,
                        config_file=str(self.fixture.config), quant_scheme="tf", providers=self.providers)
                    self.assertTrue(all(entry["max_abs"] == 0 for entry in report))


class NativeRuntimeInputContractTest(unittest.TestCase):
    def test_lstm_keeps_actual_data_and_optional_state_ports(self):
        from quant_lstm_onnx import OnnxLSTM

        cases = (
            (["x_updated", "w", "r"], ["x_updated", "", ""]),
            (["x_updated", "w", "r", "b", "", "h_updated"], ["x_updated", "h_updated", ""]),
            (["x_updated", "w", "r", "b", "", "h_updated", "c_updated", ""],
             ["x_updated", "h_updated", "c_updated"]),
        )
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                adapter = OnnxLSTM.__new__(OnnxLSTM)
                adapter.inputs = [name.removesuffix("_updated") for name in inputs] + [""] * (8 - len(inputs))
                adapter.identifier = 7
                node = helper.make_node("LSTM", inputs, ["y"], name="lstm")
                runtime = adapter.runtime_node(node)
                self.assertEqual(list(runtime.input), expected)
                self.assertEqual(list(runtime.output), ["y", "", ""])


if __name__ == "__main__":
    unittest.main()
