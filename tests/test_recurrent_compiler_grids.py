"""Check compiler grids against real ONNX ports and recurrent directions."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from aimet_common.compiler_graph_validation import validate_compiler_encoding_graph


GRU_INTERNAL = (
    "add_final_hidden", "mul_new_contribution", "mul_old_contribution", "mul_reset_hidden",
    "new_gate_input", "new_gate_output", "reset_gate_input", "reset_gate_output",
    "sub_one_minus_update", "update_gate_input", "update_gate_output",
    "weight_ih_linear", "weight_hh_linear",
)
LSTM_INTERNAL = (
    "input", "output", "cell_state", "weight_ih_linear", "weight_hh_linear",
    "input_gate_input", "forget_gate_input", "cell_gate_input", "output_gate_input",
    "input_gate_output", "forget_gate_output", "cell_gate_output", "output_gate_output",
    "cell_tanh_output",
)


def encoding(scale=2 ** -8, channels=None):
    record = {
        "bitwidth": 8, "dtype": "INT8", "is_symmetric": "True",
        "enc_type": "PER_TENSOR", "scale": scale, "n": int(-np.log2(scale)),
        "zero_point": 0, "real_min": -128 * scale, "real_max": 127 * scale,
    }
    if channels is not None:
        record["enc_type"] = "PER_CHANNEL"
        for field in ("scale", "n", "zero_point", "real_min", "real_max"):
            record[field] = [record[field]] * channels
    return record


def recurrent_graph(kind="GRU", *, direction="bidirectional", consumer=None, output_index=0):
    directions, hidden = (2 if direction == "bidirectional" else 1), 2
    gates = 3 if kind == "GRU" else 4
    names = ["y", "hidden"] + (["cell"] if kind == "LSTM" else [])
    weights = [numpy_helper.from_array(np.zeros(shape, np.float32), name)
               for name, shape in (("W", (directions, gates * hidden, 3)),
                                   ("R", (directions, gates * hidden, hidden)),
                                   ("B", (directions, 2 * gates * hidden)))]
    node = helper.make_node(kind, ["x", "W", "R", "B"], names, name="rnn",
                            direction=direction, hidden_size=hidden)
    nodes = [node]
    if consumer:
        nodes.append(helper.make_node(consumer, [names[output_index]], ["after"], name="after"))
    output = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, directions, 2, hidden])]
    output += [helper.make_tensor_value_info(name, TensorProto.FLOAT, [directions, 2, hidden])
               for name in names[1:]]
    if consumer:
        output.append(helper.make_tensor_value_info("after", TensorProto.FLOAT, None))
    model = helper.make_model(helper.make_graph(nodes, "recurrent-grids",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 2, 3])], output, weights),
        opset_imports=[helper.make_opsetid("", 18)])
    internal_names = GRU_INTERNAL if kind == "GRU" else LSTM_INTERNAL
    internal = {name: {"output": [encoding()]} for name in internal_names}
    entry = {f"is_{kind}": True, "input": {"0": encoding()}, "output": {"0": encoding()},
             "internal_ops": internal}
    if directions == 2:
        entry["internal_ops_reverse"] = copy.deepcopy(internal)
    document = {"schema_version": 4, "activation_encodings": {"rnn": entry},
                "param_encodings": {"W": encoding(channels=directions * gates * hidden),
                                    "R": encoding(channels=directions * gates * hidden),
                                    "B": encoding(channels=directions * 2 * gates * hidden)},
                "excluded_layers": []}
    return model, document


def parameter_graph(kind, *, transpose=False, group=1, lhs=False):
    if kind == "Conv":
        shape, xshape, yshape, channels = (2, 1, 1, 1), [1, 1, 2, 2], [1, 2, 2, 2], 2
    elif kind == "ConvTranspose":
        shape = (2 * group, 3, 1, 1)
        xshape, yshape, channels = [1, 2 * group, 2, 2], [1, 3 * group, 2, 2], 3
    elif kind == "BatchNormalization":
        shape, xshape, yshape, channels = (2,), [1, 2, 2, 2], [1, 2, 2, 2], 2
    elif kind == "MatMul" and lhs:
        shape, xshape, yshape, channels = (2, 3), [3, 4], [2, 4], 3
    else:
        shape = (2, 3) if transpose else (3, 2)
        xshape, yshape, channels = [1, 3], [1, 2], 2
    weights = [numpy_helper.from_array(np.ones(shape, np.float32), "W")]
    inputs, attrs = ["x", "W"], {}
    parameters = {"W": encoding(channels=channels)}
    if kind in ("Conv", "ConvTranspose", "Gemm"):
        bias_count = 3 * group if kind == "ConvTranspose" else 2
        weights.append(numpy_helper.from_array(np.zeros(bias_count, np.float32), "bias"))
        inputs.append("bias")
        parameters["bias"] = encoding(channels=bias_count)
    if kind == "ConvTranspose":
        attrs["group"] = group
    elif kind == "Gemm":
        attrs["transB"] = int(transpose)
    elif kind == "MatMul" and lhs:
        inputs = ["W", "x"]
    elif kind == "BatchNormalization":
        for name in ("bias", "mean", "variance"):
            weights.append(numpy_helper.from_array(np.ones(2, np.float32), name))
            inputs.append(name)
            parameters[name] = encoding(channels=2)
    node = helper.make_node(kind, inputs, ["y"], name="compute", **attrs)
    model = helper.make_model(helper.make_graph([node], "parameter-channels",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, xshape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, yshape)], weights),
        opset_imports=[helper.make_opsetid("", 18)])
    document = {"schema_version": 4, "activation_encodings": {"compute": {"input": {str(1 if lhs else 0): encoding()}}},
                "param_encodings": parameters, "excluded_layers": []}
    return model, document


class CompilerGraphValidationTest(unittest.TestCase):
    def test_parameter_channels_follow_supported_consumer_axis(self):
        cases = (("Conv", {}), ("ConvTranspose", {}), ("ConvTranspose", {"group": 2}),
                 ("Gemm", {}), ("Gemm", {"transpose": True}), ("MatMul", {}),
                 ("MatMul", {"lhs": True}), ("BatchNormalization", {}))
        for kind, kwargs in cases:
            with self.subTest(kind=kind, kwargs=kwargs):
                model, document = parameter_graph(kind, **kwargs)
                validate_compiler_encoding_graph(model, document, require_complete=True)
                for name, record in document["param_encodings"].items():
                    bad = copy.deepcopy(document)
                    bad["param_encodings"][name] = encoding(channels=len(record["scale"]) + 1)
                    with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, "channel"):
                        validate_compiler_encoding_graph(model, bad, require_complete=True)

    def test_conv_channel_count_cannot_match_an_arbitrary_dimension(self):
        model, document = parameter_graph("Conv")
        document["param_encodings"]["W"] = encoding(channels=1)
        with self.assertRaisesRegex(ValueError, "channel"):
            validate_compiler_encoding_graph(model, document)

    def test_shared_parameter_requires_one_consistent_channel_axis(self):
        model, document = parameter_graph("Gemm", transpose=True)
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((2, 2), np.float32), "W"))
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 2
        other = helper.make_node("Gemm", ["x", "W", "bias"], ["other"], name="other", transB=1)
        model.graph.node.append(other)
        model.graph.output.append(helper.make_tensor_value_info("other", TensorProto.FLOAT, [1, 2]))
        document["activation_encodings"]["other"] = {"input": {"0": encoding()}}
        validate_compiler_encoding_graph(model, document, require_complete=True)
        model.graph.node[1].attribute[0].i = 0
        with self.assertRaisesRegex(ValueError, "(axis|axes).*(conflict|incompatible)|incompatible.*(axis|axes)"):
            validate_compiler_encoding_graph(model, document, require_complete=True)

    def test_unknown_multidimensional_or_block_and_gate_context_is_rejected(self):
        for kind in ("PER_BLOCK", "PER_GATE"):
            model, document = parameter_graph("Gemm")
            document["param_encodings"]["W"]["enc_type"] = kind
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "unsupported|Unsupported"):
                validate_compiler_encoding_graph(model, document)
        model, document = parameter_graph("MatMul")
        model.graph.node[0].op_type = "Add"
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 3
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 2
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 3
        with self.assertRaisesRegex(ValueError, "unsupported|Unsupported"):
            validate_compiler_encoding_graph(model, document)

    def test_native_parameter_cannot_bypass_checks_when_shared_with_other_roles(self):
        model, document = recurrent_graph()
        model.graph.node.append(helper.make_node("Identity", ["W"], ["visible_weight"], name="visible_weight"))
        model.graph.output.append(helper.make_tensor_value_info("visible_weight", TensorProto.FLOAT, [2, 6, 3]))
        with self.assertRaisesRegex(ValueError, "incompatible ordinary role"):
            validate_compiler_encoding_graph(model, document)
        model, document = recurrent_graph()
        other = copy.deepcopy(model.graph.node[0])
        other.name = "other"
        for index, name in enumerate(other.output):
            other.output[index] = name + "_other"
        model.graph.node.append(other)
        document["activation_encodings"]["other"] = copy.deepcopy(document["activation_encodings"]["rnn"])
        with self.assertRaisesRegex(ValueError, "shared across multiple parameter ports"):
            validate_compiler_encoding_graph(model, document)

    def test_valid_native_summary_is_not_onnx_state_port_enumeration(self):
        for kind in ("GRU", "LSTM"):
            model, document = recurrent_graph(kind)
            validate_compiler_encoding_graph(model, document)

    def test_terminal_outputs_keep_different_directional_grids(self):
        for kind in ("GRU", "LSTM"):
            model, document = recurrent_graph(kind)
            entry = document["activation_encodings"]["rnn"]
            output = "add_final_hidden" if kind == "GRU" else "output"
            entry["internal_ops_reverse"][output]["output"][0] = encoding(2 ** -18)
            validate_compiler_encoding_graph(model, document)

    def test_mixed_direction_grid_cannot_propagate_to_identity(self):
        for kind in ("GRU", "LSTM"):
            outputs = range(2 if kind == "GRU" else 3)
            for index in outputs:
                with self.subTest(kind=kind, index=index):
                    model, document = recurrent_graph(kind, consumer="Identity", output_index=index)
                    entry = document["activation_encodings"]["rnn"]
                    key = "cell_state" if index == 2 else ("output" if kind == "LSTM" else "add_final_hidden")
                    entry["internal_ops_reverse"][key]["output"][0] = encoding(2 ** -18)
                    document["activation_encodings"]["after"] = {"input": {"0": encoding()}}
                    with self.assertRaisesRegex(ValueError, "direction.*grid|grid.*direction"):
                        validate_compiler_encoding_graph(model, document)

    def test_torch_transpose_keeps_native_direction_grids_without_scalar_claim(self):
        for kind in ("GRU", "LSTM"):
            model, document = recurrent_graph(kind, consumer="Transpose")
            model.graph.node[-1].attribute.append(helper.make_attribute("perm", [0, 2, 1, 3]))
            key = "output" if kind == "LSTM" else "add_final_hidden"
            document["activation_encodings"]["rnn"]["internal_ops_reverse"][key]["output"] = [encoding(2 ** -18)]
            validate_compiler_encoding_graph(model, document)

    def test_compound_grid_cannot_be_claimed_scalar_after_transpose(self):
        model, document = recurrent_graph(consumer="Transpose")
        model.graph.node[-1].attribute.append(helper.make_attribute("perm", [0, 2, 1, 3]))
        model.graph.node.append(helper.make_node("Identity", ["after"], ["last"], name="last"))
        model.graph.output.append(helper.make_tensor_value_info("last", TensorProto.FLOAT, None))
        document["activation_encodings"]["rnn"]["internal_ops_reverse"]["add_final_hidden"]["output"] = [encoding(2 ** -18)]
        document["activation_encodings"]["last"] = {"input": {"0": encoding()}}
        with self.assertRaisesRegex(ValueError, "direction.*grid|grid.*direction"):
            validate_compiler_encoding_graph(model, document)

    def test_output_quantizer_can_end_compound_grid_without_inventing_input_grid(self):
        model, document = recurrent_graph(consumer="Identity")
        document["activation_encodings"]["rnn"]["internal_ops_reverse"]["add_final_hidden"]["output"] = [encoding(2 ** -18)]
        document["activation_encodings"]["after"] = {"output": {"0": encoding(2 ** -10)}}
        model.graph.node.append(helper.make_node("Identity", ["after"], ["last"], name="last"))
        model.graph.output.append(helper.make_tensor_value_info("last", TensorProto.FLOAT, None))
        document["activation_encodings"]["last"] = {"input": {"0": encoding(2 ** -10)}}
        validate_compiler_encoding_graph(model, document)

    def test_uniform_direction_output_can_propagate(self):
        model, document = recurrent_graph(consumer="Identity")
        document["activation_encodings"]["after"] = {"input": {"0": encoding()}, "output": {"0": encoding()}}
        validate_compiler_encoding_graph(model, document)

    def test_consumer_quantizer_can_requantize_mixed_native_values(self):
        model, document = recurrent_graph(consumer="Relu")
        document["activation_encodings"]["rnn"]["internal_ops_reverse"]["add_final_hidden"]["output"] = [encoding(2 ** -18)]
        document["activation_encodings"]["after"] = {"input": {"0": encoding(2 ** -10)}}
        validate_compiler_encoding_graph(model, document)

    def test_unknown_and_empty_or_nonfloating_ports_are_rejected(self):
        for bad in ("999", "01", "-1"):
            model, document = recurrent_graph(consumer="Relu")
            document["activation_encodings"]["after"] = {"input": {bad: encoding()}}
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "port|input"):
                validate_compiler_encoding_graph(model, document)
        model, document = recurrent_graph(consumer="Relu")
        model.graph.node[1].input[0] = ""
        document["activation_encodings"]["after"] = {"input": {"0": encoding()}}
        with self.assertRaisesRegex(ValueError, "empty|port"):
            validate_compiler_encoding_graph(model, document)

    def test_native_kind_direction_internal_coverage_and_packed_lengths(self):
        mutations = (
            lambda d: d["activation_encodings"]["rnn"].update(is_LSTM=True),
            lambda d: d["activation_encodings"]["rnn"].pop("internal_ops_reverse"),
            lambda d: d["activation_encodings"]["rnn"]["internal_ops"].pop("update_gate_input"),
            lambda d: d["param_encodings"]["W"]["scale"].pop(),
            lambda d: d["param_encodings"].update(unknown=encoding(channels=1)),
        )
        for mutate in mutations:
            model, document = recurrent_graph()
            mutate(document)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_compiler_encoding_graph(model, document)

    def test_native_summary_must_match_forward_internal_output(self):
        model, document = recurrent_graph()
        document["activation_encodings"]["rnn"]["output"]["0"] = encoding(2 ** -10)
        with self.assertRaisesRegex(ValueError, "summary|forward"):
            validate_compiler_encoding_graph(model, document)

    def test_scalar_slot_and_container_provenance_remain_legal(self):
        model, document = recurrent_graph(consumer="Relu")
        document["activation_encodings"]["rnn_container"] = {"input": {"0": encoding()}}
        model.graph.node[1].name = "rnn_container.relu"
        with self.assertRaisesRegex(ValueError, "cannot independently verify"):
            validate_compiler_encoding_graph(model, document)
        validate_compiler_encoding_graph(model, document, reference_graph=model)
        document["activation_encodings"]["invented"] = {"input": {"0": encoding()}}
        with self.assertRaisesRegex(ValueError, "invented"):
            validate_compiler_encoding_graph(model, document, reference_graph=model)

    def test_clip_control_ports_and_control_only_parameters_are_rejected(self):
        for transparent in (False, True):
            model, document = recurrent_graph()
            model.graph.initializer.append(numpy_helper.from_array(np.array(1e-12, np.float32), "epsilon"))
            bound = "epsilon"
            if transparent:
                bound = "bound"
                model.graph.node.append(helper.make_node("Identity", ["epsilon"], [bound], name="bound"))
            model.graph.node.append(helper.make_node("Clip", ["y", bound], ["clipped"], name="clip"))
            model.graph.output.append(helper.make_tensor_value_info("clipped", TensorProto.FLOAT, None))
            with self.subTest(transparent=transparent):
                document["activation_encodings"]["clip"] = {"input": {"1": encoding()}}
                with self.assertRaisesRegex(ValueError, "Clip control"):
                    validate_compiler_encoding_graph(model, document)
                document["activation_encodings"].pop("clip")
                document["param_encodings"]["epsilon"] = encoding(channels=1)
                document["param_encodings"]["epsilon"]["enc_type"] = "PER_TENSOR"
                with self.assertRaisesRegex(ValueError, "Clip-only"):
                    validate_compiler_encoding_graph(model, document)
                model.graph.node.append(helper.make_node("Add", ["y", "epsilon"], ["shifted"], name="shifted"))
                model.graph.output.append(helper.make_tensor_value_info("shifted", TensorProto.FLOAT, None))
                validate_compiler_encoding_graph(model, document)

    def test_full_compute_coverage_skips_control_and_python_scalar_constants(self):
        model, document = recurrent_graph()
        model.graph.initializer.append(numpy_helper.from_array(np.array(1e-12, np.float32), "epsilon"))
        model.graph.node.extend([
            helper.make_node("Clip", ["y", "epsilon"], ["clipped"], name="clip"),
            helper.make_node("Constant", [], ["scalar"], name="scalar", value_float=2.),
            helper.make_node("Add", ["clipped", "scalar"], ["shifted"], name="shifted"),
        ])
        model.graph.output.append(helper.make_tensor_value_info("shifted", TensorProto.FLOAT, None))
        document["activation_encodings"].update({
            "clip": {"input": {"0": encoding()}},
            "shifted": {"input": {"0": encoding()}},
        })
        validate_compiler_encoding_graph(model, document, require_complete=True)
        document["activation_encodings"].pop("shifted")
        with self.assertRaisesRegex(ValueError, "shifted.input"):
            validate_compiler_encoding_graph(model, document, require_complete=True)
        document["excluded_layers"].append("shifted")
        validate_compiler_encoding_graph(model, document, require_complete=True)


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
        from test_onnx_gru_native import graph as gru_graph
        from test_onnx_lstm_native import graph as lstm_graph

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
        config = Path(__file__).resolve().parents[1] / "examples/config"
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
                    config = Path(__file__).resolve().parents[1] / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"
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
                    config = Path(__file__).resolve().parents[1] / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"
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
