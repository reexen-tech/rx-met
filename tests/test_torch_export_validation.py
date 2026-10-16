"""Validate final PyTorch export artifacts, including injected encoding loss."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
import torch
from torch import nn

from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings

exporter = importlib.import_module('aimet_torch.rx_export.export_onnx_json')
CONFIG = Path(__file__).resolve().parents[1] / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.sigmoid(self.linear(x) + x)


class TorchExportValidationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.x = torch.randn(2, 4)

    def sim(self, model=None, calibrate=True):
        sim = QuantizationSimModel(prepare_model((model or Model()).eval()), self.x,
                                  quant_scheme='tf', config_file=str(CONFIG))
        if calibrate:
            with torch.no_grad(), compute_encodings(sim.model):
                sim.model(self.x)
        return sim

    def export(self, sim):
        return exporter.export_onnx_json(sim, self.tmp.name, 'model', tuple(self.x.shape))

    def corrupt_final_encodings(self, mutate):
        original = exporter.export_recurrent_encodings
        def corrupt(model, onnx_path, enc_path):
            original(model, onnx_path, enc_path)
            path = Path(enc_path)
            document = json.loads(path.read_text())
            mutate(document)
            path.write_text(json.dumps(document))
        return patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt)

    def test_missing_output_fails_export(self):
        sim = self.sim()
        def remove(doc):
            del doc['activation_encodings']['module_add']['output']['0']
        with self.corrupt_final_encodings(remove), self.assertRaisesRegex(
                RuntimeError, r'module_add.*output.*0'):
            self.export(sim)

    def test_missing_whole_module_fails_export(self):
        with self.corrupt_final_encodings(lambda doc: doc['activation_encodings'].pop('linear')), \
             self.assertRaisesRegex(RuntimeError, 'linear'):
            self.export(self.sim())

    def test_missing_second_input_fails_export(self):
        def remove(doc):
            del doc['activation_encodings']['module_add']['input']['1']
        with self.corrupt_final_encodings(remove), self.assertRaisesRegex(
                RuntimeError, r'module_add.*input.*1'):
            self.export(self.sim())

    def test_missing_weight_fails_export(self):
        with self.corrupt_final_encodings(lambda doc: doc['param_encodings'].pop('linear.weight')), \
             self.assertRaisesRegex(RuntimeError, r'linear.weight'):
            self.export(self.sim())

    def test_missing_channel_fails_export(self):
        def remove(doc):
            doc['param_encodings']['linear.weight']['scale'].pop()
        with self.corrupt_final_encodings(remove), self.assertRaisesRegex(RuntimeError, r'linear.weight.*scale'):
            self.export(self.sim())

    def test_invalid_scale_fails_export(self):
        def corrupt(doc):
            doc['activation_encodings']['module_add']['output']['0']['scale'] = float('nan')
        with self.corrupt_final_encodings(corrupt), self.assertRaisesRegex(RuntimeError, r'module_add.*scale'):
            self.export(self.sim())

    def test_uncalibrated_quantizer_fails_export(self):
        with self.assertRaisesRegex(RuntimeError, r'linear.*input.*0.*(calibrat|初始化)'):
            self.export(self.sim(calibrate=False))

    def test_normal_export(self):
        self.export(self.sim())

    def test_disabled_quantizer_needs_no_encoding(self):
        sim = self.sim()
        sim.model.module_add.output_quantizers[0] = None
        _, enc_path = self.export(sim)
        document = json.loads(Path(enc_path).read_text())
        self.assertNotIn('output', document['activation_encodings']['module_add'])

    def test_gelu_intermediates_are_calibrated_before_export(self):
        class Compound(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()
            def forward(self, x):
                return self.gelu(x.pow(2) + x.pow(2))
        sim = self.sim(Compound())
        _, path = self.export(sim)
        enc = json.loads(Path(path).read_text())
        self.assertIn('gelu.erf', enc['activation_encodings'])
        self.assertEqual(set(enc['activation_encodings']['gelu.multiply']['input']), {'0', '1'})
        with self.corrupt_final_encodings(
                lambda doc: doc['activation_encodings'].pop('gelu.erf')), \
             self.assertRaisesRegex(RuntimeError, r'gelu.erf.*missing'):
            self.export(sim)
        with self.corrupt_final_encodings(
                lambda doc: doc['activation_encodings']['gelu.multiply']['input'].pop('1')), \
             self.assertRaisesRegex(RuntimeError, r'gelu.multiply.*input.*1'):
            self.export(sim)

    def test_concat_requires_all_shared_input_encodings(self):
        class Concat(nn.Module):
            def forward(self, x):
                return torch.cat((x, x + x), dim=-1)
        def remove(doc):
            del doc['activation_encodings']['module_cat']['input']['1']
        with self.corrupt_final_encodings(remove), self.assertRaisesRegex(
                RuntimeError, r'module_cat.*input.*1'):
            self.export(self.sim(Concat()))

    def test_scalar_before_tensor_preserves_input_index(self):
        class ScalarFirst(nn.Module):
            def forward(self, x):
                return torch.add(2., x)
        self.export(self.sim(ScalarFirst()))

    def test_single_line_and_multiple_lines_export_identical_encodings(self):
        class Single(nn.Module):
            def forward(self, x):
                return torch.sigmoid((x + x) * x)
        class Multiple(nn.Module):
            def forward(self, x):
                a = x + x
                b = a * x
                return torch.sigmoid(b)
        documents = []
        for model in (Single(), Multiple()):
            _, path = self.export(self.sim(model))
            documents.append(json.loads(Path(path).read_text()))
        self.assertEqual(*documents)

    def test_invalid_final_onnx_fails_export(self):
        original = exporter.export_recurrent_encodings
        def corrupt(model, graph_path, enc_path):
            original(model, graph_path, enc_path)
            graph = onnx.load(graph_path)
            graph.graph.node[0].input[0] = 'nonexistent_tensor'
            onnx.save(graph, graph_path)
        with patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt), \
             self.assertRaisesRegex(RuntimeError, 'ONNX'):
            self.export(self.sim())

    def test_duplicate_final_node_names_fail_export(self):
        original = exporter.export_recurrent_encodings
        def corrupt(model, graph_path, enc_path):
            original(model, graph_path, enc_path)
            graph = onnx.load(graph_path)
            graph.graph.node[1].name = graph.graph.node[0].name
            onnx.save(graph, graph_path)
        with patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt), \
             self.assertRaisesRegex(RuntimeError, 'duplicate name'):
            self.export(self.sim())

    def test_renamed_final_compute_node_fails_export(self):
        original = exporter.export_recurrent_encodings
        def corrupt(model, graph_path, enc_path):
            original(model, graph_path, enc_path)
            graph = onnx.load(graph_path)
            next(node for node in graph.graph.node if node.op_type == 'Add').name = 'unmapped_add'
            onnx.save(graph, graph_path)
        with patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt), \
             self.assertRaisesRegex(RuntimeError, r'unmapped_add.*(encoding|编码)'):
            self.export(self.sim())

    def test_binary_input_required_even_without_quantizer(self):
        sim = self.sim()
        sim.model.module_add.input_quantizers[1] = None
        with self.assertRaisesRegex(RuntimeError, r'module_add.*input.*1'):
            self.export(sim)

    def test_compute_encoding_required_even_without_quantizers(self):
        sim = self.sim()
        module = sim.model.module_add
        module.input_quantizers[0] = None
        module.input_quantizers[1] = None
        module.output_quantizers[0] = None
        with self.assertRaisesRegex(RuntimeError, r'module_add.*(encoding|编码)'):
            self.export(sim)

    def test_eliminated_duplicate_nodes_keep_known_module_encodings(self):
        class Duplicate(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x.pow(2) + x.pow(2))
        self.export(self.sim(Duplicate()))

    def test_orphan_activation_key_fails_export(self):
        def corrupt(doc):
            doc['activation_encodings']['unknown_node'] = doc['activation_encodings']['module_add']
        with self.corrupt_final_encodings(corrupt), self.assertRaisesRegex(
                RuntimeError, r'unknown_node.*no corresponding ONNX node'):
            self.export(self.sim())

    def test_orphan_parameter_key_fails_export(self):
        def corrupt(doc):
            doc['param_encodings']['unknown.weight'] = doc['param_encodings']['linear.weight']
        with self.corrupt_final_encodings(corrupt), self.assertRaisesRegex(
                RuntimeError, r'unknown.weight.*no corresponding ONNX initializer'):
            self.export(self.sim())

    def test_weight_encoding_required_even_without_quantizer(self):
        sim = self.sim()
        sim.model.linear.param_quantizers['weight'] = None
        with self.assertRaisesRegex(RuntimeError, r'linear.*input\[1\]'):
            self.export(sim)

    def test_matmul_requires_both_tensor_inputs(self):
        class Matmul(nn.Module):
            def forward(self, x):
                return torch.matmul(x, x.transpose(-1, -2))
        sim = self.sim(Matmul())
        self.export(sim)
        sim.model.module_matmul.input_quantizers[1] = None
        with self.assertRaisesRegex(RuntimeError, r'module_matmul.*input\[1\]'):
            self.export(sim)

    def test_unary_operator_requires_one_tensor_input(self):
        sim = self.sim()
        sim.model.module_sigmoid.input_quantizers[0] = None
        with self.assertRaisesRegex(RuntimeError, r'module_sigmoid.*input\[0\]'):
            self.export(sim)

    def test_integer_arithmetic_needs_no_float_encoding(self):
        from aimet_torch.rx_export.validation import validate_exported_artifacts
        inputs = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT64, [2])
                  for name in ('x', 'y')]
        output = onnx.helper.make_tensor_value_info('z', onnx.TensorProto.INT64, [2])
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node('Add', ['x', 'y'], ['z'], name='index_add')],
            'integer_arithmetic', inputs, [output])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 18)])
        path = Path(self.tmp.name)
        onnx.save(model, path / 'integer.onnx')
        expected = {'activation_encodings': {}, 'param_encodings': {}}
        (path / 'integer.encodings').write_text(json.dumps(expected))
        validate_exported_artifacts(str(path / 'integer.onnx'), str(path / 'integer.encodings'), expected)

    def test_linear_preserves_fused_quantization_for_all_input_ranks(self):
        import onnxruntime as ort
        from aimet_torch.staged_quantization_utils import load_quantizer_encodings
        class LinearModel(nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = nn.Linear(4, 5, bias=bias)
            def forward(self, x):
                return self.linear(x)
        for shape in ((4,), (2, 4), (2, 3, 4), (2, 3, 5, 4)):
            for bias in (False, True):
                with self.subTest(shape=shape, bias=bias):
                    # A strided last dimension also exercises non-contiguous input.
                    self.x = torch.randn(*shape[:-1], shape[-1] * 2)[..., ::2]
                    model = LinearModel(bias).eval()
                    expected = model(self.x).detach()
                    sim = self.sim(model)
                    quantized = sim.model(self.x).detach()
                    weight = sim.model.linear.weight.detach().clone()
                    graph_path, enc_path = self.export(sim)
                    graph = onnx.load(graph_path)
                    enc = json.loads(Path(enc_path).read_text())
                    compute = [n for n in graph.graph.node if n.op_type in ('Gemm', 'MatMul', 'Add')]
                    self.assertEqual(len(compute), 1)
                    self.assertEqual(compute[0].name, 'linear')
                    self.assertEqual(compute[0].op_type, 'Gemm' if bias else 'MatMul')
                    self.assertEqual(set(enc['activation_encodings']), {'linear'})
                    self.assertEqual(set(enc['activation_encodings']['linear']['input']), {'0'})
                    self.assertEqual(set(enc['activation_encodings']['linear']['output']), {'0'})
                    self.assertLessEqual(set(enc['param_encodings']), {t.name for t in graph.graph.initializer})
                    actual = ort.InferenceSession(graph_path, providers=['CPUExecutionProvider']).run(
                        None, {'input': self.x.numpy()})[0]
                    torch.testing.assert_close(torch.from_numpy(actual), expected)
                    torch.testing.assert_close(sim.model(self.x), quantized, atol=0, rtol=0)
                    torch.testing.assert_close(sim.model.linear.weight, weight, atol=0, rtol=0)
                    fresh = self.sim(model, calibrate=False)
                    load_quantizer_encodings(fresh.model, enc_path, skip_if_not_found=False,
                                            allow_overwrite=False, verbose=False)
                    torch.testing.assert_close(fresh.model(self.x), quantized, atol=1e-6, rtol=1e-5)

    def test_three_dimensional_linear_still_rejects_missing_input_encoding(self):
        self.x = torch.randn(2, 3, 4)
        with self.corrupt_final_encodings(
                lambda doc: doc['activation_encodings']['linear']['input'].pop('0')), \
             self.assertRaisesRegex(RuntimeError, r'linear.*input.*0'):
            self.export(self.sim())

    def test_custom_batchnorm_preserves_parameter_names(self):
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
        class CustomBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = QuantizableBatchNorm2d(4)
                self.bn.running_mean.fill_(0.3)
                self.bn.running_var.fill_(1.7)
            def forward(self, x):
                return self.bn(x)
        self.x = torch.randn(2, 4, 3, 3)
        graph_path, enc_path = self.export(self.sim(CustomBN()))
        graph = onnx.load(graph_path)
        enc = json.loads(Path(enc_path).read_text())
        initializers = {t.name for t in graph.graph.initializer}
        self.assertLessEqual(set(enc['param_encodings']), initializers)
        for op, parameter in [('Mul', 'bn.weight'), ('Add', 'bn.bias')]:
            node = next(n for n in graph.graph.node if n.op_type == op)
            self.assertEqual(node.input[1], parameter)
            self.assertIn(parameter, enc['param_encodings'])
        for node in graph.graph.node:
            if node.op_type in ('Sub', 'Div'):
                self.assertEqual(set(enc['activation_encodings'][node.name]['input']), {'0', '1'})
            elif node.op_type in ('Mul', 'Add'):
                # Parameter QDQ owns gamma/beta; an activation QDQ on this
                # same edge would quantize them a second time.
                self.assertEqual(set(enc['activation_encodings'][node.name]['input']), {'0'})
                self.assertIn(node.input[1], enc['param_encodings'])

    def test_equal_parameters_keep_distinct_names_after_constant_deduplication(self):
        from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
        class Branches(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = QuantizableBatchNorm2d(4)
                self.right = QuantizableBatchNorm2d(4)
                self.left.running_mean.fill_(0.3)
                self.right.running_mean.fill_(-0.8)
            def forward(self, x):
                return self.left(x) + self.right(x)
        self.x = torch.randn(2, 4, 3, 3)
        model = Branches().eval()
        expected = model(self.x).detach().numpy()
        path, enc_path = self.export(self.sim(model))
        graph = onnx.load(path)
        enc = json.loads(Path(enc_path).read_text())
        for owner in ('left', 'right'):
            for op, param in [('Mul', 'weight'), ('Add', 'bias')]:
                node = next(n for n in graph.graph.node
                            if n.name.startswith(owner + '.') and n.op_type == op)
                self.assertEqual(node.input[1], owner + '.' + param)
                self.assertIn(node.input[1], enc['param_encodings'])
        import onnxruntime as ort
        actual = ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(
            None, {'input': self.x.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))

    def test_batchnorm_exposes_unencoded_float_statistics(self):
        class Batchnorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(4)
                self.bn.running_mean.fill_(0.3)
                self.bn.running_var.fill_(1.7)
            def forward(self, x):
                return self.bn(x)
        with self.assertRaisesRegex(RuntimeError, r'bn.running_mean.*missing encoding'):
            self.export(self.sim(Batchnorm()))


if __name__ == '__main__':
    unittest.main()
