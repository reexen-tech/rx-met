"""GELU mathematical literals retain their Python-scalar quantization semantics."""

import json
import math
import unittest

import numpy as np
import onnx
import torch
from torch import nn
from onnx import helper, numpy_helper, TensorProto

from aimet_common import libpymo
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.math_scalar_constants import identify_math_scalar_constants
from tests.support.onnx import OnnxFixture
from tests.support.torch_export import TorchExportFixture


class GeluModel(nn.Module):
    def __init__(self, approximate):
        super().__init__()
        self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x):
        return self.gelu(x)


def set_onnx_grid(quantizer, scale=.0625):
    quantizer.set_bitwidth(8)
    encoding = libpymo.TfEncoding()
    encoding.bw = 8
    encoding.delta = scale
    encoding.offset = -128
    encoding.min = -128 * scale
    encoding.max = 127 * scale
    quantizer.load_encodings([encoding])


def gelu_graph(approximate='none', *, constant_nodes=False, final_half=False):
    nodes, initializers = [], []

    def scalar(name, value):
        tensor = numpy_helper.from_array(np.array(value, np.float32), name)
        if constant_nodes:
            nodes.append(helper.make_node('Constant', [], [name], name='constant_' + name, value=tensor))
        else:
            initializers.append(tensor)
        return name

    one, half = scalar('one', 1.), scalar('half', .5)
    if approximate == 'none':
        denominator = scalar('sqrt_two', math.sqrt(2.))
        nodes.extend([
            helper.make_node('Div', ['x', denominator], ['normalized'], name='divide'),
            helper.make_node('Erf', ['normalized'], ['activation'], name='erf'),
        ])
    else:
        exponent = scalar('exponent', 3.)
        cubic = scalar('cubic_coefficient', .044715)
        normal = scalar('normal_coefficient', math.sqrt(2. / math.pi))
        nodes.extend([
            helper.make_node('Pow', ['x', exponent], ['cube'], name='cube'),
            helper.make_node('Mul', ['cube', cubic], ['cubic'], name='cubic_scale'),
            helper.make_node('Add', ['x', 'cubic'], ['inner'], name='inner'),
            helper.make_node('Mul', ['inner', normal], ['normalized'], name='normal_scale'),
            helper.make_node('Tanh', ['normalized'], ['activation'], name='tanh'),
        ])
    nodes.append(helper.make_node('Add', ['activation', one], ['shifted'], name='shift'))
    if final_half:
        nodes.extend([
            helper.make_node('Mul', ['x', 'shifted'], ['product'], name='product'),
            helper.make_node('Mul', ['product', half], ['y'], name='half_scale'),
        ])
    else:
        nodes.extend([
            helper.make_node('Mul', ['x', half], ['scaled_input'], name='half_scale'),
            helper.make_node('Mul', ['scaled_input', 'shifted'], ['y'], name='product'),
        ])
    graph = helper.make_graph(nodes, 'gelu',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 16])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 16])], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=8)


class MathScalarConstantTest(unittest.TestCase):
    def fixed_torch_gelu(self, fixture, approximate, x):
        sim = fixture.sim(GeluModel(approximate), calibrate=False, x=x)
        for module in sim.model.modules():
            for kind in ('input_quantizers', 'output_quantizers'):
                for quantizer in getattr(module, kind, ()):
                    if quantizer is not None:
                        quantizer.set_range(torch.full(quantizer.shape, quantizer.qmin * .0625),
                                            torch.full(quantizer.shape, quantizer.qmax * .0625))
        return sim

    def test_rx_gelu_erf_and_tanh_match_torch_with_fixed_grids(self):
        torch_fixture = TorchExportFixture(self)
        onnx_fixture = OnnxFixture(self)
        torch.manual_seed(51)
        x = torch.randn(2, 16) * .8 + torch.linspace(-.4, .3, 32).reshape(2, 16)
        for approximate in ('none', 'tanh'):
            with self.subTest(approximate=approximate):
                source = self.fixed_torch_gelu(torch_fixture, approximate, x)
                model_path, _ = torch_fixture.export(source, x=x)
                model = onnx.load(model_path)
                feed = {model.graph.input[0].name: x.numpy()}
                sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(onnx_fixture.config),
                    quant_scheme='tf', work_dir=onnx_fixture.path / approximate)
                for quantizer in sim.qc_quantize_op_dict.values():
                    if quantizer.enabled:
                        set_onnx_grid(quantizer)
                protected = identify_math_scalar_constants(model)
                self.assertEqual(len(protected), 3 if approximate == 'none' else 5)
                for name in set(protected.values()):
                    quantizer = sim.qc_quantize_op_dict.get(name)
                    self.assertTrue(quantizer is None or not quantizer.enabled, name)
                with torch.no_grad():
                    expected = source.model(x).numpy()
                actual = sim.session.run(None, feed)[0]
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)
                artifacts = pipeline.export_onnx_json(sim, model, onnx_fixture.path / ('export-' + approximate), 'gelu')
                document = json.loads(artifacts['encodings'].read_text())
                for (name, index), tensor in protected.items():
                    self.assertNotIn(str(index), document['activation_encodings'].get(name, {}).get('input', {}))
                    self.assertNotIn(tensor, document['param_encodings'])
                report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, [actual],
                    config_file=str(onnx_fixture.config), quant_scheme='tf')
                self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_complete_formulas_recognize_initializer_and_constant_ports(self):
        for approximate in ('none', 'tanh'):
            for constant_nodes in (False, True):
                for final_half in (False, True):
                    with self.subTest(approximate=approximate, constant_nodes=constant_nodes,
                                      final_half=final_half):
                        model = gelu_graph(approximate, constant_nodes=constant_nodes, final_half=final_half)
                        before = model.SerializeToString()
                        expected = {('shift', 1): 'one', ('half_scale', 1): 'half'}
                        expected.update({('divide', 1): 'sqrt_two'} if approximate == 'none' else {
                            ('cube', 1): 'exponent', ('cubic_scale', 1): 'cubic_coefficient',
                            ('normal_scale', 1): 'normal_coefficient'})
                        self.assertEqual(identify_math_scalar_constants(model), expected)
                        self.assertEqual(model.SerializeToString(), before)

    def test_shared_literals_exempt_only_gelu_consumers(self):
        model = gelu_graph()
        model.graph.node.append(helper.make_node('Mul', ['x', 'half'], ['parameter_result'],
                                                 name='tensor_parameter'))
        model.graph.output.append(helper.make_tensor_value_info('parameter_result', TensorProto.FLOAT, [2, 16]))
        protected = identify_math_scalar_constants(model)
        self.assertEqual(protected[('half_scale', 1)], 'half')
        self.assertNotIn(('tensor_parameter', 1), protected)

    def test_shared_initializer_uses_raw_math_value_and_quantized_tensor_value(self):
        fixture = OnnxFixture(self)
        torch_fixture = TorchExportFixture(self)
        model = gelu_graph()
        model.graph.node.append(helper.make_node('Mul', ['x', 'half'], ['parameter_result'],
                                                 name='tensor_parameter'))
        model.graph.output.append(helper.make_tensor_value_info('parameter_result', TensorProto.FLOAT, [2, 16]))
        x = torch.linspace(-2., 2., 32).reshape(2, 16)
        feed = {'x': x.numpy()}
        sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(fixture.config),
            quant_scheme='tf', work_dir=fixture.path / 'shared')
        for quantizer in sim.qc_quantize_op_dict.values():
            if quantizer.enabled:
                set_onnx_grid(quantizer)
        self.assertTrue(sim.qc_quantize_op_dict['half'].enabled)
        set_onnx_grid(sim.qc_quantize_op_dict['half'], 1.)
        quantized_half = sim.qc_quantize_op_dict['half'].quantize_dequantize(np.array(.5, np.float32))
        self.assertEqual(np.asarray(quantized_half).item(), 0.)
        source = self.fixed_torch_gelu(torch_fixture, 'none', x)
        with torch.no_grad():
            expected = source.model(x).numpy()
        actual = sim.session.run(None, feed)
        np.testing.assert_allclose(actual[0], expected, rtol=0, atol=2e-7)
        np.testing.assert_array_equal(actual[1], np.zeros_like(actual[1]))
        with self.assertRaisesRegex(RuntimeError, '(?i)(math|gelu)'):
            pipeline.export_onnx_json(sim, model, fixture.path / 'shared-conflicting-export', 'gelu')
        set_onnx_grid(sim.qc_quantize_op_dict['half'], .25)
        actual = sim.session.run(None, feed)
        np.testing.assert_allclose(actual[0], expected, rtol=0, atol=2e-7)
        quantized_input = sim.qc_quantize_op_dict['x'].quantize_dequantize(feed['x'])
        expected_tensor = sim.qc_quantize_op_dict['parameter_result'].quantize_dequantize(quantized_input * .5)
        self.assertTrue(np.any(expected_tensor != 0))
        np.testing.assert_array_equal(actual[1], expected_tensor)
        artifacts = pipeline.export_onnx_json(sim, model, fixture.path / 'shared-export', 'gelu')
        document = json.loads(artifacts['encodings'].read_text())
        self.assertNotIn('1', document['activation_encodings']['half_scale']['input'])
        ordinary = document['activation_encodings']['tensor_parameter'].get('input', {})
        self.assertTrue('1' in ordinary or 'half' in document['param_encodings'])
        report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, actual,
            config_file=str(fixture.config), quant_scheme='tf')
        self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_ordinary_scalar_parameters_and_tensor_constants_are_not_exempt(self):
        for constant_nodes in (False, True):
            with self.subTest(constant_nodes=constant_nodes):
                model = gelu_graph(constant_nodes=constant_nodes)
                nodes = [node for node in model.graph.node
                         if node.op_type == 'Constant' and node.output[0] == 'half']
                nodes.append(helper.make_node('Mul', ['x', 'half'], ['y'], name='tensor_parameter'))
                model.graph.ClearField('node')
                model.graph.node.extend(nodes)
                parameters = [tensor for tensor in model.graph.initializer if tensor.name == 'half']
                model.graph.ClearField('initializer')
                model.graph.initializer.extend(parameters)
                self.assertEqual(identify_math_scalar_constants(model), {})
                fixture = OnnxFixture(self)
                feed = {'x': np.ones((2, 16), np.float32)}
                sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(fixture.config),
                    quant_scheme='tf', work_dir=fixture.path / 'ordinary')
                self.assertTrue(sim.qc_quantize_op_dict['half'].enabled)

    def test_changed_or_unproven_formulas_do_not_partially_exempt_literals(self):
        cases = ('near_coefficient', 'tensor_literal', 'double_literal', 'overridable_literal',
                 'different_data_input', 'custom_operator', 'intermediate_output', 'side_consumer')
        for case in cases:
            with self.subTest(case=case):
                model = gelu_graph('tanh')
                tensors = {tensor.name: tensor for tensor in model.graph.initializer}
                nodes = {node.name: node for node in model.graph.node}
                if case == 'near_coefficient':
                    value = np.nextafter(np.float32(.044715), np.float32(1.))
                    tensors['cubic_coefficient'].CopyFrom(numpy_helper.from_array(np.array(value), 'cubic_coefficient'))
                elif case == 'tensor_literal':
                    tensors['half'].CopyFrom(numpy_helper.from_array(np.array([.5], np.float32), 'half'))
                elif case == 'double_literal':
                    tensors['half'].CopyFrom(numpy_helper.from_array(np.array(.5, np.float64), 'half'))
                elif case == 'overridable_literal':
                    model.graph.input.append(helper.make_tensor_value_info('half', TensorProto.FLOAT, []))
                elif case == 'different_data_input':
                    model.graph.input.append(helper.make_tensor_value_info('other', TensorProto.FLOAT, [2, 16]))
                    nodes['half_scale'].input[0] = 'other'
                elif case == 'custom_operator':
                    nodes['tanh'].domain = 'custom.test'
                elif case == 'intermediate_output':
                    model.graph.output.append(helper.make_tensor_value_info('activation', TensorProto.FLOAT, [2, 16]))
                elif case == 'side_consumer':
                    model.graph.node.append(helper.make_node('Identity', ['activation'], ['side'], name='side'))
                self.assertEqual(identify_math_scalar_constants(model), {})


if __name__ == '__main__':
    unittest.main()
