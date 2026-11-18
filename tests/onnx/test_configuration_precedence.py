"""Apply explicit JSON2 activation settings after JSON1 graph configuration."""

import json
import unittest
from collections import OrderedDict

import numpy as np
import torch
from onnx import helper, numpy_helper
from torch import nn

from aimet_common import libpymo
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_torch.model_preparer import prepare_model
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.v2.quantsim import QuantizationSimModel
from tests.support import onnx as fixtures


class ConfigurationPrecedenceTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        config = json.loads(self.fixture.config.read_text())
        config['supergroups'] = [{'op_list': ['Relu', 'Sigmoid']}]
        config['model_input'] = {'is_input_quantized': 'True'}
        config['model_output'] = {'is_output_quantized': 'True'}
        self.fixture.config.write_text(json.dumps(config))
        graph = helper.make_graph([
            helper.make_node('Relu', ['x'], ['mid'], name='op'),
            helper.make_node('Sigmoid', ['mid'], ['y'], name='tail'),
        ], 'precedence', [helper.make_tensor_value_info('x', 1, [1, 4])],
            [helper.make_tensor_value_info('y', 1, [1, 4])])
        self.model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        self.feed = {'x': np.array([[.1, .2, .3, .4]], dtype=np.float32)}
        self.index = 0

    def sim(self):
        self.index += 1
        return pipeline.create_onnx_ptq_sim(self.model, self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / f'sim-{self.index}')

    def apply(self, sim, settings):
        path = self.fixture.path / 'mixed.json'
        path.write_text(json.dumps(settings))
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)
        return path

    def test_explicit_output_bitwidth_restores_the_torch_quantization_point(self):
        sim = self.sim()
        self.assertFalse(sim.qc_quantize_op_dict['mid'].enabled)
        path = self.fixture.path / 'explicit.json'
        path.write_text(json.dumps({'layer_name_config': {'op': {'output_bitwidth': 16}}}))
        control = QuantizationSimModel(prepare_model(nn.Sequential(OrderedDict([
            ('op', nn.ReLU()), ('tail', nn.Sigmoid())])).eval()),
            torch.from_numpy(self.feed['x']), config_file=str(self.fixture.config), quant_scheme='tf')
        self.assertIsNone(control.model.op.output_quantizers[0])
        apply_mixed_precision_bitwidth(control.model, str(path), verbose=False)
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)
        focal = control.model.op.output_quantizers[0]
        target = sim.qc_quantize_op_dict['mid']
        self.assertTrue(target.enabled)
        self.assertEqual(target.bitwidth, focal.bitwidth)
        focal.set_range(torch.tensor(focal.qmin * .25), torch.tensor(focal.qmax * .25))
        focal.requires_grad_(False)
        for module in control.model.modules():
            for role in ('input_quantizers', 'output_quantizers'):
                quantizers = getattr(module, role, ())
                for index, q in enumerate(quantizers):
                    if q is not focal:
                        quantizers[index] = None
        for q in sim.qc_quantize_op_dict.values():
            if q is not target:
                q.enabled = False
        encoding = libpymo.TfEncoding()
        encoding.bw, encoding.delta, encoding.offset = 16, .25, -(1 << 15)
        encoding.min, encoding.max = -(1 << 15) * .25, ((1 << 15) - 1) * .25
        target.load_encodings([encoding])
        with torch.no_grad():
            expected = control.model(torch.from_numpy(self.feed['x'])).numpy()
        # The float Sigmoid kernels may differ by one F32 ULP after identical QDQ.
        np.testing.assert_allclose(pipeline.run_onnx_feed(sim, self.feed)[0], expected, atol=1e-7, rtol=0)

    def test_parameter_bitwidth_does_not_revive_explicitly_disabled_parameters(self):
        config = json.loads(self.fixture.config.read_text())
        config['params'] = {'weight': {'is_quantized': 'False'}, 'bias': {'is_quantized': 'False'}}
        self.fixture.config.write_text(json.dumps(config))
        sim = pipeline.create_onnx_ptq_sim(fixtures.make_graph(), self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'disabled-parameters')
        self.assertFalse(sim.qc_quantize_op_dict['w'].enabled)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.apply(sim, {'layer_name_config': {'fc': {
            'weight_bitwidth': 16, 'bias_bitwidth': 32, 'output_bitwidth': 16}}})
        self.assertFalse(sim.qc_quantize_op_dict['w'].enabled)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.assertTrue(sim.qc_quantize_op_dict['y'].enabled)

    def test_fallback_none_and_domain_only_settings_do_not_restore_disabled_points(self):
        for settings in (
                {'default_bitwidth': {'output': 16}},
                {'layer_name_config': {'op': {'output_symmetric': False, 'output_signed': False}}},
                {'layer_name_config': {'tail': {'input_bitwidths': [None]}}}):
            with self.subTest(settings=settings):
                sim = self.sim()
                self.apply(sim, settings)
                self.assertFalse(sim.qc_quantize_op_dict['mid'].enabled)

    def test_shared_grid_conflict_keeps_flags_domains_and_exclusions_unchanged(self):
        sim = self.sim()
        self.apply(sim, {'layer_name_config': {'op': {'disable_quantization': True}}})
        before = {name: (q.enabled, q.bitwidth, q.signed) for name, q in sim.qc_quantize_op_dict.items()}
        disabled = set(sim._rx_explicit_disabled_quantizers)
        excluded = list(sim._excluded_layer_names)
        with self.assertRaisesRegex(ValueError, 'Conflicting shared quantizer bitwidth'):
            self.apply(sim, {'default_config': {'signed': False}, 'layer_name_config': {
                'op': {'output_bitwidth': 16}, 'tail': {'input_bitwidth': 8}}})
        self.assertEqual({name: (q.enabled, q.bitwidth, q.signed)
                          for name, q in sim.qc_quantize_op_dict.items()}, before)
        self.assertEqual(sim._rx_explicit_disabled_quantizers, disabled)
        self.assertEqual(sim._excluded_layer_names, excluded)

    @unittest.skipUnless(torch.cuda.is_available(), 'native GRU requires CUDA')
    def test_native_input_claim_cannot_disable_a_normalized_linear_output(self):
        from aimet_onnx.rx_ptq.linear import linear_groups
        from tests.support.recurrent import gru_graph
        source, feed = gru_graph(directions=1, initial_state=False)
        weight = np.linspace(-.15, .2, 9, dtype=np.float32).reshape(3, 3)
        source.graph.initializer.extend([
            numpy_helper.from_array(weight, 'fc_w'),
            numpy_helper.from_array(np.array([.01, -.02, .03], dtype=np.float32), 'fc_b'),
        ])
        recurrent = source.graph.node[0]
        recurrent.input[0] = 'fc_terminal'
        source.graph.node.insert(0, helper.make_node('MatMul', ['x', 'fc_w'], ['product'], name='fc'))
        source.graph.node.insert(1, helper.make_node('Add', ['product', 'fc_b'], ['fc_terminal'], name='bias'))
        sim = pipeline.create_onnx_ptq_sim(source, feed, config_file=str(self.fixture.config),
            quant_scheme='tf', providers=('CUDAExecutionProvider', 'CPUExecutionProvider'),
            work_dir=self.fixture.path / 'linear-native-owner')
        group = linear_groups(sim._rx_normalized_model)['fc']
        self.assertIn('fc_terminal', sim._rx_native_io)
        self.assertTrue(sim.qc_quantize_op_dict['fc_terminal'].enabled)
        self.apply(sim, {'layer_type_config': {'QuantizedLinear': {'output_bitwidth': 16}}})
        self.assertTrue(sim.qc_quantize_op_dict['fc_terminal'].enabled)
        self.assertEqual(sim.qc_quantize_op_dict['fc_terminal'].bitwidth, 16)
        for name in (group['gemm_input'], group['gemm_output']):
            q = sim.qc_quantize_op_dict.get(name)
            self.assertTrue(q is None or not q.enabled, name)


class InstanceNormConfigurationTest(unittest.TestCase):
    SETTINGS = {'layer_type_config': {
        'QuantizedInstanceNorm1d': {'output_bitwidth': 4},
        'QuantizedInstanceNorm2d': {'output_bitwidth': 16},
        'QuantizedInstanceNorm3d': {'output_bitwidth': 8},
    }}

    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.index = 0

    def sim(self, shape, *, unbatched=False, unknown_rank=False):
        nodes = []
        initializers = [numpy_helper.from_array(np.ones(2, dtype=np.float32), 'w'),
                        numpy_helper.from_array(np.zeros(2, dtype=np.float32), 'b')]
        if unbatched:
            initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), 'axes'))
            nodes.append(helper.make_node('Unsqueeze', ['x', 'axes'], ['batched'], name='batch'))
        nodes.append(helper.make_node('InstanceNormalization',
            ['batched' if unbatched else 'x', 'w', 'b'], ['normalized' if unbatched else 'y'], name='norm'))
        if unbatched:
            nodes.append(helper.make_node('Squeeze', ['normalized', 'axes'], ['y'], name='unbatch'))
        graph = helper.make_graph(nodes, 'instance-norm',
            [helper.make_tensor_value_info('x', 1, None if unknown_rank else shape)],
            [helper.make_tensor_value_info('y', 1, shape)], initializers)
        source = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        feed = {'x': np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 64}
        self.index += 1
        return pipeline.create_onnx_ptq_sim(source, feed, config_file=str(self.fixture.config),
            quant_scheme='tf', work_dir=self.fixture.path / f'norm-{self.index}')

    def test_instance_norm_dimensions_select_the_matching_node_rank(self):
        for shape, bits in (([1, 2, 3], 4), ([1, 2, 3, 3], 16), ([1, 2, 3, 3, 3], 8)):
            with self.subTest(shape=shape):
                sim = self.sim(shape)
                self.fixture.apply_config(sim, self.SETTINGS)
                self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, bits)

    def test_unbatched_export_uses_the_instance_norm_node_input_rank(self):
        sim = self.sim([2, 3, 3], unbatched=True)
        self.fixture.apply_config(sim, self.SETTINGS)
        self.assertEqual(sim.qc_quantize_op_dict['normalized'].bitwidth, 16)

    def test_unknown_rank_rejects_non_equivalent_aliases_before_mutation(self):
        sim = self.sim([1, 2, 3, 3], unknown_rank=True)
        before = {name: (q.enabled, q.bitwidth) for name, q in sim.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(ValueError, 'ambiguous JSON2 aliases for InstanceNormalization'):
            self.fixture.apply_config(sim, self.SETTINGS)
        self.assertEqual({name: (q.enabled, q.bitwidth) for name, q in sim.qc_quantize_op_dict.items()}, before)


if __name__ == '__main__':
    unittest.main()
