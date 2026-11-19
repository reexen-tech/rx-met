"""Preserve Torch's parameter and activation grids across eval Conv-BN."""

import copy
import json
import unittest

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from torch import nn

from aimet_common import libpymo
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.conv_bn import (
    apply_conv_bn_boundaries, conv_bn_groups, conv_bn_output_intents,
)
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from tests.support import onnx as fixtures


class ConvBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[.27]], [[-.37]]], [[[.41]], [[.23]]]]))
            self.conv.bias.copy_(torch.tensor([.041, -.063]))
            self.bn.weight.copy_(torch.tensor([1.3, .7]))
            self.bn.bias.copy_(torch.tensor([.17, -.11]))
            self.bn.running_mean.copy_(torch.tensor([.13, -.27]))
            self.bn.running_var.copy_(torch.tensor([.7, 1.3]))

    def forward(self, x):
        return self.bn(self.conv(x))


def conv_bn_graph(module):
    tensors = [('conv.weight', module.conv.weight), ('conv.bias', module.conv.bias),
               ('bn.weight', module.bn.weight), ('bn.bias', module.bn.bias),
               ('bn.running_mean', module.bn.running_mean), ('bn.running_var', module.bn.running_var)]
    graph = helper.make_graph([
        helper.make_node('Conv', ['x', 'conv.weight', 'conv.bias'], ['mid'], name='conv'),
        helper.make_node('BatchNormalization', ['mid', 'bn.weight', 'bn.bias',
            'bn.running_mean', 'bn.running_var'], ['y'], name='bn', epsilon=module.bn.eps),
    ], 'conv_bn', [helper.make_tensor_value_info('x', 1, [2, 2, 3, 3])],
        [helper.make_tensor_value_info('y', 1, [2, 2, 3, 3])],
        [numpy_helper.from_array(value.detach().numpy().copy(), name) for name, value in tensors])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)


def fixed_onnx(quantizer, scale=.0625):
    quantizer.set_bitwidth(8)
    encoding = libpymo.TfEncoding()
    encoding.bw, encoding.delta, encoding.offset = 8, scale, -128
    encoding.min, encoding.max = -128 * scale, 127 * scale
    quantizer.load_encodings([encoding])


class ConvBnBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        config = json.loads(self.fixture.config.read_text())
        config['defaults']['per_channel_quantization'] = 'False'
        config['model_input'] = {'is_input_quantized': 'True'}
        config['model_output'] = {'is_output_quantized': 'True'}
        self.fixture.config.write_text(json.dumps(config))
        self.module = ConvBatchNorm().eval()
        self.feed = {'x': np.linspace(-1.7, 1.2, 36, dtype=np.float32).reshape(2, 2, 3, 3)}

    def test_fixed_grids_match_torch_without_prefolding_quantized_weights(self):
        x = torch.from_numpy(self.feed['x'])
        torch_sim = QuantizationSimModel(prepare_model(copy.deepcopy(self.module)), x,
            config_file=str(self.fixture.config), quant_scheme='tf')
        self.assertIsNone(torch_sim.model.conv.output_quantizers[0])
        self.assertIsNone(torch_sim.model.bn.input_quantizers[0])
        self.assertTrue(all(q is None for q in torch_sim.model.bn.param_quantizers.values()))
        for module in torch_sim.model.modules():
            for role in ('input_quantizers', 'output_quantizers', 'param_quantizers'):
                quantizers = getattr(module, role, ())
                if hasattr(quantizers, 'values'):
                    quantizers = quantizers.values()
                for q in quantizers:
                    if q is not None:
                        q.set_range(torch.full(q.shape, q.qmin * .0625),
                                    torch.full(q.shape, q.qmax * .0625))
        source = conv_bn_graph(self.module)
        original = source.SerializeToString()
        sim = pipeline.create_onnx_ptq_sim(source, self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'sim')
        for q in sim.qc_quantize_op_dict.values():
            if q.enabled:
                fixed_onnx(q)
        with torch.no_grad():
            expected = torch_sim.model(x).numpy()
        actual = pipeline.run_onnx_feed(sim, self.feed)[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(sim.qc_quantize_op_dict['mid'].enabled)
        self.assertTrue(sim.qc_quantize_op_dict['conv.weight'].enabled)
        self.assertTrue(sim.qc_quantize_op_dict['conv.bias'].enabled)
        for name in ('bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'):
            self.assertFalse(sim.qc_quantize_op_dict[name].enabled, name)
        self.assertEqual(source.SerializeToString(), original)

    def test_matcher_excludes_branched_exposed_training_and_dynamic_stats(self):
        for variant in ('branch', 'exposed', 'training', 'dynamic_stats'):
            with self.subTest(variant=variant):
                source = conv_bn_graph(self.module)
                if variant == 'branch':
                    source.graph.node.append(helper.make_node('Relu', ['mid'], ['branch'], name='branch'))
                elif variant == 'exposed':
                    source.graph.output.append(helper.make_tensor_value_info('mid', 1, [2, 2, 3, 3]))
                elif variant == 'training':
                    source.graph.node[-1].attribute.append(helper.make_attribute('training_mode', 1))
                else:
                    source.graph.input.append(helper.make_tensor_value_info('bn.running_mean', 1, [2]))
                self.assertEqual(conv_bn_groups(source), {})

    def test_constant_identity_bn_parameters_are_recognized(self):
        source = conv_bn_graph(self.module)
        source.graph.node[-1].input[4] = 'variance_alias'
        source.graph.node.insert(1, helper.make_node('Identity', ['bn.running_var'],
                                                   ['variance_alias'], name='variance_alias'))
        group = conv_bn_groups(source)['conv']
        self.assertEqual(group['bn_params'][-1], 'variance_alias')
        self.assertIn('bn.running_var', group['bn_param_tensors'])
        self.assertIn('variance_alias', group['bn_param_tensors'])

    def test_constant_conv_parameter_aliases_remain_visible_to_export_validation(self):
        source = conv_bn_graph(self.module)
        source.graph.node[0].input[1] = 'weight_alias'
        source.graph.node.insert(0, helper.make_node('Identity', ['conv.weight'],
                                                    ['weight_alias'], name='weight_alias'))
        group = conv_bn_groups(source)['conv']
        self.assertEqual(group['conv_weight'], 'weight_alias')
        self.assertEqual(group['conv_param_tensors'], ['conv.bias', 'conv.weight', 'weight_alias'])

    def test_configured_conv_output_and_model_output_intents(self):
        source = conv_bn_graph(self.module)
        config = json.loads(self.fixture.config.read_text())
        config['op_type'] = {'Conv': {'is_output_quantized': 'False'},
                             'BatchNormalization': {'is_output_quantized': 'True'}}
        config['model_output'] = {}
        self.fixture.config.write_text(json.dumps(config))
        self.assertEqual(conv_bn_output_intents(source, str(self.fixture.config)), {'conv': False})
        config['model_output'] = {'is_output_quantized': 'True'}
        self.fixture.config.write_text(json.dumps(config))
        self.assertEqual(conv_bn_output_intents(source, str(self.fixture.config)), {'conv': True})

    def test_supergroup_looks_through_foldable_bn(self):
        source = conv_bn_graph(self.module)
        source.graph.node.append(helper.make_node('Relu', ['y'], ['z'], name='relu'))
        source.graph.output[0].name = 'z'
        config = json.loads(self.fixture.config.read_text())
        config['supergroups'] = [{'op_list': ['Conv', 'Relu']}]
        self.fixture.config.write_text(json.dumps(config))
        self.assertEqual(conv_bn_output_intents(source, str(self.fixture.config)), {'conv': False})
        control = nn.Sequential(copy.deepcopy(self.module.conv), copy.deepcopy(self.module.bn), nn.ReLU()).eval()
        sim = QuantizationSimModel(prepare_model(control), torch.from_numpy(self.feed['x']),
            config_file=str(self.fixture.config), quant_scheme='tf')
        self.assertIsNone(sim.model.get_submodule('1').output_quantizers[0])

    def test_repeated_application_preserves_terminal_configuration(self):
        source = conv_bn_graph(self.module)
        sim = pipeline.create_onnx_ptq_sim(source, self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf', work_dir=self.fixture.path / 'repeat')
        apply_conv_bn_boundaries(sim, source, config_file=str(self.fixture.config))
        output = sim.qc_quantize_op_dict['y']
        output.set_bitwidth(16)
        output.enabled = False
        apply_conv_bn_boundaries(sim, source, config_file=str(self.fixture.config))
        self.assertEqual(output.bitwidth, 16)
        self.assertFalse(output.enabled)
        self.assertFalse(sim.qc_quantize_op_dict['mid'].enabled)

    def test_shared_bn_parameters_fail_before_quantizer_mutation(self):
        from aimet_onnx.rx_ptq.input_quantization import InputQuantizationSim
        source = conv_bn_graph(self.module)
        source.graph.node.append(helper.make_node('Concat', ['bn.weight', 'bn.bias'],
                                                   ['shared'], name='shared', axis=0))
        source.graph.output.append(helper.make_tensor_value_info('shared', 1, [4]))
        sim = InputQuantizationSim(copy.deepcopy(source), dummy_input=self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            path=str(self.fixture.path / 'shared'))
        before = {name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(ValueError, 'Conv-BN parameter.*shared'):
            apply_conv_bn_boundaries(sim, source, config_file=str(self.fixture.config))
        self.assertEqual({name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}, before)

    def test_already_folded_conv_control_has_no_group_or_extra_boundary(self):
        folded = torch.nn.utils.fusion.fuse_conv_bn_eval(self.module.conv, self.module.bn)
        graph = helper.make_graph([
            helper.make_node('Conv', ['x', 'w', 'b'], ['y'], name='conv'),
        ], 'folded', [helper.make_tensor_value_info('x', 1, [2, 2, 3, 3])],
            [helper.make_tensor_value_info('y', 1, [2, 2, 3, 3])], [
                numpy_helper.from_array(folded.weight.detach().numpy(), 'w'),
                numpy_helper.from_array(folded.bias.detach().numpy(), 'b'),
            ])
        source = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
        self.assertEqual(conv_bn_groups(source), {})
        sim = pipeline.create_onnx_ptq_sim(source, self.feed, config_file=str(self.fixture.config),
            quant_scheme='tf', work_dir=self.fixture.path / 'folded')
        control = QuantizationSimModel(prepare_model(nn.Sequential(folded).eval()),
            torch.from_numpy(self.feed['x']), config_file=str(self.fixture.config), quant_scheme='tf')
        for module in control.model.modules():
            for role in ('input_quantizers', 'output_quantizers', 'param_quantizers'):
                qs = getattr(module, role, ())
                if hasattr(qs, 'values'):
                    qs = qs.values()
                for q in qs:
                    if q is not None:
                        q.set_range(torch.full(q.shape, q.qmin * .0625),
                                    torch.full(q.shape, q.qmax * .0625))
        for q in sim.qc_quantize_op_dict.values():
            if q.enabled:
                fixed_onnx(q)
        with torch.no_grad():
            expected = control.model(torch.from_numpy(self.feed['x'])).numpy()
        np.testing.assert_array_equal(pipeline.run_onnx_feed(sim, self.feed)[0], expected)
        from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow, disable_onnx_bias_quantizers
        disable_onnx_bias_quantizers(sim)
        sim.compute_encodings([self.feed, {'x': -self.feed['x']}])
        apply_power_of_2_workflow(sim, verbose=False)
        expected = pipeline.run_onnx_feed(sim, self.feed)
        artifacts = pipeline.export_onnx_json(sim, source, self.fixture.path / 'folded-export', 'folded')
        report = pipeline.verify_onnx_ptq_artifacts(artifacts, self.feed, expected,
            config_file=str(self.fixture.config), quant_scheme='tf')
        self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_compiler_export_rejects_unfolded_quantized_parameter_group(self):
        source = conv_bn_graph(self.module)
        original = source.SerializeToString()
        sim = pipeline.create_onnx_ptq_sim(source, self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'unsafe-export')
        for q in sim.qc_quantize_op_dict.values():
            if q.enabled:
                fixed_onnx(q)
        with self.assertRaisesRegex(RuntimeError, 'conv.*bn.*Fold BatchNorm.*recalibrate'):
            pipeline.export_onnx_json(sim, source, self.fixture.path / 'unsafe-pair', 'model')
        self.assertEqual(source.SerializeToString(), original)

    def test_compiler_export_cannot_bypass_guard_with_active_weight_source_alias(self):
        source = conv_bn_graph(self.module)
        source.graph.node[0].input[1] = 'weight_alias'
        source.graph.node.insert(0, helper.make_node('Identity', ['conv.weight'],
                                                    ['weight_alias'], name='weight_alias'))
        sim = pipeline.create_onnx_ptq_sim(source, self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'alias-export')
        self.assertIn('conv.weight', sim.qc_quantize_op_dict)
        for q in sim.qc_quantize_op_dict.values():
            q.enabled = False
        sim.qc_quantize_op_dict['conv.weight'].enabled = True
        fixed_onnx(sim.qc_quantize_op_dict['conv.weight'])
        with self.assertRaisesRegex(RuntimeError, 'conv.*bn.*Fold BatchNorm.*recalibrate'):
            pipeline.export_onnx_json(sim, source, self.fixture.path / 'alias-pair', 'model')

    def test_all_float_group_can_export_and_reload_without_changed_weights(self):
        source = conv_bn_graph(self.module)
        sim = pipeline.create_onnx_ptq_sim(source, self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'float')
        for q in sim.qc_quantize_op_dict.values():
            q.enabled = False
        expected = pipeline.run_onnx_feed(sim, self.feed)
        artifacts = pipeline.export_onnx_json(sim, source, self.fixture.path / 'float-export', 'float')
        exported = onnx.load(str(artifacts['onnx']))
        self.assertEqual([node.op_type for node in exported.graph.node], ['Conv', 'BatchNormalization'])
        for before, after in zip(source.graph.initializer, exported.graph.initializer):
            np.testing.assert_array_equal(numpy_helper.to_array(before), numpy_helper.to_array(after))
        report = pipeline.verify_onnx_ptq_artifacts(artifacts, self.feed, expected,
            config_file=str(self.fixture.config), quant_scheme='tf')
        self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_global_conv_output_setting_is_rejected_even_at_same_bitwidth(self):
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        config = {'layer_type_config': {'QuantizedConv2d': {'output_bitwidth': 8}}}
        mixed = self.fixture.path / 'conv-output.json'
        mixed.write_text(json.dumps(config))
        control = QuantizationSimModel(prepare_model(copy.deepcopy(self.module)),
            torch.from_numpy(self.feed['x']), config_file=str(self.fixture.config), quant_scheme='tf')
        self.assertIsNone(control.model.conv.output_quantizers[0])
        apply_mixed_precision_bitwidth(control.model, str(mixed), verbose=False)
        self.assertEqual(control.model.conv.output_quantizers[0].bitwidth, 8)
        sim = pipeline.create_onnx_ptq_sim(conv_bn_graph(self.module), self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'global-output')
        before = {name: (q.enabled, q.bitwidth, q.signed) for name, q in sim.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(ValueError, 'Conv/BatchNormalization.*internal|internal.*Conv/BatchNormalization'):
            apply_onnx_mixed_precision_bitwidth(sim, str(mixed), verbose=False)
        self.assertEqual({name: (q.enabled, q.bitwidth, q.signed)
                          for name, q in sim.qc_quantize_op_dict.items()}, before)

    def test_named_conv_output_and_bn_input_settings_are_rejected_before_mutation(self):
        for name, settings in (('conv', {'output_bitwidth': 16}), ('bn', {'input_bitwidth': 16})):
            with self.subTest(name=name):
                sim = pipeline.create_onnx_ptq_sim(conv_bn_graph(self.module), self.feed,
                    config_file=str(self.fixture.config), quant_scheme='tf',
                    work_dir=self.fixture.path / f'internal-{name}')
                before = {key: (q.enabled, q.bitwidth, q.signed)
                          for key, q in sim.qc_quantize_op_dict.items()}
                with self.assertRaisesRegex(ValueError, 'Conv/BatchNormalization.*internal|internal.*Conv/BatchNormalization'):
                    self.fixture.apply_config(sim, {'layer_name_config': {name: settings}})
                self.assertEqual({key: (q.enabled, q.bitwidth, q.signed)
                                  for key, q in sim.qc_quantize_op_dict.items()}, before)

    def test_enabled_bn_terminal_output_configuration_is_allowed(self):
        sim = pipeline.create_onnx_ptq_sim(conv_bn_graph(self.module), self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'terminal-output')
        self.assertTrue(sim.qc_quantize_op_dict['y'].enabled)
        self.fixture.apply_config(sim, {'layer_name_config': {'bn': {'output_bitwidth': 16}}})
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)
        self.assertTrue(sim.qc_quantize_op_dict['y'].enabled)
        self.assertFalse(sim.qc_quantize_op_dict['mid'].enabled)
        self.assertEqual(sim.qc_quantize_op_dict['conv.weight'].bitwidth, 8)

    def test_disabled_bn_terminal_output_request_is_rejected_before_mutation(self):
        config = json.loads(self.fixture.config.read_text())
        config['op_type'] = {'Conv': {'is_output_quantized': 'False'},
                             'BatchNormalization': {'is_output_quantized': 'True'}}
        config['model_output'] = {}
        self.fixture.config.write_text(json.dumps(config))
        sim = pipeline.create_onnx_ptq_sim(conv_bn_graph(self.module), self.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'disabled-terminal')
        self.assertFalse(sim.qc_quantize_op_dict['y'].enabled)
        before = {name: (q.enabled, q.bitwidth, q.signed) for name, q in sim.qc_quantize_op_dict.items()}
        with self.assertRaisesRegex(ValueError, 'Conv/BatchNormalization.*terminal|terminal.*Conv/BatchNormalization'):
            self.fixture.apply_config(sim, {'layer_name_config': {'bn': {'output_bitwidth': 16}}})
        self.assertEqual({name: (q.enabled, q.bitwidth, q.signed)
                          for name, q in sim.qc_quantize_op_dict.items()}, before)

    def test_folded_single_conv_accepts_global_output_configuration(self):
        folded = torch.nn.utils.fusion.fuse_conv_bn_eval(self.module.conv, self.module.bn)
        graph = helper.make_graph([
            helper.make_node('Conv', ['x', 'w', 'b'], ['y'], name='conv'),
        ], 'folded-config', [helper.make_tensor_value_info('x', 1, [2, 2, 3, 3])],
            [helper.make_tensor_value_info('y', 1, [2, 2, 3, 3])], [
                numpy_helper.from_array(folded.weight.detach().numpy(), 'w'),
                numpy_helper.from_array(folded.bias.detach().numpy(), 'b'),
            ])
        source = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
        sim = pipeline.create_onnx_ptq_sim(source, self.feed, config_file=str(self.fixture.config),
            quant_scheme='tf', work_dir=self.fixture.path / 'folded-config')
        self.fixture.apply_config(sim, {'layer_type_config': {'QuantizedConv2d': {'output_bitwidth': 16}}})
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)
        self.assertTrue(sim.qc_quantize_op_dict['y'].enabled)


if __name__ == '__main__':
    unittest.main()
