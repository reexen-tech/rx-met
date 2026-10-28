"""ONNX configuration planning, shared-edge conflicts and precision policy."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth

from tests.support.paths import REPO_ROOT
from tests.support import onnx as fixtures


class OnnxConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.path = self.fixture.path
        self.config = self.fixture.config
        self.feed = self.fixture.feed

    def activation_sim(self):
        graph = helper.make_graph([
            helper.make_node('Log', ['x'], ['logged'], name='log'),
            helper.make_node('Clip', ['logged'], ['clipped'], name='clip'),
            helper.make_node('Softmax', ['clipped'], ['y'], name='softmax', axis=1),
        ], 'activations', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        return pipeline.create_onnx_ptq_sim(model, {'x': np.ones((1, 4), dtype=np.float32)},
            config_file=str(self.fixture.config), quant_scheme='tf', work_dir=self.path / 'activations')

    def test_input_settings_do_not_overwrite_parameters(self):
        for matmul_add in (False, True):
            with self.subTest(matmul_add=matmul_add):
                sim, _ = self.fixture.sim(matmul_add=matmul_add)
                self.fixture.apply_config(sim, {'layer_type_config': {'QuantizedLinear': {
                    'weight_bitwidth': 16, 'weight_symmetric': True,
                    'bias_bitwidth': 32, 'bias_symmetric': True,
                    'input_bitwidth': 8, 'input_symmetric': False,
                }}})
                quantizers = sim.qc_quantize_op_dict
                for name, bits, symmetric in [('x', 8, False), ('w', 16, True), ('b', 32, True)]:
                    self.assertEqual(quantizers[name].bitwidth, bits, name)
                    self.assertEqual(quantizers[name].use_symmetric_encodings, symmetric, name)

    def test_missing_activation_quantizer_preserves_input_slot_indices(self):
        sim, _ = self.fixture.sim(bias=False)
        op = SimpleNamespace(name='concat', type='Concat', parameters={},
            inputs=[SimpleNamespace(name=n, is_parm=False) for n in ['missing', 'x']], outputs=[])
        sim.connected_graph.get_all_ops = lambda: {'concat': op}
        with patch.object(sim, '_get_weight_and_bias', return_value=(None, None)):
            self.fixture.apply_config(sim, {'layer_name_config': {'concat': {'input_bitwidths': [4, 16]}}})
        self.assertEqual(sim.qc_quantize_op_dict['x'].bitwidth, 16)

    def test_disabling_layer_still_disables_its_parameters(self):
        sim, _ = self.fixture.sim()
        self.fixture.apply_config(sim, {'layer_name_config': {'fc': {'disable_quantization': True}}})
        self.assertTrue(all(not q.enabled for q in sim.qc_quantize_op_dict.values()))

    def test_shared_edge_conflict_is_detected_before_any_write(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                sim, _ = self.fixture.sim(second_layer=True)
                if reverse:
                    ops = dict(reversed(list(sim.connected_graph.get_all_ops().items())))
                    sim.connected_graph.get_all_ops = lambda: ops
                with self.assertRaisesRegex(ValueError, 'Conflicting shared quantizer bitwidth'):
                    self.fixture.apply_config(sim, {'layer_name_config': {
                        'fc': {'weight_bitwidth': 16, 'output_bitwidth': 16},
                        'fc2': {'input_bitwidth': 8},
                    }})
                self.assertEqual(sim.qc_quantize_op_dict['w'].bitwidth, 8)
                self.assertEqual(sim.qc_quantize_op_dict['mid'].bitwidth, 8)

    def test_shared_edge_symmetry_conflict_is_rejected(self):
        sim, _ = self.fixture.sim(second_layer=True)
        with self.assertRaisesRegex(ValueError, 'Conflicting shared quantizer symmetric'):
            self.fixture.apply_config(sim, {'layer_name_config': {
                'fc': {'output_symmetric': True}, 'fc2': {'input_symmetric': False}}})

    def test_default_is_a_fallback_for_shared_edges(self):
        for settings in [{'fc': {'output_bitwidth': 16}}, {'fc2': {'input_bitwidth': 16}},
                         {'fc': {'output_bitwidth': 16}, 'fc2': {'input_bitwidth': 16}}]:
            with self.subTest(settings=settings):
                sim, _ = self.fixture.sim(second_layer=True)
                self.fixture.apply_config(sim, {'default_bitwidth': {'output': 8},
                                                'layer_name_config': settings})
                self.assertEqual(sim.qc_quantize_op_dict['mid'].bitwidth, 16)

    def test_disabling_shared_edge_cannot_silently_disable_explicit_consumer(self):
        sim, _ = self.fixture.sim(second_layer=True)
        with self.assertRaisesRegex(ValueError, 'Conflicting shared quantizer enabled'):
            self.fixture.apply_config(sim, {'layer_name_config': {
                'fc': {'disable_quantization': True}, 'fc2': {'input_bitwidth': 8}}})
        self.assertTrue(sim.qc_quantize_op_dict['mid'].enabled)

    def test_dedicated_matmul_configuration_wins_over_linear_fallback(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                sim, _ = self.fixture.sim(bias=False)
                entries = [('QuantizedLinear', {'output_bitwidth': 16}),
                           ('QuantizedMatMul', {'output_bitwidth': 4})]
                self.fixture.apply_config(sim, {'layer_type_config': dict(reversed(entries) if reverse else entries)})
                self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 4)

    def test_unknown_json2_types_and_fields_are_rejected(self):
        sim, _ = self.fixture.sim()
        for config in [
            {'layer_type_config': {'QuantizedLinaer': {'input_bitwidth': 8}}},
            {'layer_name_config': {'fc': {'weigth_bitwidth': 8}}},
            {'layer_name_config': {'fc': {'input_symmetric': 'False'}}},
            {'layer_name_config': {'fc': {'weight_bitwidth': True}}},
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.fixture.apply_config(sim, config)

    def test_known_absent_types_and_integer_only_settings_can_be_reused(self):
        sim, _ = self.fixture.sim()
        self.fixture.apply_config(sim, {'layer_type_config': {
            'QuantizedSoftmax': {'output_bitwidth': 16},
            'QuantizedVar': {'input_bitwidth': 16},
            'QuantizedPad': {'disable_quantization': True}}})
        self.assertTrue(sim.qc_quantize_op_dict['x'].enabled)

    def test_repository_json2_can_be_applied_to_onnx(self):
        from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
        sim, _ = self.fixture.sim(bias=False)
        config = REPO_ROOT / 'examples/config/quick_start_full_quant.json'
        result = apply_onnx_mixed_precision_bitwidth(sim, str(config), verbose=False)
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 8)
        self.assertIn('QuantizedVar', result['unsupported_types'])

    def test_onnx_per_channel_setting_is_applied(self):
        sim, _ = self.fixture.sim()
        for enabled in (False, True):
            self.fixture.apply_config(sim, {'layer_type_config': {
                'QuantizedLinear': {'per_channel_quantization': enabled}}})
            self.assertEqual(sim.qc_quantize_op_dict['w'].quant_info.usePerChannelMode, enabled)

    def test_torch_per_channel_setting_preserves_percentile_configuration(self):
        import torch
        from torch import nn
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.v2.quantsim import QuantizationSimModel
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        model = nn.Sequential(nn.Linear(4, 2)).eval()
        sim = QuantizationSimModel(prepare_model(model), torch.ones(1, 4),
            config_file=str(self.fixture.config), quant_scheme='percentile')
        sim.set_percentile_value(99.9)
        config_path = self.path / 'torch-mixed.json'
        for enabled, shape in [(False, ()), (True, (2, 1))]:
            config_path.write_text(json.dumps({'layer_type_config': {
                'QuantizedLinear': {'per_channel_quantization': enabled}}}))
            apply_mixed_precision_bitwidth(sim.model, str(config_path), verbose=False)
            q = sim.model.get_submodule('0').param_quantizers['weight']
            self.assertEqual(q.shape, shape)
            self.assertEqual(q.encoding_analyzer.percentile, 99.9)
        from aimet_torch.v2.nn import compute_encodings
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(torch.ones(1, 4))
            sim.model(torch.arange(4, dtype=torch.float32).reshape(1, 4))
        self.assertEqual(q.get_scale().shape, (2, 1))
        self.assertTrue(torch.all(torch.isfinite(sim.model(torch.ones(1, 4)))))

    def test_onnx_granularity_change_after_calibration_fails_before_writes(self):
        sim, _ = self.fixture.sim()
        self.fixture.calibrate(sim)
        with self.assertRaisesRegex(ValueError, 'before calibration'):
            self.fixture.apply_config(sim, {'layer_type_config': {'QuantizedLinear': {
                'per_channel_quantization': False, 'input_bitwidth': 4}}})
        self.assertEqual(sim.qc_quantize_op_dict['x'].bitwidth, 8)
        self.assertTrue(sim.qc_quantize_op_dict['w'].quant_info.usePerChannelMode)

    def test_log_clamp_softmax_type_settings_and_comment_aliases(self):
        sim = self.activation_sim()
        self.fixture.apply_config(sim, {'layer_type_config': {
            'QuantizedLog': {'input_bitwidth': 8, 'output_bitwidth': 16},
            'QuantizedClip': {'input_bitwidth': 16, 'output_bitwidth': 16, 'comment': 'first alias'},
            'QuantizedClamp': {'input_bitwidth': 16, 'output_bitwidth': 16, 'comment': 'second alias'},
            'QuantizedSoftmax': {'input_bitwidth': 16, 'output_bitwidth': 4},
        }})
        for name, bits in [('x', 8), ('logged', 16), ('clipped', 16), ('y', 4)]:
            self.assertEqual(sim.qc_quantize_op_dict[name].bitwidth, bits)

    def test_conflicting_aliases_fail_but_exact_name_takes_precedence(self):
        sim = self.activation_sim()
        config = {'layer_type_config': {
            'QuantizedClip': {'output_bitwidth': 8}, 'QuantizedClamp': {'output_bitwidth': 16}}}
        with self.assertRaisesRegex(ValueError, 'ambiguous JSON2 aliases'):
            self.fixture.apply_config(sim, config)
        config['layer_name_config'] = {'clip': {'output_bitwidth': 4}}
        self.fixture.apply_config(sim, config)
        self.assertEqual(sim.qc_quantize_op_dict['clipped'].bitwidth, 4)

    def test_convolution_types_use_weight_rank_instead_of_json_order(self):
        weight = numpy_helper.from_array(np.ones((2, 1, 1, 1), dtype=np.float32), 'w')
        graph = helper.make_graph([helper.make_node('Conv', ['x', 'w'], ['y'], name='conv')],
            'conv', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 2, 2])], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        sim = pipeline.create_onnx_ptq_sim(model, {'x': np.ones((1, 1, 2, 2), dtype=np.float32)},
            config_file=str(self.fixture.config), quant_scheme='tf', work_dir=self.path / 'conv')
        self.fixture.apply_config(sim, {'layer_type_config': {
            'QuantizedConv2d': {'output_bitwidth': 16}, 'QuantizedConv1d': {'output_bitwidth': 4}}})
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)

    def test_batchnorm_generic_parameter_settings_are_applied(self):
        tensors = [numpy_helper.from_array(np.ones(2, dtype=np.float32), name)
                   for name in ['scale', 'bias', 'mean', 'variance']]
        graph = helper.make_graph([
            helper.make_node('BatchNormalization', ['x', 'scale', 'bias', 'mean', 'variance'], ['y'], name='bn')
        ], 'bn', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 2, 2])],
           [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2, 2, 2])], tensors)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        sim = pipeline.create_onnx_ptq_sim(model, {'x': np.ones((1, 2, 2, 2), dtype=np.float32)},
            config_file=str(self.fixture.config), quant_scheme='tf', work_dir=self.path / 'bn')
        self.fixture.apply_config(sim, {'layer_type_config': {'QuantizedBatchNorm2d': {
            'param_bitwidth': 16, 'param_symmetric': False}}})
        for name in ['mean', 'variance']:
            self.assertEqual(sim.qc_quantize_op_dict[name].bitwidth, 16)
            self.assertFalse(sim.qc_quantize_op_dict[name].use_symmetric_encodings)


if __name__ == "__main__":
    unittest.main()
