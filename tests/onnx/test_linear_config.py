"""Configuration and bias contracts for normalized ONNX Linear groups."""

import json
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper

from aimet_onnx.rx_ptq import mixed_precision, pipeline
from aimet_onnx.rx_ptq.linear import linear_groups, normalize_linear_graph
from aimet_onnx.rx_ptq.po2 import align_onnx_bias_scale
from tests.support import onnx as fixtures


def normalized_fc():
    return normalize_linear_graph(fixtures.make_graph(matmul_add=True))


class NormalizedLinearConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)

    def sim(self, model=None):
        model = normalized_fc() if model is None else model
        return pipeline.create_onnx_ptq_sim(model, self.fixture.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / 'sim')

    def test_original_add_alias_sets_live_terminal_output(self):
        sim = self.sim()
        self.fixture.apply_config(sim, {'layer_name_config': {
            'fc': {'input_bitwidth': 8, 'weight_bitwidth': 16, 'bias_bitwidth': 32},
            'add': {'output_bitwidth': 16},
        }})
        self.assertEqual(sim.qc_quantize_op_dict['w'].bitwidth, 16)
        self.assertEqual(sim.qc_quantize_op_dict['b'].bitwidth, 32)
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)

    def test_alias_metadata_survives_model_serialization(self):
        model = onnx.load_model_from_string(normalized_fc().SerializeToString())
        sim = self.sim(model)
        self.fixture.apply_config(sim, {'layer_name_config': {'add': {'output_bitwidth': 16}}})
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)

    def test_conflicting_alias_outputs_fail_before_mutations(self):
        sim = self.sim()
        with self.assertRaisesRegex(ValueError, 'Conflicting.*(Linear|linear|output)'):
            self.fixture.apply_config(sim, {'layer_name_config': {
                'fc': {'weight_bitwidth': 16, 'output_bitwidth': 8},
                'add': {'output_bitwidth': 16},
            }})
        self.assertEqual(sim.qc_quantize_op_dict['w'].bitwidth, 8)

    def test_original_add_input_configuration_is_rejected(self):
        sim = self.sim()
        with self.assertRaisesRegex(ValueError, '(internal|intermediate).*add|add.*(internal|intermediate)'):
            self.fixture.apply_config(sim, {'layer_name_config': {'add': {'input_bitwidth': 16}}})

    def test_disabling_only_original_add_cannot_disable_the_whole_linear(self):
        sim = self.sim()
        with self.assertRaisesRegex(ValueError, 'add.*disable_quantization'):
            self.fixture.apply_config(sim, {'layer_name_config': {'add': {'disable_quantization': True}}})
        self.assertTrue(sim.qc_quantize_op_dict['w'].enabled)

    def test_explicitly_disabling_both_original_nodes_disables_the_group(self):
        sim = self.sim()
        self.fixture.apply_config(sim, {'layer_name_config': {
            'fc': {'disable_quantization': True}, 'add': {'disable_quantization': True},
        }})
        self.assertTrue(all(not sim.qc_quantize_op_dict[name].enabled for name in ('x', 'w', 'b', 'y')))

    def test_explicit_matrix_parameter_port_configuration_is_not_silently_lost(self):
        sim = self.sim()
        with self.assertRaisesRegex(ValueError, 'fc.*input_bitwidths.*weight'):
            self.fixture.apply_config(sim, {'layer_name_config': {'fc': {'input_bitwidths': [8, 16]}}})

    def test_linear_type_does_not_inherit_general_matmul_configuration(self):
        sim = self.sim()
        self.fixture.apply_config(sim, {'layer_type_config': {
            'QuantizedMatMul': {'input_bitwidth': 4, 'output_bitwidth': 4},
            'QuantizedAdd': {'output_bitwidth': 8},
            'QuantizedLinear': {'input_bitwidth': 8, 'output_bitwidth': 16},
        }})
        self.assertEqual(sim.qc_quantize_op_dict['x'].bitwidth, 8)
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)

    def test_none_and_builtin_default_config_accept_split_linear(self):
        for config in (None, 'default'):
            with self.subTest(config=config):
                sim = pipeline.create_onnx_ptq_sim(fixtures.make_graph(matmul_add=True), self.fixture.feed,
                    config_file=config, quant_scheme='tf',
                    work_dir=self.fixture.path / f'alias-{config}')
                self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
                self.assertFalse(sim.qc_quantize_op_dict['w'].quant_info.usePerChannelMode)
                self.assertTrue(sim.qc_quantize_op_dict['y'].enabled)

    def test_generated_reshape_defaults_do_not_reconfigure_linear_outputs(self):
        model = fixtures.make_graph(matmul_add=True)
        model.graph.input[0].CopyFrom(helper.make_tensor_value_info('x', 1, [1, 1, 4]))
        model.graph.output[0].CopyFrom(helper.make_tensor_value_info('y', 1, [1, 1, 2]))
        model = normalize_linear_graph(model)
        group = linear_groups(model)['fc']
        self.fixture.feed['x'] = self.fixture.feed['x'].reshape(1, 1, 4)
        sim = self.sim(model)
        self.fixture.apply_config(sim, {'default_bitwidth': {'output': 8},
            'layer_type_config': {'QuantizedLinear': {'input_bitwidth': 16, 'output_bitwidth': 16}}})
        self.assertEqual(sim.qc_quantize_op_dict['x'].bitwidth, 16)
        self.assertFalse(sim.qc_quantize_op_dict[group['gemm_output']].enabled)
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)


class LinearJson1ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)

    def validate(self, config):
        self.fixture.config.write_text(json.dumps(config))
        mixed_precision.validate_linear_quantizer_config(normalized_fc(), str(self.fixture.config))

    def test_equivalent_generic_op_settings_are_allowed(self):
        config = json.loads(self.fixture.config.read_text())
        config['op_type'] = {name: {'is_input_quantized': 'True', 'is_output_quantized': 'True'}
                             for name in ('Gemm', 'MatMul', 'Add')}
        self.validate(config)

    def test_builtin_config_alias_is_resolved_before_preflight(self):
        for config in (None, 'default'):
            with self.subTest(config=config):
                mixed_precision.validate_linear_quantizer_config(normalized_fc(), config)

    def test_nondefault_matrix_settings_cannot_be_silently_lost(self):
        for field, value in [('is_symmetric', 'False'), ('per_channel_quantization', 'False'),
                             ('is_input_quantized', 'False')]:
            with self.subTest(field=field):
                config = json.loads(self.fixture.config.read_text())
                config['op_type'] = {'MatMul': {field: value}}
                with self.assertRaisesRegex(ValueError, 'fc.*' + field):
                    self.validate(config)

    def test_disabled_internal_edge_is_rejected(self):
        config = json.loads(self.fixture.config.read_text())
        config['op_type'] = {'Add': {'is_input_quantized': 'False'}}
        with self.assertRaisesRegex(ValueError, 'add.*is_input_quantized'):
            self.validate(config)
    def test_only_active_inequivalent_supergroups_are_rejected(self):
        config = json.loads(self.fixture.config.read_text())
        config['supergroups'] = [{'op_list': ['Add', 'Relu']}]
        self.validate(config)
        model = normalized_fc()
        model.graph.node.extend([helper.make_node('Relu', ['y'], ['z'], name='relu')])
        model.graph.output[0].name = 'z'
        self.fixture.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'supergroup'):
            mixed_precision.validate_linear_quantizer_config(model, str(self.fixture.config))
        for alias in (None, 'default'):
            mixed_precision.validate_linear_quantizer_config(model, alias)

    def test_matmul_add_pass_and_equivalent_supergroup_are_allowed(self):
        config = json.loads(self.fixture.config.read_text())
        config['supergroup_pass_list'] = ['MatmulAdd']
        config['supergroups'] = [{'op_list': ['MatMul', 'Add']}]
        self.validate(config)


class SharedLinearBiasTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)

    def sim(self, *, same_inputs=False):
        model = normalized_fc()
        del model.metadata_props[:]
        model.graph.node.extend([helper.make_node('Gemm', ['x2', 'w', 'b'], ['y2'], name='fc2')])
        model.graph.input.extend([helper.make_tensor_value_info('x2', 1, [1, 4])])
        model.graph.output.extend([helper.make_tensor_value_info('y2', 1, [1, 2])])
        feed = dict(self.fixture.feed)
        feed['x2'] = feed['x'] if same_inputs else 20 * feed['x']
        sim = pipeline.create_onnx_ptq_sim(model, feed, config_file=str(self.fixture.config),
            quant_scheme='tf', work_dir=self.fixture.path / 'shared')
        from aimet_onnx.rx_ptq.po2 import disable_onnx_bias_quantizers
        disable_onnx_bias_quantizers(sim)
        sim.compute_encodings([feed])
        return sim

    def test_conflicting_shared_bias_scales_fail_before_reenabling_bias(self):
        sim = self.sim()
        bias = sim.qc_quantize_op_dict['b']
        old_bitwidth = bias.bitwidth
        with self.assertRaisesRegex(ValueError, '[Ss]hared bias.*b'):
            align_onnx_bias_scale(sim)
        self.assertFalse(bias.enabled)
        self.assertEqual(bias.bitwidth, old_bitwidth)
        self.assertEqual(sim._rx_calibration_bias_quantizers, {'b'})

    def test_compatible_shared_bias_scales_are_accepted(self):
        sim = self.sim(same_inputs=True)
        align_onnx_bias_scale(sim)
        self.assertTrue(sim.qc_quantize_op_dict['b'].enabled)

    def test_explicitly_disabled_shared_bias_is_preserved(self):
        sim = self.sim()
        sim._rx_calibration_bias_quantizers.clear()
        align_onnx_bias_scale(sim)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)


if __name__ == '__main__':
    unittest.main()
