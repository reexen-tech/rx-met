"""Regression coverage for Po2 grids, JSON2 planning and calibration dtypes."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from aimet_common.power_of_2 import recompute_min_max_for_new_scale
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.calibration import _validate_feed, iter_calibration_feeds
from aimet_onnx.rx_ptq.po2 import align_encoding_values, apply_power_of_2_workflow
import test_onnx_ptq_alignment as fixtures


class OnnxRemainingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path

    def test_asymmetric_offset_and_range_are_one_integer_grid(self):
        values = align_encoding_values(scale=.01, offset=-13, real_min=-.13,
            real_max=2.42, bitwidth=8, symmetric=False)
        self.assertEqual(values['offset'], -8)
        self.assertEqual(values['real_min'], -.125)
        self.assertEqual(values['real_max'], 247 * .015625)

    def test_strict_and_unsigned_symmetric_grid_sizes(self):
        for offset, low, high in [(-127, -1.27, 1.27), (0, 0., 2.55)]:
            with self.subTest(offset=offset):
                values = align_encoding_values(scale=.01, offset=offset, real_min=low,
                    real_max=high, bitwidth=8, symmetric=True, strict_symmetric=True)
                limit = 127 if offset < 0 else 255
                self.assertEqual(values['real_max'], values['scale'] * limit)

    def test_torch_and_onnx_share_asymmetric_per_channel_grid(self):
        import torch
        from aimet_torch.v2.quantization.affine import QuantizeDequantize
        from aimet_torch.power_of_2_quantization import modify_quantizer_to_power_of_2
        quantizer = QuantizeDequantize((2, 1), bitwidth=8, symmetric=False)
        quantizer.set_range(torch.tensor([[-.13], [-.07]]), torch.tensor([[2.42], [1.2]]))
        before = [value.detach().flatten().tolist() for value in
                  (quantizer.get_scale(), quantizer.get_offset(), quantizer.get_min(), quantizer.get_max())]
        expected = [align_encoding_values(scale=s, offset=int(o), real_min=low, real_max=high,
                     bitwidth=8, symmetric=False) for s, o, low, high in zip(*before)]
        self.assertTrue(modify_quantizer_to_power_of_2(quantizer, 'test', method='cover_range', verbose=False))
        for index, value in enumerate(expected):
            self.assertEqual(quantizer.get_offset().flatten()[index].item(), value['offset'])
            self.assertEqual(quantizer.get_min().flatten()[index].item(), value['real_min'])
            self.assertEqual(quantizer.get_max().flatten()[index].item(), value['real_max'])

    def test_common_grid_keeps_zero_representable_when_rounding_shrinks_scale(self):
        offset, low, high, _, _ = recompute_min_max_for_new_scale(.5, 0, 255, -200., -200, False)
        self.assertEqual(offset, -255)
        self.assertEqual(high, 0.)
        self.assertEqual(low, -127.5)

    def test_strict_symmetric_encodings_survive_native_reload(self):
        config = json.loads(self.fixture.config.read_text())
        config['defaults']['strict_symmetric'] = 'True'
        self.fixture.config.write_text(json.dumps(config))
        sim, model = self.fixture.sim(bias=False)
        self.fixture.calibrate(sim, bias=False)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        expected = pipeline.run_onnx_feed(sim, self.fixture.feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / 'strict-export', 'model')
        report = self.fixture.verify(artifacts, expected)
        self.assertEqual(report[0]['max_abs'], 0)

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
        config = Path(__file__).resolve().parents[1] / 'examples/config/quick_start_full_quant.json'
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


class CalibrationDtypeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='onnx-calibration-dtypes-')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)

    def test_multi_input_npz_preserves_indices_masks_and_half_precision(self):
        ids = helper.make_tensor_value_info('ids', TensorProto.INT64, [2])
        mask = helper.make_tensor_value_info('mask', TensorProto.BOOL, [2, 2])
        values = helper.make_tensor_value_info('values', TensorProto.FLOAT16, [2, 2])
        weight = numpy_helper.from_array(np.arange(8, dtype=np.float16).reshape(4, 2), 'table')
        nodes = [helper.make_node('Gather', ['table', 'ids'], ['selected'], axis=0),
                 helper.make_node('Where', ['mask', 'values', 'selected'], ['y'])]
        graph = helper.make_graph(nodes, 'typed-inputs', [ids, mask, values],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 2])], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
        model_path = self.path / 'model.onnx'
        onnx.save(model, model_path)
        calib = self.path / 'calib'
        calib.mkdir()
        feed = {'ids': np.array([1, 3], dtype=np.int64),
                'mask': np.array([[True, False], [False, True]]),
                'values': np.full((2, 2), 10, dtype=np.float16)}
        np.savez(calib / 'sample.npz', **feed)
        _, _, feeds = pipeline.load_onnx_ptq_inputs(model_path, calib)
        for name in feed:
            self.assertEqual(feeds[0][name].dtype, feed[name].dtype)
        actual = pipeline.run_onnx_feed(model_path, feeds[0])[0]
        np.testing.assert_array_equal(actual, np.array([[10, 3], [6, 10]], dtype=np.float16))

    def test_float_indices_are_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(TypeError, 'float32.*int64'):
            _validate_feed({'ids': np.array([1.5], dtype=np.float32)}, {'ids': [1]},
                           'sample', {'ids': np.dtype('int64')})

    def test_integer_narrowing_checks_range(self):
        with self.assertRaisesRegex(ValueError, 'exceed model dtype'):
            _validate_feed({'ids': np.array([2 ** 40], dtype=np.int64)}, {'ids': [1]},
                           'sample', {'ids': np.dtype('int32')})

    def test_float_casting_checks_overflow(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            _validate_feed({'x': np.array([1e20], dtype=np.float32)}, {'x': [1]},
                           'sample', {'x': np.dtype('float16')})

    def test_float64_images_are_converted_to_declared_float32(self):
        feed = _validate_feed({'x': np.array([.25], dtype=np.float64)}, {'x': [1]},
                              'sample', {'x': np.dtype('float32')})
        self.assertEqual(feed['x'].dtype, np.float32)

    def test_shape_only_legacy_loader_preserves_dtype(self):
        np.save(self.path / 'sample.npy', np.array([2 ** 54 + 7], dtype=np.int64))
        feed = next(iter_calibration_feeds(self.path, {'ids': [1]}))
        self.assertEqual(feed['ids'].dtype, np.int64)
        self.assertEqual(feed['ids'][0], 2 ** 54 + 7)

    def test_scalar_input_remains_scalar(self):
        feed = _validate_feed({'flag': np.array(True)}, {'flag': []}, 'sample',
                              {'flag': np.dtype('bool')})
        self.assertEqual(feed['flag'].shape, ())


if __name__ == '__main__':
    unittest.main()
