"""Regressions for direct ONNX PTQ configuration, bias safety and artifact reload."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from aimet_common.power_of_2 import compute_aligned_bias_range, validate_bias_values
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.rx_ptq.po2 import (
    align_onnx_bias_scale, apply_power_of_2_workflow, disable_onnx_bias_quantizers,
)


def make_graph(*, bias=True, second_layer=False, matmul_add=False):
    weights = np.array([[.01, -.02, .01, -.02], [-.02, .01, -.02, .01]], dtype=np.float32)
    initializers = [numpy_helper.from_array(weights.T if matmul_add or not bias else weights, 'w')]
    nodes = []
    output = 'mid' if second_layer else 'y'
    if bias:
        initializers.append(numpy_helper.from_array(np.array([.5, -.25], dtype=np.float32), 'b'))
    if bias and not matmul_add:
        nodes.append(helper.make_node('Gemm', ['x', 'w', 'b'], [output], name='fc', transB=1))
    else:
        nodes.append(helper.make_node('MatMul', ['x', 'w'], ['product' if bias else output], name='fc'))
        if bias:
            nodes.append(helper.make_node('Add', ['product', 'b'], [output], name='add'))
    if second_layer:
        initializers.extend([
            numpy_helper.from_array(np.array([[.1, -.2], [-.2, .1]], dtype=np.float32), 'w2'),
            numpy_helper.from_array(np.array([.1, -.1], dtype=np.float32), 'b2'),
        ])
        nodes.append(helper.make_node('Gemm', ['mid', 'w2', 'b2'], ['y'], name='fc2', transB=1))
    graph = helper.make_graph(nodes, 'alignment',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)


class OnnxAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='onnx-ptq-alignment-')
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.config = self.path / 'config.json'
        self.config.write_text(json.dumps({
            'defaults': {'ops': {'is_output_quantized': 'True', 'is_symmetric': 'True'},
                         'params': {'is_quantized': 'True', 'is_symmetric': 'True'},
                         'per_channel_quantization': 'True', 'strict_symmetric': 'False',
                         'unsigned_symmetric': 'False'},
            'params': {}, 'op_type': {}, 'supergroups': [], 'model_input': {}, 'model_output': {},
        }))
        self.feed = {'x': np.array([[.1, -.1, .1, -.1]], dtype=np.float32)}
        self.index = 0

    def sim(self, **kwargs):
        self.index += 1
        model = make_graph(**kwargs)
        sim = pipeline.create_onnx_ptq_sim(model, self.feed, config_file=str(self.config),
            quant_scheme='tf', work_dir=self.path / f'sim{self.index}')
        return sim, model

    def calibrate(self, sim, *, bias=True):
        if bias:
            disable_onnx_bias_quantizers(sim)
        sim.compute_encodings([self.feed, {'x': self.feed['x'] + 1.0}])

    def apply_config(self, sim, config):
        path = self.path / 'mixed.json'
        path.write_text(json.dumps(config))
        apply_onnx_mixed_precision_bitwidth(sim, str(path), verbose=False)

    def test_input_settings_do_not_overwrite_parameters(self):
        for matmul_add in (False, True):
            with self.subTest(matmul_add=matmul_add):
                sim, _ = self.sim(matmul_add=matmul_add)
                self.apply_config(sim, {'layer_type_config': {'QuantizedLinear': {
                    'weight_bitwidth': 16, 'weight_symmetric': True,
                    'bias_bitwidth': 32, 'bias_symmetric': True,
                    'input_bitwidth': 8, 'input_symmetric': False,
                }}})
                quantizers = sim.qc_quantize_op_dict
                for name, bits, symmetric in [('x', 8, False), ('w', 16, True), ('b', 32, True)]:
                    self.assertEqual(quantizers[name].bitwidth, bits, name)
                    self.assertEqual(quantizers[name].use_symmetric_encodings, symmetric, name)

    def test_missing_activation_quantizer_preserves_input_slot_indices(self):
        sim, _ = self.sim(bias=False)
        op = SimpleNamespace(name='concat', type='Concat', parameters={},
            inputs=[SimpleNamespace(name=n, is_parm=False) for n in ['missing', 'x']], outputs=[])
        sim.connected_graph.get_all_ops = lambda: {'concat': op}
        with patch.object(sim, '_get_weight_and_bias', return_value=(None, None)):
            self.apply_config(sim, {'layer_name_config': {'concat': {'input_bitwidths': [4, 16]}}})
        self.assertEqual(sim.qc_quantize_op_dict['x'].bitwidth, 16)

    def test_disabling_layer_still_disables_its_parameters(self):
        sim, _ = self.sim()
        self.apply_config(sim, {'layer_name_config': {'fc': {'disable_quantization': True}}})
        self.assertTrue(all(not q.enabled for q in sim.qc_quantize_op_dict.values()))

    def test_bias32_uses_accumulator_scale_and_signed_symmetric_grid(self):
        for matmul_add in (False, True):
            with self.subTest(matmul_add=matmul_add):
                sim, _ = self.sim(matmul_add=matmul_add)
                self.calibrate(sim)
                report = align_onnx_bias_scale(sim, bias_bitwidth=32)
                self.assertEqual(report['modified'], 1)
                q = sim.qc_quantize_op_dict['b']
                self.assertTrue(q.enabled)
                self.assertTrue(q.use_symmetric_encodings)
                self.assertFalse(q.use_unsigned_symmetric)
                self.assertFalse(q.use_strict_symmetric)
                sx = sim.qc_quantize_op_dict['x'].get_encodings()[0].delta
                sw = sim.qc_quantize_op_dict['w'].get_encodings()
                for bias_encoding, weight_encoding in zip(q.get_encodings(), sw):
                    self.assertEqual(bias_encoding.bw, 32)
                    self.assertEqual(bias_encoding.delta, sx * weight_encoding.delta)
                    self.assertEqual(bias_encoding.offset, -(2 ** 31))
                self.assertFalse(sim._rx_calibration_bias_quantizers)

    def test_explicitly_disabled_bias_stays_disabled(self):
        sim, _ = self.sim(second_layer=True)
        sim.qc_quantize_op_dict['b'].enabled = False
        self.calibrate(sim)
        self.assertEqual(sim._rx_calibration_bias_quantizers, {'b2'})
        align_onnx_bias_scale(sim)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.assertTrue(sim.qc_quantize_op_dict['b2'].enabled)

    def test_json1_disabled_bias_is_preserved_through_export(self):
        config = json.loads(self.config.read_text())
        config['params']['bias'] = {'is_quantized': 'False'}
        self.config.write_text(json.dumps(config))
        sim, model = self.sim()
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.calibrate(sim)
        apply_power_of_2_workflow(sim, verbose=False)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / 'explicit-disable', 'model')
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.assertNotIn('b', artifacts['coverage']['enabled_quantizer_names'])
        self.assertNotIn('b', json.loads(artifacts['encodings'].read_text())['param_encodings'])

    def test_bias16_overflow_is_rejected_without_reenabling_bias(self):
        sim, _ = self.sim()
        self.calibrate(sim)
        with self.assertRaisesRegex(ValueError, 'bias is not representable'):
            align_onnx_bias_scale(sim, bias_bitwidth=16)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)

    def test_later_bias32_overflow_does_not_partially_update_earlier_bias(self):
        sim, _ = self.sim(second_layer=True)
        self.calibrate(sim)
        bias2 = sim.model.get_initializer('b2')
        bias2.CopyFrom(numpy_helper.from_array(np.full((2,), 1e12, dtype=np.float32), 'b2'))
        q = sim.qc_quantize_op_dict['b']
        old_width = q.bitwidth
        with self.assertRaisesRegex(ValueError, 'b2: bias is not representable'):
            align_onnx_bias_scale(sim, bias_bitwidth=32)
        self.assertEqual(q.bitwidth, old_width)
        self.assertFalse(q.enabled)
        self.assertIsNone(q.get_encodings())

    def test_invalid_bias_bitwidth_is_rejected(self):
        sim, _ = self.sim()
        for bitwidth in (1, 33, 16.0, True):
            with self.subTest(bitwidth=bitwidth), self.assertRaisesRegex(ValueError, 'bias_bitwidth'):
                align_onnx_bias_scale(sim, bias_bitwidth=bitwidth)

    def test_nonfinite_bias_input_scale_is_rejected(self):
        sim, _ = self.sim()
        self.calibrate(sim)
        q = sim.qc_quantize_op_dict['x']
        encodings = q.get_encodings()
        encodings[0].delta = float('nan')
        q.load_encodings(encodings)
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            align_onnx_bias_scale(sim)

    def test_invalid_scales_fail_before_po2_mutation_and_export(self):
        for scale in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(scale=scale):
                sim, model = self.sim(bias=False)
                self.calibrate(sim, bias=False)
                old_input_scale = sim.qc_quantize_op_dict['x'].get_encodings()[0].delta
                q = sim.qc_quantize_op_dict['w']
                encodings = q.get_encodings()
                encodings[0].delta = scale
                q.load_encodings(encodings)
                with self.assertRaisesRegex(ValueError, 'w.*finite and positive'):
                    apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
                self.assertTrue(q.enabled)
                self.assertEqual(sim.qc_quantize_op_dict['x'].get_encodings()[0].delta, old_input_scale)
                output = self.path / 'invalid-export'
                with self.assertRaisesRegex(ValueError, 'w.*finite and positive'):
                    pipeline.export_onnx_json(sim, model, output, 'model')
                self.assertFalse(list(output.glob('*')))

    def test_uninitialized_quantizer_is_rejected(self):
        sim, _ = self.sim(bias=False)
        with self.assertRaisesRegex(RuntimeError, 'no encodings'):
            apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        self.assertTrue(all(q.enabled for q in sim.qc_quantize_op_dict.values()))

    def test_required_weight_cannot_be_disabled_after_po2(self):
        sim, model = self.sim(bias=False)
        self.calibrate(sim, bias=False)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        sim.qc_quantize_op_dict['w'].enabled = False
        with self.assertRaisesRegex(RuntimeError, 'Required quantizers.*w'):
            pipeline.export_onnx_json(sim, model, self.path / 'disabled', 'model')

    def test_temporarily_disabled_bias_cannot_be_exported_without_alignment(self):
        sim, model = self.sim()
        self.calibrate(sim)
        with self.assertRaisesRegex(RuntimeError, 'await scale alignment'):
            pipeline.export_onnx_json(sim, model, self.path / 'pending', 'model')

    def artifacts(self, *, matmul_add=False, mixed=False):
        sim, model = self.sim(matmul_add=matmul_add)
        if mixed:
            self.apply_config(sim, {'layer_type_config': {'QuantizedLinear': {
                'weight_bitwidth': 16, 'input_bitwidth': 8, 'output_bitwidth': 16,
            }}})
        self.calibrate(sim)
        apply_power_of_2_workflow(sim, verbose=False)
        expected = pipeline.run_onnx_feed(sim, self.feed)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / f'export{self.index}', 'model')
        return artifacts, expected

    def verify(self, artifacts, expected):
        return pipeline.verify_onnx_ptq_artifacts(artifacts, self.feed, expected,
            config_file=str(self.config), quant_scheme='tf')

    def test_artifacts_restore_quantized_outputs_and_compiler_grids(self):
        for matmul_add, mixed in [(False, False), (True, False), (False, True)]:
            with self.subTest(matmul_add=matmul_add, mixed=mixed):
                artifacts, expected = self.artifacts(matmul_add=matmul_add, mixed=mixed)
                report = self.verify(artifacts, expected)
                self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_pipeline_records_actual_quantized_reload_validation(self):
        model_path = self.path / 'input.onnx'
        onnx.save(make_graph(), model_path)
        calibration_dir = self.path / 'calib'
        calibration_dir.mkdir()
        np.save(calibration_dir / 'sample0.npy', self.feed['x'])
        np.save(calibration_dir / 'sample1.npy', self.feed['x'] + 1.0)
        for align_bias in (False, True):
            with self.subTest(align_bias=align_bias):
                result = pipeline.run_onnx_ptq(model_path=model_path, calibration_dir=calibration_dir,
                    output_dir=self.path / f'pipeline-{align_bias}', filename_prefix='model',
                    config_file=str(self.config), bitwidth_config_file='', quant_scheme='tf',
                    align_bias_scale=align_bias)
                metadata = json.loads(result['metadata'].read_text())
                self.assertTrue(all(item['max_abs'] == 0 for item in
                    metadata['restored_quantized_vs_pre_export']))
                self.assertIn('b', metadata['coverage']['enabled_quantizer_names'])

    def test_percentile_ptq_keeps_configured_bias_quantization(self):
        model = make_graph()
        model.graph.initializer[-1].CopyFrom(
            numpy_helper.from_array(np.array([0., .125], dtype=np.float32), 'b'))
        model_path = self.path / 'percentile-input.onnx'
        onnx.save(model, model_path)
        calibration_dir = self.path / 'percentile-calib'
        calibration_dir.mkdir()
        np.save(calibration_dir / 'sample.npy', self.feed['x'])
        mixed = self.path / 'percentile-mixed.json'
        mixed.write_text(json.dumps({'layer_type_config': {'QuantizedLinear': {
            'weight_bitwidth': 8, 'bias_bitwidth': 16,
            'input_bitwidth': 8, 'output_bitwidth': 8}}}))
        observed = []
        evaluate = pipeline.evaluate_onnx_dataset
        def capture(source, *args, **kwargs):
            if hasattr(source, 'qc_quantize_op_dict'):
                bias = source.qc_quantize_op_dict['b']
                observed.append((bias.enabled, bias.bitwidth))
                if bias.enabled:
                    self.assertTrue(all(np.isfinite(e.delta) and e.delta > 0 for e in bias.get_encodings()))
            return evaluate(source, *args, **kwargs)
        with patch.object(pipeline, 'evaluate_onnx_dataset', side_effect=capture):
            pipeline.run_onnx_ptq(model_path=model_path, calibration_dir=calibration_dir,
                output_dir=self.path / 'percentile-run', filename_prefix='model',
                config_file=str(self.config), bitwidth_config_file=str(mixed),
                quant_scheme='percentile')
        self.assertEqual(observed[:2], [(True, 16), (True, 32)])

    def test_missing_native_encoding_is_rejected(self):
        artifacts, expected = self.artifacts()
        path = artifacts['aimet_encodings']
        raw = json.loads(path.read_text())
        raw['param_encodings'] = [item for item in raw['param_encodings'] if item['name'] != 'w']
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, 'Required quantizers.*w'):
            self.verify(artifacts, expected)

    def test_modified_native_scale_is_rejected(self):
        artifacts, expected = self.artifacts()
        path = artifacts['aimet_encodings']
        raw = json.loads(path.read_text())
        raw['param_encodings'][0]['scale'][0] *= 2
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            self.verify(artifacts, expected)

    def test_missing_compiler_weight_is_rejected(self):
        artifacts, expected = self.artifacts()
        path = artifacts['encodings']
        payload = json.loads(path.read_text())
        del payload['param_encodings']['w']
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            self.verify(artifacts, expected)

    def test_changed_quantized_output_is_rejected(self):
        artifacts, expected = self.artifacts()
        with self.assertRaisesRegex(RuntimeError, 'quantized output'):
            self.verify(artifacts, [value + 1 for value in expected])

    def test_nonfinite_outputs_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for reference, actual in [(0., value), (value, 0.)]:
                with self.subTest(reference=reference, actual=actual):
                    with self.assertRaisesRegex(ValueError, 'NaN.*Inf'):
                        pipeline.compare_outputs([np.array([reference])], [np.array([actual])])


class CommonBiasValidationTest(unittest.TestCase):
    def test_zero_and_boundary_values_are_representable(self):
        validate_bias_values([0, -32768, 32767], [1], 16)

    def test_invalid_scale_and_overflow_are_rejected(self):
        for scale in (0, -1, np.nan, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                compute_aligned_bias_range(scale, 1, 32)
        for bias in (32768, -32769, np.nan, np.inf, 1e12):
            with self.subTest(bias=bias), self.assertRaisesRegex(ValueError, 'representable'):
                validate_bias_values([bias], [1], 16)

    def test_incompatible_channel_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'output-channel'):
            validate_bias_values([1, 2], [1, 2, 3], 32)


if __name__ == '__main__':
    unittest.main()
