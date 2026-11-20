"""ONNX export and strict restored-output validation."""

import json
import unittest
import numpy as np
import onnx
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
from aimet_onnx.rx_ptq.input_quantization import input_quantizer

from tests.support import onnx as fixtures
from tests.support.onnx import make_graph


class OnnxArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.path = self.fixture.path
        self.config = self.fixture.config
        self.feed = self.fixture.feed

    def test_required_weight_cannot_be_disabled_after_po2(self):
        sim, model = self.fixture.sim(bias=False)
        self.fixture.calibrate(sim, bias=False)
        apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        sim.qc_quantize_op_dict['w'].enabled = False
        with self.assertRaisesRegex(RuntimeError, 'Required quantizers.*w'):
            pipeline.export_onnx_json(sim, model, self.path / 'disabled', 'model')

    def test_artifacts_restore_quantized_outputs_and_compiler_grids(self):
        for matmul_add, mixed in [(False, False), (True, False), (False, True)]:
            with self.subTest(matmul_add=matmul_add, mixed=mixed):
                artifacts, expected = self.fixture.artifacts(matmul_add=matmul_add, mixed=mixed)
                report = self.fixture.verify(artifacts, expected)
                self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_matmul_input_precision_survives_export_and_reload(self):
        fixture = fixtures.OnnxGraphFixture(self)
        for normalized_linear in (False, True):
            with self.subTest(normalized_linear=normalized_linear):
                nodes = [
                    onnx.helper.make_node('Relu', ['x'], ['before'], name='producer'),
                    onnx.helper.make_node('MatMul', ['before', 'w'],
                        ['product' if normalized_linear else 'y'], name='matmul'),
                ]
                constants = {'w': np.array([[.02, -.01], [-.03, .02],
                                           [.01, -.02], [-.01, .03]], dtype=np.float32)}
                if normalized_linear:
                    constants['b'] = np.array([.01, -.02], dtype=np.float32)
                    nodes.append(onnx.helper.make_node('Add', ['product', 'b'], ['y'], name='add'))
                sim, model, feed = fixture.model(nodes, {'x': [1, 4]}, {'y': [1, 2]}, constants)
                fixture.apply_config(sim, {
                    'default_bitwidth': {'weight': 16, 'output': 16},
                    'layer_name_config': {
                        'producer': {'output_bitwidth': 16},
                        'matmul': {'input_bitwidths': [8, 8], 'output_bitwidth': 16},
                    },
                })
                sim.compute_encodings([feed, {'x': feed['x'] * 1.5}])
                apply_power_of_2_workflow(sim, align_bias_scale=normalized_linear, verbose=False)
                op = sim.connected_graph.get_all_ops()['matmul']
                self.assertEqual(input_quantizer(sim, op, 0).bitwidth, 8)
                expected = sim.session.run(None, feed)
                artifacts = pipeline.export_onnx_json(sim, model,
                    fixture.path / f'export-{normalized_linear}', 'model')
                payload = json.loads(artifacts['encodings'].read_text())
                activations = payload['activation_encodings']
                self.assertEqual(activations['producer']['output']['0']['bitwidth'], 16)
                self.assertEqual(activations['matmul']['input']['0']['bitwidth'], 8)
                self.assertEqual(payload['param_encodings']['w']['bitwidth'], 8)
                report = pipeline.verify_onnx_ptq_artifacts(artifacts, feed, expected,
                    config_file=str(fixture.config), quant_scheme='tf')
                self.assertTrue(all(item['max_abs'] == 0 for item in report))

    def test_pipeline_records_actual_quantized_reload_validation(self):
        model_path = self.path / 'input.onnx'
        onnx.save(make_graph(), model_path)
        calibration_dir = self.path / 'calib'
        calibration_dir.mkdir()
        np.save(calibration_dir / 'sample0.npy', self.feed['x'])
        np.save(calibration_dir / 'sample1.npy', self.feed['x'] + 1.0)
        for enable_po2 in (False, True):
            with self.subTest(enable_po2=enable_po2):
                result = pipeline.run_onnx_ptq(model_path=model_path, calibration_dir=calibration_dir,
                    output_dir=self.path / f'pipeline-{enable_po2}', filename_prefix='model',
                    config_file=str(self.config), bitwidth_config_file='', quant_scheme='tf',
                    enable_po2=enable_po2)
                metadata = json.loads(result['metadata'].read_text())
                self.assertTrue(all(item['max_abs'] == 0 for item in
                    metadata['restored_quantized_vs_pre_export']))
                self.assertIn('b', metadata['coverage']['enabled_quantizer_names'])
                self.assertEqual(metadata['bias_align']['modified'], 1)
                self.assertEqual('po2' in metadata['evaluation'], enable_po2)
                self.assertEqual('all_power_of_2' in metadata, enable_po2)
                if enable_po2:
                    self.assertEqual(metadata['po2_bias_align']['modified'], 1)
                    self.assertTrue(metadata['all_power_of_2'])
                payload = json.loads(result['encodings'].read_text())
                bias, weight = (payload['param_encodings'][name] for name in ('b', 'w'))
                self.assertEqual(bias['bitwidth'], 32)
                sx = payload['activation_encodings']['fc']['input']['0']['scale']
                np.testing.assert_array_equal(bias['scale'], np.asarray(weight['scale']) * sx)

    def test_pipeline_requires_bias_alignment_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, 'requires align_bias_scale=True'):
            pipeline.run_onnx_ptq(model_path=self.path / 'missing.onnx',
                calibration_dir=self.path / 'missing-calib', output_dir=self.path / 'rejected',
                filename_prefix='model', align_bias_scale=False, enable_po2=False)

    def test_pipeline_preserves_explicitly_disabled_bias_with_or_without_po2(self):
        config = json.loads(self.config.read_text())
        config['params']['bias'] = {'is_quantized': 'False'}
        self.config.write_text(json.dumps(config))
        model_path = self.path / 'disabled-bias.onnx'
        onnx.save(make_graph(), model_path)
        calibration_dir = self.path / 'disabled-calib'
        calibration_dir.mkdir()
        np.save(calibration_dir / 'sample.npy', self.feed['x'])
        for enable_po2 in (False, True):
            with self.subTest(enable_po2=enable_po2):
                result = pipeline.run_onnx_ptq(model_path=model_path, calibration_dir=calibration_dir,
                    output_dir=self.path / f'disabled-bias-{enable_po2}', filename_prefix='model',
                    config_file=str(self.config), bitwidth_config_file='', quant_scheme='tf',
                    enable_po2=enable_po2)
                payload = json.loads(result['encodings'].read_text())
                self.assertNotIn('b', payload['param_encodings'])
                metadata = json.loads(result['metadata'].read_text())
                self.assertEqual(metadata['bias_align']['modified'], 0)
                self.assertNotIn('b', metadata['coverage']['enabled_quantizer_names'])
                self.assertTrue(all(item['max_abs'] == 0 for item in
                    metadata['restored_quantized_vs_pre_export']))

    def test_missing_native_encoding_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        path = artifacts['aimet_encodings']
        raw = json.loads(path.read_text())
        raw['param_encodings'] = [item for item in raw['param_encodings'] if item['name'] != 'w']
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, 'Required quantizers.*w'):
            self.fixture.verify(artifacts, expected)

    def test_modified_native_scale_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        path = artifacts['aimet_encodings']
        raw = json.loads(path.read_text())
        raw['param_encodings'][0]['scale'][0] *= 2
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            self.fixture.verify(artifacts, expected)

    def test_missing_compiler_weight_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        path = artifacts['encodings']
        payload = json.loads(path.read_text())
        del payload['param_encodings']['w']
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, 'compiler encodings'):
            self.fixture.verify(artifacts, expected)

    def test_changed_quantized_output_is_rejected(self):
        artifacts, expected = self.fixture.artifacts()
        with self.assertRaisesRegex(RuntimeError, 'quantized output'):
            self.fixture.verify(artifacts, [value + 1 for value in expected])

    def test_nonfinite_outputs_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            for reference, actual in [(0., value), (value, 0.)]:
                with self.subTest(reference=reference, actual=actual):
                    with self.assertRaisesRegex(ValueError, 'NaN.*Inf'):
                        pipeline.compare_outputs([np.array([reference])], [np.array([actual])])


if __name__ == "__main__":
    unittest.main()
