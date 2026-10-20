"""Default precision parity and complete-dataset ONNX validation regressions."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import onnx

from aimet_common.evaluation import classification_accuracy, compare_output_batches
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.evaluation import evaluate_onnx_dataset, load_onnx_validation_inputs
import test_onnx_ptq_alignment as fixtures


class PtqEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path

    def test_dataset_metrics_cover_every_feed_and_weight_by_element_count(self):
        report = compare_output_batches(
            [[np.zeros(1)], [np.zeros(3)]], [[np.array([10.])], [np.zeros(3)]])
        self.assertEqual(report[0]['value_count'], 4)
        self.assertEqual(report[0]['rmse'], 5.)
        self.assertEqual(report[0]['mean_abs'], 2.5)

    def test_classification_uses_full_model_label_space(self):
        scores = np.zeros((2, 1000))
        scores[0, 217] = 10
        scores[1, 0] = 10
        report = classification_accuracy([[scores]], [217, 482])
        self.assertEqual(report['sample_count'], 2)
        self.assertEqual(report['correct_count'], 1)
        self.assertEqual(report['top1_accuracy'], .5)

    def test_nonfinite_later_batch_is_rejected(self):
        class Session:
            def run(self, _names, feed):
                return [feed['x']]
        with self.assertRaisesRegex(ValueError, 'NaN.*Inf'):
            evaluate_onnx_dataset(Session(), [{'x': np.ones((1, 2))},
                                             {'x': np.full((1, 2), np.nan)}])

    def test_path_inference_creates_one_session_for_the_entire_dataset(self):
        model_path = self.path / 'input.onnx'
        onnx.save(fixtures.make_graph(), model_path)
        import onnxruntime as ort
        original = ort.InferenceSession
        with patch.object(ort, 'InferenceSession', wraps=original) as create:
            outputs, report = evaluate_onnx_dataset(model_path,
                [self.fixture.feed, {'x': self.fixture.feed['x'] + 1}])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(report['feed_count'], 2)

    def test_manifest_labels_follow_file_names_instead_of_manifest_order(self):
        directory = self.path / 'val'
        directory.mkdir()
        np.save(directory / 'val_0000.npy', self.fixture.feed['x'])
        np.save(directory / 'val_0001.npy', self.fixture.feed['x'] + 1)
        (self.path / 'manifest.json').write_text(json.dumps({
            'val_files': ['val_0001.npy', 'val_0000.npy'], 'val_labels': [217, 0]}))
        feeds, labels = load_onnx_validation_inputs(fixtures.make_graph(), directory)
        self.assertEqual(len(feeds), 2)
        self.assertEqual(labels, [0, 217])

    def test_old_or_incomplete_manifest_is_rejected(self):
        directory = self.path / 'val'
        directory.mkdir()
        np.save(directory / 'val_0000.npy', self.fixture.feed['x'])
        for manifest in [{'val_files': ['val_0000.npy']},
                         {'val_files': ['missing.npy'], 'val_labels': [0]}]:
            (self.path / 'manifest.json').write_text(json.dumps(manifest))
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                load_onnx_validation_inputs(fixtures.make_graph(), directory)

    def test_restore_checks_later_validation_samples(self):
        artifacts, expected = self.fixture.artifacts()
        expected_second = [value + 1 for value in expected]
        with self.assertRaisesRegex(RuntimeError, 'batch 1'):
            pipeline.verify_onnx_ptq_artifacts(artifacts, self.fixture.feed, expected,
                config_file=str(self.fixture.config), quant_scheme='tf',
                validation_feeds=[self.fixture.feed, self.fixture.feed],
                expected_validation_outputs=[expected, expected_second])

    def test_pipeline_uses_separate_validation_feeds_and_reports_accuracy(self):
        model = fixtures.make_graph()
        model_path = self.path / 'input.onnx'
        onnx.save(model, model_path)
        calib, val = self.path / 'calib', self.path / 'val'
        calib.mkdir()
        val.mkdir()
        np.save(calib / 'calib_0000.npy', self.fixture.feed['x'])
        np.save(calib / 'calib_0001.npy', self.fixture.feed['x'] + 1)
        np.save(val / 'val_0000.npy', self.fixture.feed['x'] + .2)
        np.save(val / 'val_0001.npy', self.fixture.feed['x'] + .3)
        np.save(val / 'val_0002.npy', self.fixture.feed['x'] + .4)
        result = pipeline.run_onnx_ptq(model_path=model_path, calibration_dir=calib,
            output_dir=self.path / 'pipeline', filename_prefix='model',
            config_file=str(self.fixture.config), bitwidth_config_file='', quant_scheme='tf',
            validation_dir=val, validation_labels=[0, 0, 0])
        metadata = json.loads(result['metadata'].read_text())
        self.assertEqual(metadata['calibration_count'], 2)
        self.assertEqual(metadata['evaluation_source'], 'validation')
        for stage in ['fp32', 'ptq', 'po2']:
            self.assertEqual(metadata['evaluation'][stage]['feed_count'], 3)
            self.assertEqual(metadata['evaluation'][stage]['sample_count'], 3)
            self.assertIn('top1_accuracy', metadata['evaluation'][stage])
        self.assertEqual(metadata['restored_quantized_vs_pre_export'][0]['value_count'], 6)

    def torch_sim(self):
        import torch
        from torch import nn
        from aimet_torch.model_preparer import prepare_model
        from aimet_torch.v2.quantsim import QuantizationSimModel
        return QuantizationSimModel(prepare_model(nn.Sequential(nn.Linear(4, 2)).eval()),
            torch.ones(1, 4), config_file=str(self.fixture.config), quant_scheme='tf',
            default_output_bw=8, default_param_bw=8)

    def test_default_precision_matches_across_torch_and_onnx(self):
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        for defaults, weight_bits, activation_bits in [
            ({'weight': 16, 'output': 16}, 16, 16), ({'output': 16}, 8, 16),
            ({'weight': 16}, 16, 8),
        ]:
            with self.subTest(defaults=defaults):
                config = {'default_bitwidth': defaults}
                sim, _ = self.fixture.sim()
                self.fixture.apply_config(sim, config)
                torch_sim = self.torch_sim()
                config_path = self.path / 'default.json'
                config_path.write_text(json.dumps(config))
                apply_mixed_precision_bitwidth(torch_sim.model, str(config_path), verbose=False)
                layer = torch_sim.model.get_submodule('0')
                for quantizer in [sim.qc_quantize_op_dict['w'], layer.param_quantizers['weight']]:
                    self.assertEqual(quantizer.bitwidth, weight_bits)
                for quantizer in [sim.qc_quantize_op_dict['x'], sim.qc_quantize_op_dict['y'],
                                  layer.input_quantizers[0], layer.output_quantizers[0]]:
                    self.assertEqual(quantizer.bitwidth, activation_bits)

    def test_torch_defaults_do_not_create_explicitly_disabled_inputs(self):
        from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
        sim = self.torch_sim()
        layer = sim.model.get_submodule('0')
        layer.input_quantizers[0] = None
        config = self.path / 'default.json'
        config.write_text(json.dumps({'default_bitwidth': {'output': 16}}))
        apply_mixed_precision_bitwidth(sim.model, str(config), verbose=False)
        self.assertIsNone(layer.input_quantizers[0])
        self.assertEqual(layer.output_quantizers[0].bitwidth, 16)


if __name__ == '__main__':
    unittest.main()
