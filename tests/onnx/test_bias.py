"""ONNX accumulator bias safety, explicit granularity and quantizer lifecycle."""

import json
import unittest
from unittest.mock import patch
import numpy as np
import onnx
from onnx import helper, numpy_helper
from aimet_common.power_of_2 import compute_aligned_bias_range, validate_bias_values
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import align_onnx_bias_scale, apply_power_of_2_workflow

from tests.support import onnx as fixtures
from tests.support.onnx import make_graph


class OnnxBiasTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.path = self.fixture.path
        self.config = self.fixture.config
        self.feed = self.fixture.feed

    def test_bias32_uses_accumulator_scale_and_signed_symmetric_grid(self):
        for matmul_add in (False, True):
            with self.subTest(matmul_add=matmul_add):
                sim, _ = self.fixture.sim(matmul_add=matmul_add)
                self.fixture.calibrate(sim)
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
        sim, _ = self.fixture.sim(second_layer=True)
        sim.qc_quantize_op_dict['b'].enabled = False
        self.fixture.calibrate(sim)
        self.assertEqual(sim._rx_calibration_bias_quantizers, {'b2'})
        align_onnx_bias_scale(sim)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.assertTrue(sim.qc_quantize_op_dict['b2'].enabled)

    def test_json1_disabled_bias_is_preserved_through_export(self):
        config = json.loads(self.config.read_text())
        config['params']['bias'] = {'is_quantized': 'False'}
        self.config.write_text(json.dumps(config))
        sim, model = self.fixture.sim()
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.fixture.calibrate(sim)
        apply_power_of_2_workflow(sim, verbose=False)
        artifacts = pipeline.export_onnx_json(sim, model, self.path / 'explicit-disable', 'model')
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)
        self.assertNotIn('b', artifacts['coverage']['enabled_quantizer_names'])
        self.assertNotIn('b', json.loads(artifacts['encodings'].read_text())['param_encodings'])

    def test_bias16_overflow_is_rejected_without_reenabling_bias(self):
        sim, _ = self.fixture.sim()
        self.fixture.calibrate(sim)
        with self.assertRaisesRegex(ValueError, 'bias is not representable'):
            align_onnx_bias_scale(sim, bias_bitwidth=16)
        self.assertFalse(sim.qc_quantize_op_dict['b'].enabled)

    def test_later_bias32_overflow_does_not_partially_update_earlier_bias(self):
        sim, _ = self.fixture.sim(second_layer=True)
        self.fixture.calibrate(sim)
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
        sim, _ = self.fixture.sim()
        for bitwidth in (1, 33, 16.0, True):
            with self.subTest(bitwidth=bitwidth), self.assertRaisesRegex(ValueError, 'bias_bitwidth'):
                align_onnx_bias_scale(sim, bias_bitwidth=bitwidth)

    def test_nonfinite_bias_input_scale_is_rejected(self):
        sim, _ = self.fixture.sim()
        self.fixture.calibrate(sim)
        q = sim.qc_quantize_op_dict['x']
        encodings = q.get_encodings()
        encodings[0].delta = float('nan')
        q.load_encodings(encodings)
        with self.assertRaisesRegex(ValueError, 'finite and positive'):
            align_onnx_bias_scale(sim)

    def test_temporarily_disabled_bias_cannot_be_exported_without_alignment(self):
        sim, model = self.fixture.sim()
        self.fixture.calibrate(sim)
        with self.assertRaisesRegex(RuntimeError, 'await scale alignment'):
            pipeline.export_onnx_json(sim, model, self.path / 'pending', 'model')

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


class OnnxBiasGranularityTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)

    def sim(self, channels, *, weight_per_channel=True, bias_per_channel=True):
        model = fixtures.make_graph()
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(
            np.full((channels, 4), .125, dtype=np.float32), 'w'))
        model.graph.initializer[1].CopyFrom(numpy_helper.from_array(
            np.full((channels,), .125, dtype=np.float32), 'b'))
        model.graph.output[0].CopyFrom(helper.make_tensor_value_info('y', 1, [1, channels]))
        sim = pipeline.create_onnx_ptq_sim(model, self.fixture.feed,
            config_file=str(self.fixture.config), quant_scheme='tf',
            work_dir=self.fixture.path / f'sim-{channels}-{weight_per_channel}-{bias_per_channel}')
        sim.qc_quantize_op_dict['w'].enable_per_channel_quantization(weight_per_channel)
        sim.qc_quantize_op_dict['b'].enable_per_channel_quantization(bias_per_channel)
        self.fixture.calibrate(sim)
        return sim, model

    def test_single_channel_keeps_explicit_bias_mode(self):
        for mode in (True, False):
            with self.subTest(per_channel=mode):
                sim, model = self.sim(1, bias_per_channel=mode)
                apply_power_of_2_workflow(sim, verbose=False)
                self.assertEqual(sim.qc_quantize_op_dict['b'].quant_info.usePerChannelMode, mode)
                artifacts = pipeline.export_onnx_json(sim, model, self.fixture.path / f'export-{mode}', 'bias')
                record = json.loads(artifacts['encodings'].read_text())['param_encodings']['b']
                self.assertEqual(record['enc_type'], 'PER_CHANNEL' if mode else 'PER_TENSOR')
                self.assertEqual(len(record['scale']), 1)

    def test_scalar_weight_grid_expands_into_configured_bias_channels(self):
        sim, _ = self.sim(2, weight_per_channel=False)
        apply_power_of_2_workflow(sim, verbose=False)
        q = sim.qc_quantize_op_dict['b']
        self.assertTrue(q.quant_info.usePerChannelMode)
        scales = [encoding.delta for encoding in q.get_encodings()]
        self.assertEqual(len(scales), 2)
        self.assertEqual(scales[0], scales[1])

    def test_per_tensor_bias_does_not_silently_change_to_per_channel(self):
        sim, _ = self.sim(2, bias_per_channel=False)
        with self.assertRaisesRegex(ValueError, 'bias.*per-tensor|per-tensor.*bias'):
            apply_power_of_2_workflow(sim, verbose=False)
        self.assertFalse(sim.qc_quantize_op_dict['b'].quant_info.usePerChannelMode)


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


if __name__ == "__main__":
    unittest.main()
