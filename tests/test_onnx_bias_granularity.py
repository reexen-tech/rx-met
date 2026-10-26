"""Accumulator alignment preserves explicit bias granularity, even at one channel."""

import json
import unittest

import numpy as np
from onnx import helper, numpy_helper

from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
import test_onnx_ptq_alignment as fixtures


class OnnxBiasGranularityTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

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


if __name__ == '__main__':
    unittest.main()
