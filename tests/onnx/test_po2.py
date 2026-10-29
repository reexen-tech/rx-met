"""ONNX power-of-two grid alignment and failure atomicity."""

import json
import unittest
from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
from aimet_common.power_of_2 import recompute_min_max_for_new_scale
from aimet_onnx.rx_ptq.po2 import align_encoding_values

from tests.support import onnx as fixtures


class OnnxPowerOfTwoTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OnnxFixture(self)
        self.path = self.fixture.path
        self.config = self.fixture.config
        self.feed = self.fixture.feed

    def test_invalid_scales_fail_before_po2_mutation_and_export(self):
        for scale in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(scale=scale):
                sim, model = self.fixture.sim(bias=False)
                self.fixture.calibrate(sim, bias=False)
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
                self.assertFalse(list(output.glob('*/exports/*')))
                records = [json.loads(path.read_text()) for path in output.glob('*/run.json')]
                self.assertTrue(records)
                self.assertTrue(all(record['status'] == 'failed' for record in records))

    def test_uninitialized_quantizer_is_rejected(self):
        sim, _ = self.fixture.sim(bias=False)
        with self.assertRaisesRegex(RuntimeError, 'no encodings'):
            apply_power_of_2_workflow(sim, align_bias_scale=False, verbose=False)
        self.assertTrue(all(q.enabled for q in sim.qc_quantize_op_dict.values()))

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
        for signed in (True, False):
            with self.subTest(signed=signed):
                quantizer = QuantizeDequantize((2, 1), bitwidth=8, symmetric=False, signed=signed)
                quantizer.set_range(torch.tensor([[-.13], [-.07]]), torch.tensor([[2.42], [1.2]]))
                before = [value.detach().flatten().tolist() for value in
                    (quantizer.get_scale(), quantizer.get_offset(), quantizer.get_min(), quantizer.get_max())]
                expected = [align_encoding_values(scale=s, offset=int(o)+quantizer.qmin,
                    real_min=low, real_max=high, bitwidth=8, symmetric=False)
                    for s, o, low, high in zip(*before)]
                self.assertTrue(modify_quantizer_to_power_of_2(quantizer, 'test', method='cover_range', verbose=False))
                for index, value in enumerate(expected):
                    self.assertEqual(quantizer.get_offset().flatten()[index].item()+quantizer.qmin, value['offset'])
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


if __name__ == "__main__":
    unittest.main()
