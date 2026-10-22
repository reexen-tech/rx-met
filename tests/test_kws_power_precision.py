"""Regression for small spectral components in the KWS power chain."""
import json
import tempfile
import unittest

import torch

from examples import quick_start_kws as kws
from aimet_torch.v2.quantization.affine import AffineQuantizerBase


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class KwsPowerPrecisionTest(unittest.TestCase):
    def test_small_power_survives_existing_qdq_boundaries(self):
        torch.manual_seed(42)
        model = kws.AttMHRNN(rnn_type='gru', rnn_units=4, heads=1,
                             dense_units=(4,), dropout=0).cuda().eval()
        prepared = kws.model_preparer.prepare_model(model)
        sim = kws.quantsim.QuantizationSimModel(
            prepared, torch.zeros(2, kws.TARGET_LEN, device='cuda'),
            config_file=str(kws.CONFIG_FILE), default_output_bw=8, default_param_bw=8)
        baseline_config = json.loads(kws.BITWIDTH_CONFIG_FILE.read_text())
        baseline_config['layer_name_config'] = {}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as config:
            json.dump(baseline_config, config)
            config.flush()
            kws.apply_mixed_precision_bitwidth(sim.model, config_file=config.name, verbose=False)
        before = {n: q.bitwidth for n, q in sim.model.named_modules()
                  if isinstance(q, AffineQuantizerBase)}
        graph = sim.model.code
        kws.apply_mixed_precision_bitwidth(
            sim.model, config_file=str(kws.BITWIDTH_CONFIG_FILE), verbose=False)
        after = {n: q.bitwidth for n, q in sim.model.named_modules()
                 if isinstance(q, AffineQuantizerBase)}
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual(graph, sim.model.code)
        for name, width in before.items():
            if not name.startswith(('speech_features.windowing.',
                                    'speech_features.mag_rdft_mel.',
                                    'module_clamp.', 'module_log.input_quantizers.',
                                    'convs.0.conv.output_quantizers.',
                                    'convs.0.act.input_quantizers.')):
                self.assertEqual(after[name], width, name)

        prefix = 'speech_features.mag_rdft_mel.'
        real_square = sim.model.get_submodule(prefix + 'module_mul_1')
        imag_square = sim.model.get_submodule(prefix + 'module_mul_2')
        add = sim.model.get_submodule(prefix + 'module_add')
        sqrt = sim.model.get_submodule(prefix + 'module_sqrt')
        # Ranges from the failure: ordinary spectral values are small relative
        # to the shared square/sum range. Exercise the actual operator chain.
        with torch.no_grad():
            for module in (real_square, imag_square):
                for q in module.input_quantizers:
                    if q is not None:
                        q.set_range(torch.tensor(-40., device='cuda'), torch.tensor(40., device='cuda'))
                module.output_quantizers[0].set_range(
                    torch.tensor(-1145., device='cuda'), torch.tensor(1145., device='cuda'))
            for q in add.input_quantizers:
                q.set_range(torch.tensor(-1145., device='cuda'), torch.tensor(1145., device='cuda'))
            add.output_quantizers[0].set_range(
                torch.tensor(-2048., device='cuda'), torch.tensor(2048., device='cuda'))
            sqrt.input_quantizers[0].set_range(
                torch.tensor(-2048., device='cuda'), torch.tensor(2048., device='cuda'))
            sqrt.output_quantizers[0].set_range(
                torch.tensor(-64., device='cuda'), torch.tensor(64., device='cuda'))
            sim.model.requires_grad_(False)
            real = torch.tensor([.01, 30.], device='cuda')
            imag = torch.tensor([.01, 20.], device='cuda')
            power = add(real_square(real, real), imag_square(imag, imag))
            self.assertGreater(float(power[0]), 0.)
            actual = sqrt(power)
            expected = torch.sqrt(real.square() + imag.square())
            torch.testing.assert_close(actual, expected, rtol=.05, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
