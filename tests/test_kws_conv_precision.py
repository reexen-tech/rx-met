"""Small positive values must survive the first KWS Conv/ReLU edge."""
import unittest

import torch
from torch import nn

from examples import quick_start_kws as kws
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize


class KwsConvPrecisionTest(unittest.TestCase):
    def test_conv_relu_intermediate_grid(self):
        def make():
            block = nn.Module()
            block.conv = QuantizationMixin.from_module(nn.Conv2d(1, 1, 1))
            block.act = QuantizationMixin.from_module(nn.ReLU())
            for module in (block.conv, block.act):
                module.input_quantizers[0] = QuantizeDequantize((), 8, True)
                module.output_quantizers[0] = QuantizeDequantize((), 8, True)
            block.conv.param_quantizers['weight'] = QuantizeDequantize((), 8, True)
            block.conv.param_quantizers['bias'] = QuantizeDequantize((), 16, True)
            block.conv.weight.data.fill_(1)
            block.conv.bias.data.zero_()
            model = nn.Module()
            model.convs = nn.ModuleList([block])
            return model, block

        model, block = make()
        kws.apply_mixed_precision_bitwidth(
            model, config_file=str(kws.BITWIDTH_CONFIG_FILE), verbose=False)
        self.assertEqual(block.conv.param_quantizers['weight'].bitwidth, 8)
        self.assertEqual(block.conv.param_quantizers['bias'].bitwidth, 16)
        self.assertEqual(block.conv.output_quantizers[0].bitwidth, 16)
        self.assertEqual(block.act.input_quantizers[0].bitwidth, 16)
        self.assertEqual(block.act.output_quantizers[0].bitwidth, 8)
        block.conv.input_quantizers[0] = None
        block.conv.param_quantizers['weight'] = None
        block.conv.param_quantizers['bias'] = None
        for q in (block.conv.output_quantizers[0], block.act.input_quantizers[0]):
            q.set_range(torch.tensor(-512.), torch.tensor(508.))
        block.act.output_quantizers[0].set_range(torch.tensor(-256.), torch.tensor(254.))
        model.requires_grad_(False)
        x = torch.tensor([[[[2.]]]])
        actual = block.act(block.conv(x))
        torch.testing.assert_close(actual, x, rtol=0, atol=0)
        # The old 8-bit Po2 intermediate rounds 2 / 4 to the even code 0.
        for q in (block.conv.output_quantizers[0], block.act.input_quantizers[0]):
            q.bitwidth = 8
            q.set_range(torch.tensor(-512.), torch.tensor(508.))
        torch.testing.assert_close(block.act(block.conv(x)), torch.zeros_like(x))


if __name__ == '__main__':
    unittest.main()
