"""Explicit QAT gradients at quantized Log/Sqrt singularities."""
import unittest

import torch

from aimet_torch._base.nn.modules import custom
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize


def make_module(cls, device='cpu'):
    module = QuantizationMixin.from_module(cls()).to(device)
    for index in range(len(module.input_quantizers)):
        q = QuantizeDequantize((), 8, True).to(device)
        q.set_range(torch.tensor(-1.28, device=device), torch.tensor(1.27, device=device))
        module.input_quantizers[index] = q
    q = QuantizeDequantize((), 8, True).to(device)
    q.set_range(torch.tensor(-32., device=device), torch.tensor(31.75, device=device))
    module.output_quantizers[0] = q
    return module.requires_grad_(False)


class QuantizedSingularGradientsTest(unittest.TestCase):
    def test_sqrt_zero_code_has_finite_zero_qat_gradient(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for upstream in [0., 1., -1.]:
                with self.subTest(device=device, upstream=upstream):
                    module = make_module(custom.Sqrt, device)
                    x = torch.tensor([0., .001, .004, .01, .1], device=device, requires_grad=True)
                    effective = module.input_quantizers[0](x)
                    expected = module.output_quantizers[0](torch.sqrt(effective))
                    actual = module(x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual.backward(torch.full_like(actual, upstream))
                    self.assertTrue(torch.isfinite(x.grad).all())
                    torch.testing.assert_close(x.grad[effective == 0],
                                               torch.zeros_like(x.grad[effective == 0]))

    def test_zero_spectral_power_does_not_contaminate_square_inputs(self):
        square = make_module(custom.Multiply)
        sqrt = make_module(custom.Sqrt)
        x = torch.tensor([.001, .004, .1], requires_grad=True)
        sqrt(square(x, x)).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.testing.assert_close(x.grad[:2], torch.zeros_like(x[:2]))

    def test_positive_domain_matches_qdq_gradients(self):
        for cls, fn in [(custom.Sqrt, torch.sqrt), (custom.Log, torch.log)]:
            module = make_module(cls)
            a = torch.tensor([.1, .2, .3], requires_grad=True)
            b = a.detach().clone().requires_grad_()
            actual = module(a)
            expected = module.output_quantizers[0](fn(module.input_quantizers[0](b)))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-6)

    def test_unquantized_sqrt_active_singularity_is_preserved(self):
        module = make_module(custom.Sqrt)
        module.input_quantizers[0] = None
        x = torch.tensor([0.], requires_grad=True)
        module(x).sum().backward()
        self.assertTrue(torch.isinf(x.grad).all())

    def test_output_mask_or_zero_log_cotangent_does_not_create_nan(self):
        for cls in [custom.Sqrt, custom.Log]:
            module = make_module(cls)
            if cls is custom.Log:
                module.output_quantizers[0] = None
            else:
                module.input_quantizers[0] = None
            x = torch.tensor([0.], requires_grad=True)
            actual = module(x)
            actual.backward(torch.zeros_like(actual))
            torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_negative_inputs_and_nonfinite_upstream_remain_visible(self):
        for cls, fn in [(custom.Sqrt, torch.sqrt), (custom.Log, torch.log)]:
            module = make_module(cls)
            x = torch.tensor([-.1], requires_grad=True)
            raw = []
            handle = module.output_quantizers[0].register_forward_pre_hook(
                lambda _, args: raw.append(args[0].detach()))
            y = module(x)
            handle.remove()
            self.assertTrue(torch.isnan(raw[0]).all())
            # 冻结输出 QDQ 也必须保留非法输入产生的 NaN，不能变成有限最小码。
            self.assertTrue(torch.isnan(y).all())
            expected = module.output_quantizers[0](fn(module.input_quantizers[0](x)))
            torch.testing.assert_close(y, expected, equal_nan=True, rtol=0, atol=0)
            y.backward(torch.zeros_like(y))
            if cls is custom.Sqrt:
                self.assertTrue(torch.isnan(x.grad).all())
            module.output_quantizers[0] = None
            self.assertTrue(torch.isnan(module(x)).all())
            for upstream in [float('nan'), float('inf')]:
                x = torch.tensor([0.], requires_grad=True)
                module(x).backward(torch.tensor([upstream]))
                self.assertFalse(torch.isfinite(x.grad).any())

    def test_all_quantizers_disabled_keeps_torch_semantics(self):
        for cls, fn in [(custom.Sqrt, torch.sqrt), (custom.Log, torch.log)]:
            module = make_module(cls)
            module.input_quantizers[0] = None
            module.output_quantizers[0] = None
            a = torch.tensor([0., .1], requires_grad=True)
            b = a.detach().clone().requires_grad_()
            actual, expected = module(a), fn(b)
            torch.testing.assert_close(actual, expected, equal_nan=True)
            actual.backward(torch.zeros_like(actual))
            expected.backward(torch.zeros_like(expected))
            torch.testing.assert_close(a.grad, b.grad, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
