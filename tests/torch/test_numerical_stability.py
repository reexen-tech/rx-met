"""QDQ nonfinite values and finite QAT gradients at Log/Sqrt singularities."""
import copy
import unittest

import torch

from aimet_torch._base.nn.modules import custom
from aimet_torch._base.nn.modules.custom import Log
from aimet_torch.v2.nn import QuantizationMixin, compute_encodings
from aimet_torch.v2.quantization.affine import QuantizeDequantize


class QdqNonfiniteTest(unittest.TestCase):
    def quantizer(self, device, bits, symmetric, shape=(), block_size=None):
        q = QuantizeDequantize(shape, bits, symmetric, block_size=block_size).to(device)
        q.set_range(torch.full(shape, -1., device=device), torch.full(shape, 1., device=device))
        return q.requires_grad_(False)

    def test_nan_is_preserved_for_frozen_tensor_and_channel_encodings(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for bits in [8, 16]:
                for symmetric in [False, True]:
                    for shape in [(), (2, 1)]:
                        for requires_grad in [False, True]:
                            with self.subTest(device=device, bits=bits, symmetric=symmetric,
                                              shape=shape, requires_grad=requires_grad):
                                q = self.quantizer(device, bits, symmetric, shape)
                                x = torch.tensor([[float('nan'), float('-inf'), .1],
                                                  [0., float('inf'), float('nan')]],
                                                 device=device, requires_grad=requires_grad)
                                y = q(x)
                                self.assertTrue(torch.equal(torch.isnan(y), torch.isnan(x)))
                                self.assertTrue(torch.isfinite(y[~torch.isnan(x)]).all())
                                if requires_grad:
                                    y.backward(torch.ones_like(y))
                                    torch.testing.assert_close(x.grad[torch.isnan(x)],
                                                               torch.zeros_like(x.grad[torch.isnan(x)]),
                                                               rtol=0, atol=0)

    def test_finite_outputs_and_gradients_match_original_fast_kernel(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            q = self.quantizer(device, 8, False)
            x = torch.tensor([-5., -.5, -.1, 0., .1, .5, 5.], device=device, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            y = q(x)
            scale, offset = q.get_scale(), q.get_offset()
            expected = torch.fake_quantize_per_tensor_affine(
                reference, scale.reshape(()), -offset.to(torch.int32).reshape(()), q.qmin, q.qmax)
            torch.testing.assert_close(y, expected, rtol=0, atol=0)
            y.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

    def test_nan_is_visible_with_trainable_encodings_and_no_grad(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            frozen = self.quantizer(device, 8, True)
            trainable = copy.deepcopy(frozen).requires_grad_(True)
            x = torch.tensor([float('nan'), .1], device=device)
            self.assertTrue(torch.isnan(trainable(x)[0]))
            with torch.no_grad():
                self.assertTrue(torch.isnan(trainable(x)[0]))
                self.assertTrue(torch.isnan(frozen(x)[0]))

    def test_block_encoding_keeps_nan_and_infinity_saturation(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            q = self.quantizer(device, 8, True, (2, 1), (1, 3))
            x = torch.tensor([[float('nan'), float('-inf'), .1],
                              [0., float('inf'), float('nan')]], device=device, requires_grad=True)
            y = q(x)
            self.assertTrue(torch.equal(torch.isnan(y), torch.isnan(x)))
            self.assertTrue(torch.isfinite(y[~torch.isnan(x)]).all())
            y.backward(torch.ones_like(y))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_fast_path_preserves_transposed_and_channels_last_layout(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for channels_last in [False, True]:
                for has_nan in [False, True]:
                    with self.subTest(device=device, channels_last=channels_last, has_nan=has_nan):
                        q = self.quantizer(device, 8, True)
                        value = torch.linspace(-.8, .8, 120, device=device)
                        if channels_last:
                            value = value.reshape(2, 3, 4, 5).to(memory_format=torch.channels_last)
                        else:
                            value = value.reshape(10, 12).t()
                        value = value.detach().requires_grad_()
                        if has_nan:
                            with torch.no_grad():
                                value[(0,) * value.ndim] = float('nan')
                        reference = value.detach().clone(memory_format=torch.preserve_format).requires_grad_()
                        scale, offset = q.get_scale(), q.get_offset()
                        expected = torch.fake_quantize_per_tensor_affine(
                            reference, scale.reshape(()), -offset.to(torch.int32).reshape(()), q.qmin, q.qmax)
                        actual = q(value)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(value)))
                        torch.testing.assert_close(actual[~torch.isnan(value)], expected[~torch.isnan(value)],
                                                   rtol=0, atol=0)
                        actual.backward(torch.ones_like(actual))
                        expected.backward(torch.ones_like(expected))
                        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


def make_log(device, symmetric=True, shape=()):
    module = QuantizationMixin.from_module(Log()).to(device)
    module.input_quantizers[0] = QuantizeDequantize((), 16, True).to(device)
    module.input_quantizers[0].set_range(
        torch.tensor(-202.4, device=device), torch.tensor(202.4, device=device))
    output = QuantizeDequantize(shape, 8, symmetric).to(device)
    output.set_range(torch.full(shape, -32., device=device),
                     torch.full(shape, 31.75, device=device))
    module.output_quantizers[0] = output
    return module


class LogZeroCodeTest(unittest.TestCase):
    def make_block_log(self, device, symmetric, block_size):
        module = make_log(device)
        output = QuantizeDequantize((2, 2), 8, symmetric,
                                   block_size=block_size).to(device)
        lower = torch.tensor([[-32., -16.], [-8., -4.]], device=device)
        output.set_range(lower, -lower)
        module.output_quantizers[0] = output
        return module

    def test_block_zero_codes_use_their_own_finite_lower_bound(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for symmetric in [False, True]:
                for block_size in [(1, 2), (-1, -1)]:
                    for frozen in [False, True]:
                        with self.subTest(device=device, symmetric=symmetric,
                                          block_size=block_size, frozen=frozen):
                            module = self.make_block_log(device, symmetric, block_size)
                            module.requires_grad_(not frozen)
                            x = torch.tensor([[[0., .1, 1e-12, .2],
                                               [0., .1, .003, .2]]], device=device)
                            x = x.repeat(3, 1, 1).requires_grad_()
                            effective = module.input_quantizers[0](x)
                            expected = module.output_quantizers[0](torch.log(effective))
                            seen = []
                            handle = module.output_quantizers[0].register_forward_pre_hook(
                                lambda _, args: seen.append(args[0].detach().clone()))
                            actual = module(x)
                            handle.remove()
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            lower = module.output_quantizers[0].get_min().detach()
                            lower = lower.repeat_interleave(2, dim=-1).expand_as(x)
                            zero = effective == 0
                            torch.testing.assert_close(seen[0][zero], lower[zero], rtol=0, atol=0)
                            self.assertTrue(torch.isfinite(seen[0]).all())
                            actual.sum().backward()
                            self.assertTrue(torch.isfinite(x.grad).all())
                            torch.testing.assert_close(x.grad[zero], torch.zeros_like(x.grad[zero]))
                            for name, parameter in module.named_parameters():
                                if parameter.requires_grad:
                                    self.assertIsNotNone(parameter.grad, name)
                                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_block_encoding_gradients_match_finite_saturated_reference(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for symmetric in [False, True]:
                with self.subTest(device=device, symmetric=symmetric):
                    module = self.make_block_log(device, symmetric, (1, 2))
                    reference = copy.deepcopy(module.output_quantizers[0])
                    x = torch.tensor([[0., 1e-12, 0., .003],
                                      [0., 1e-12, .003, 0.]], device=device, requires_grad=True)
                    module(x).sum().backward()
                    reference(torch.full_like(x, -10000.)).sum().backward()
                    for actual, expected in zip(module.output_quantizers[0].parameters(),
                                                reference.parameters()):
                        self.assertTrue(torch.isfinite(actual.grad).all())
                        torch.testing.assert_close(actual.grad, expected.grad)

    def test_block_calibration_and_invalid_inputs_keep_log_semantics(self):
        module = self.make_block_log('cpu', False, (1, 2)).requires_grad_(False)
        x = torch.tensor([[1e-12, .003, .1, .2], [.004, .01, .1, .3]])
        seen = []
        handle = module.output_quantizers[0].register_forward_pre_hook(
            lambda _, args: seen.append(args[0].detach().clone()))
        with torch.no_grad(), compute_encodings(module):
            module(x)
        handle.remove()
        torch.testing.assert_close(seen[0], torch.log(x), rtol=0, atol=0)
        x[0, 0] = -1.
        x[1, 2] = float('nan')
        actual = module(x)
        expected = module.output_quantizers[0](torch.log(module.input_quantizers[0](x)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertTrue(torch.isnan(actual[0, 0]))
        self.assertTrue(torch.isnan(actual[1, 2]))

    def test_forward_is_exact_qdq_and_zero_gradient_is_finite(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for symmetric in [False, True]:
                for shape in [(), (2, 1)]:
                    with self.subTest(device=device, symmetric=symmetric, shape=shape):
                        module = make_log(device, symmetric, shape).requires_grad_(False)
                        x = torch.tensor([[0., 1e-12, .003, .004, .1],
                                          [0., .002, .004, .01, .2]],
                                         device=device, requires_grad=True)
                        effective = module.input_quantizers[0](x)
                        expected = module.output_quantizers[0](torch.log(effective))
                        seen = []
                        handle = module.output_quantizers[0].register_forward_pre_hook(
                            lambda _, args: seen.append(args[0].detach()))
                        actual = module(x)
                        handle.remove()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertTrue(torch.isfinite(seen[0]).all())
                        actual.sum().backward()
                        self.assertTrue(torch.isfinite(x.grad).all())
                        torch.testing.assert_close(x.grad[effective == 0],
                                                   torch.zeros_like(x.grad[effective == 0]))

    def test_regular_input_and_encoding_gradients_match_reference(self):
        module = make_log('cpu')
        reference = copy.deepcopy(module)
        x = torch.tensor([.1, .2, .3], requires_grad=True)
        other = x.detach().clone().requires_grad_()
        module(x).sum().backward()
        reference.output_quantizers[0](
            torch.log(reference.input_quantizers[0](other))).sum().backward()
        torch.testing.assert_close(x.grad, other.grad)
        for a, b in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)

    def test_zero_encoding_gradient_matches_finite_saturated_reference(self):
        for symmetric in [False, True]:
            module = make_log('cpu', symmetric)
            reference = copy.deepcopy(module.output_quantizers[0])
            x = torch.tensor([0., 1e-12], requires_grad=True)
            module(x).sum().backward()
            reference(torch.full_like(x, -10000.)).sum().backward()
            for a, b in zip(module.output_quantizers[0].parameters(), reference.parameters()):
                self.assertTrue(torch.isfinite(a.grad).all())
                torch.testing.assert_close(a.grad, b.grad)

    def test_disabled_output_keeps_log_zero_and_invalid_input(self):
        module = make_log('cpu').requires_grad_(False)
        module.output_quantizers[0] = None
        x = torch.tensor([0., -1., .1], requires_grad=True)
        actual = module(x)
        expected = torch.log(module.input_quantizers[0](x))
        torch.testing.assert_close(actual, expected, equal_nan=True)
        self.assertTrue(torch.isneginf(actual[0]))
        self.assertTrue(torch.isnan(actual[1]))
        module.output_quantizers[0] = make_log('cpu').output_quantizers[0]
        self.assertTrue(torch.isnan(module(x)[1]))

    def test_calibration_observes_original_log(self):
        module = make_log('cpu').requires_grad_(False)
        seen = []
        handle = module.output_quantizers[0].register_forward_pre_hook(
            lambda _, args: seen.append(args[0].detach()))
        x = torch.tensor([1e-12, .003, .1])
        with torch.no_grad(), compute_encodings(module):
            module(x)
        handle.remove()
        torch.testing.assert_close(seen[0], torch.log(x), rtol=0, atol=0)


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
