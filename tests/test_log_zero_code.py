"""QDQ equivalence and gradients for zero-coded Log inputs."""
import copy
import unittest

import torch

from aimet_torch._base.nn.modules.custom import Log
from aimet_torch.v2.nn import QuantizationMixin, compute_encodings
from aimet_torch.v2.quantization.affine import QuantizeDequantize


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


if __name__ == '__main__':
    unittest.main()
