"""冻结 affine QDQ 不能将非法输入 NaN 伪装成有限编码。"""
import copy
import unittest

import torch

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


if __name__ == '__main__':
    unittest.main()
