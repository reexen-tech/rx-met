"""The public QDQ and Q-then-DQ paths must execute the same affine grid."""
import unittest

import torch

from aimet_common.quantization_grid import IntegerGrid
from aimet_torch.v2.quantization.affine import QuantizeDequantize


class AffineQdqConsistencyTest(unittest.TestCase):
    @staticmethod
    def quantizer(scale, shape, device, *, zero_point=0, bits=16):
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        scales = [scale, scale] if shape else scale
        offsets = [-zero_point, -zero_point] if shape else -zero_point
        q = QuantizeDequantize(shape, qmin=lower, qmax=upper, symmetric=(zero_point == 0)).to(device)
        q.set_integer_grid(IntegerGrid.from_offset(scales, offsets, bits, True), symmetric=(zero_point == 0))
        return q

    def assert_grid(self, scale, inputs, *, zero_point=0, bits=16):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for shape in ((), (2, 1)):
                q = self.quantizer(scale, shape, device, zero_point=zero_point, bits=bits)
                for train_grid in (False, True):
                    with self.subTest(device=device, shape=shape, train_grid=train_grid):
                        q.requires_grad_(train_grid)
                        value = torch.tensor([inputs, inputs], device=device, dtype=torch.float32, requires_grad=True)
                        with torch.no_grad():
                            s = q.get_scale()
                            raw = torch.round(value / s) + zero_point
                            code = raw.clamp(q.qmin, q.qmax)
                            expected = (code - zero_point) * s
                            mask = ((raw >= q.qmin) & (raw <= q.qmax)).to(value.dtype)
                            encoding = q.get_encodings()
                            manual = encoding.dequantize(encoding.quantize(value))
                            torch.testing.assert_close(manual, expected, rtol=0, atol=0, equal_nan=True)
                            torch.testing.assert_close(q(value), expected, rtol=0, atol=0, equal_nan=True)
                        actual = q(value)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        dx, = torch.autograd.grad(actual, value, torch.ones_like(actual))
                        torch.testing.assert_close(dx, mask, rtol=0, atol=0)

    def test_division_boundary_matches_split_encoding(self):
        # FP32 2.17 / .02 is slightly above 108.5; multiplying by the rounded reciprocal lands on 108.5.
        self.assert_grid(.02, [2.17, -2.17, 2.21, -2.21])

    def test_ste_uses_same_saturation_boundary(self):
        self.assert_grid(.00804289523512125, [263.5455627441406])

    def test_odd_zero_point_half_ties(self):
        self.assert_grid(.5, [.25, -.25, .75, -.75], zero_point=7)

    def test_saturation_precedes_integer_conversion(self):
        self.assert_grid(1., [1.e20, -1.e20, float('inf'), -float('inf'), float('nan')], bits=8)

    def test_exported_module_executes_same_grid(self):
        if not getattr(torch.compiler, "is_exporting", None):
            self.skipTest("Requires a PyTorch version with torch.export support")
        from aimet_torch.v2.experimental.export import export
        for shape in ((), (2, 1)):
            with self.subTest(shape=shape):
                q = self.quantizer(.02, shape, "cpu").requires_grad_(False)
                x = torch.tensor([[2.17, -2.17, 1.e20, -1.e20]])
                x = x.expand(2, -1).contiguous()
                expected = q.get_encodings().dequantize(q.get_encodings().quantize(x))
                exported = export(q, (x,)).module()
                torch.testing.assert_close(exported(x), expected, rtol=0, atol=0)

    def test_low_precision_inputs_keep_the_encoding_scale(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    q = self.quantizer(.02, (), device, bits=8).requires_grad_(False)
                    x = torch.linspace(-3., 3., 2049, device=device).to(dtype).requires_grad_()
                    raw = torch.round(x.detach().float() / q.get_scale())
                    expected_codes = raw.clamp(-128, 127)
                    expected = (expected_codes * q.get_scale()).to(dtype)
                    encoding = q.get_encodings()
                    codes = encoding.quantize(x)
                    torch.testing.assert_close(codes, expected_codes.to(dtype), rtol=0, atol=0)
                    torch.testing.assert_close(encoding.dequantize(codes), expected, rtol=0, atol=0)
                    actual = q(x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    grad, = torch.autograd.grad(actual, x, torch.ones_like(actual))
                    torch.testing.assert_close(grad, ((raw >= -128) & (raw <= 127)).to(dtype), rtol=0, atol=0)

    def test_exported_low_precision_inputs_keep_the_encoding_scale(self):
        if not getattr(torch.compiler, "is_exporting", None):
            self.skipTest("Requires a PyTorch version with torch.export support")
        from aimet_torch.v2.experimental.export import export
        q = self.quantizer(.02, (), "cpu", bits=8).requires_grad_(False)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.linspace(-3., 3., 2049).to(dtype)
                expected = q.get_encodings().dequantize(q.get_encodings().quantize(x))
                torch.testing.assert_close(export(q, (x,)).module()(x), expected, rtol=0, atol=0)

    def test_trainable_scale_and_offset_gradients(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                q = self.quantizer(.02, (), device, zero_point=5, bits=8).requires_grad_(True)
                x = torch.tensor([-5., -.136, .017, 3.6], device=device, requires_grad=True)
                upstream = x.new_tensor([.5, 2., -1., .75])
                y = q(x)
                dx, ds, do = torch.autograd.grad(y, (x, q.scale, q.offset), upstream)
                scale, offset = q.get_scale().detach(), q.get_offset().detach()
                units = x.detach() / scale
                raw = units.round() - offset
                mask = (raw >= q.qmin) & (raw <= q.qmax)
                effective = raw.clamp(q.qmin, q.qmax) + offset
                torch.testing.assert_close(dx, upstream * mask)
                torch.testing.assert_close(ds, (upstream * (effective - units * mask)).sum())
                torch.testing.assert_close(do, (upstream * ~mask * scale).sum())


if __name__ == '__main__':
    unittest.main()
