"""GRU 在固定输入与编码下的反向一致性。"""
import unittest

import torch
from quant_gru import QuantGRU


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class GruBackwardContractTest(unittest.TestCase):
    def model_and_input(self, bits, bidirectional=False, carrier='float32'):
        torch.manual_seed(42)
        model = QuantGRU(9, 17, bidirectional=bidirectional,
                         quant_storage_dtype=carrier).cuda()
        x = torch.randn(7, 129, 9, device='cuda') * .5
        h = torch.randn(2 if bidirectional else 1, 129, 17, device='cuda') * .1
        if bits is not None:
            model.set_all_bitwidth(bits)
            model.calibrating = True
            with torch.no_grad():
                model(x, h)
            model.calibrating = False
            model.finalize_calibration()
            model.use_quantization = True
            model.set_quant_params_locked(True)
        return model, x, h

    def gradients(self, model, x, h, seed=123):
        model.zero_grad(set_to_none=True)
        x = x.detach().clone().requires_grad_()
        h = h.detach().clone().requires_grad_()
        y, last = model(x, h)
        torch.manual_seed(seed)
        torch.autograd.backward((y, last), (torch.randn_like(y), torch.randn_like(last)))
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        gradients.update(input=x.grad, hidden=h.grad)
        return (y.detach(), last.detach()), gradients

    def test_repeated_backward_is_bitwise_identical(self):
        for bits, bidirectional, carrier in [
                (None, False, 'float32'), (8, False, 'float32'),
                (16, False, 'float32'), (8, True, 'float32'), (8, False, 'int32')]:
            with self.subTest(bits=bits, bidirectional=bidirectional, carrier=carrier):
                model, x, h = self.model_and_input(bits, bidirectional, carrier)
                model.train()
                outputs, expected = self.gradients(model, x, h)
                for repeat in range(4):
                    actual_outputs, actual = self.gradients(model, x, h)
                    for left, right in zip(outputs, actual_outputs):
                        self.assertTrue(torch.equal(left, right), '固定输入的前向发生变化')
                    for name in expected:
                        difference = (expected[name] - actual[name]).abs().max().item()
                        self.assertTrue(torch.equal(expected[name], actual[name]),
                                        f'{name}: repeat={repeat}, max_abs={difference}')


if __name__ == '__main__':
    unittest.main()
