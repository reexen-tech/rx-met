"""eval 不关闭 autograd，仍需保存真实前向的门值和截断 mask。"""
import tempfile
import unittest
from unittest import mock

import torch
import quant_gru
from quant_gru import QuantGRU

from tests.test_gru_clamp_chain_rule import fixed_model


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class GruEvalBackwardTest(unittest.TestCase):
    def backward(self, model, value, h0):
        model.zero_grad(set_to_none=True)
        value = value.detach().clone().requires_grad_()
        h0 = h0.detach().clone().requires_grad_()
        y, last = model(value, h0)
        pending, visited, saved = [y.grad_fn], set(), []
        while pending:
            node = pending.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            if type(node).__name__ == 'GRUFunctionBackward':
                saved.extend(t.detach().clone() for t in node.saved_tensors)
            pending.extend(parent for parent, _ in node.next_functions)
        self.assertTrue(saved)
        (y.square().sum() + last.sum()).backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()
                     if p.requires_grad}
        gradients.update(input=value.grad, hidden=h0.grad)
        return (y.detach(), last.detach()), saved, gradients

    def check_modes(self, model, x, h):
        model.train()
        expected, cache, gradients = self.backward(model, x, h)
        model.eval()
        actual, actual_cache, actual_gradients = self.backward(model, x, h)
        self.assertFalse(model.training)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for index, (left, right) in enumerate(zip(cache, actual_cache)):
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=f'缓存 {index}')
        for name in gradients:
            torch.testing.assert_close(gradients[name], actual_gradients[name], rtol=0, atol=0,
                                       msg=f'梯度 {name}')

    def test_eval_saturation_uses_the_same_masks(self):
        for carrier in ['float32', 'int32']:
            for clipped in [None, 'weight_ih_linear', 'new_gate_input', 'new_gate_output']:
                with self.subTest(carrier=carrier, clipped=clipped), tempfile.TemporaryDirectory() as directory:
                    model, _ = fixed_model(directory, clipped)
                    model.quant_storage_dtype = carrier
                    x = torch.tensor([[[.5]], [[.4]]], device='cuda')
                    h = torch.tensor([[[.2]]], device='cuda')
                    self.check_modes(model, x, h)

    def test_eval_floating_and_8_16_bit_bidirectional_backward(self):
        for bits in [None, 8, 16]:
            with self.subTest(bits=bits):
                torch.manual_seed(42)
                model = QuantGRU(3, 5, bidirectional=True).cuda()
                # 单样本避免 bias 的 atomicAdd 并发归约顺序差异，
                # 使零容差断言只检查 train/eval 的缓存和反向语义。
                x = torch.randn(4, 1, 3, device='cuda') * .2
                h = torch.randn(2, 1, 5, device='cuda') * .1
                if bits is not None:
                    model.set_all_bitwidth(bits)
                    model.calibrating = True
                    with torch.no_grad():
                        model(x, h)
                    model.calibrating = False
                    model.finalize_calibration()
                    model.use_quantization = True
                    model.set_quant_params_locked(True)
                self.check_modes(model, x, h)

    def test_eval_with_frozen_parameters_still_supports_input_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _ = fixed_model(directory, 'weight_ih_linear')
            model.requires_grad_(False)
            self.check_modes(model, torch.tensor([[[.5]]], device='cuda'),
                             torch.tensor([[[.2]]], device='cuda'))

    def test_eval_parameter_gradients_without_input_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _ = fixed_model(directory, 'weight_ih_linear')
            model.eval()
            y, _ = model(torch.tensor([[[.5]]], device='cuda'))
            self.assertGreater(y.grad_fn.saved_tensors[13].numel(), 0)
            y.sum().backward()
            torch.testing.assert_close(model.bias_ih_l0.grad,
                                       torch.zeros_like(model.bias_ih_l0), rtol=0, atol=0)

    def test_eval_no_grad_keeps_inference_cache_path(self):
        for carrier, function in [('float32', 'forward_quant_float_storage'),
                                  ('int32', 'forward_quant_int_storage')]:
            with self.subTest(carrier=carrier), tempfile.TemporaryDirectory() as directory:
                model, _ = fixed_model(directory)
                model.eval()
                model.quant_storage_dtype = carrier
                original = getattr(quant_gru.gru_ops, function)
                with mock.patch.object(quant_gru.gru_ops, function, wraps=original) as call:
                    with torch.no_grad():
                        y, _ = model(torch.tensor([[[.5]]], device='cuda'))
                self.assertFalse(call.call_args.kwargs['is_training'])
                self.assertFalse(y.requires_grad)


if __name__ == '__main__':
    unittest.main()
