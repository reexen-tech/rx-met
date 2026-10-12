"""Training semantics of the BatchNorm used by the KWS example."""
import unittest

import torch
from torch import nn

from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from aimet_torch.utils_rx import set_train_mode_freeze_bn


class QuantizableBatchNormTest(unittest.TestCase):
    def test_training_statistics_gradients_and_eval_match_torch(self):
        for channels in (1, 4):
            for momentum in (0.01, 0.1, None):
                with self.subTest(channels=channels, momentum=momentum):
                    torch.manual_seed(19)
                    reference = nn.BatchNorm2d(channels, momentum=momentum)
                    actual = QuantizableBatchNorm2d(channels, momentum=momentum)
                    actual.load_state_dict(reference.state_dict())
                    mean_buffer, var_buffer = actual.running_mean, actual.running_var
                    for _ in range(3):
                        x = torch.randn(2, channels, 3, 5, requires_grad=True)
                        y = x.detach().clone().requires_grad_()
                        left, right = reference(x), actual(y)
                        torch.testing.assert_close(left, right)
                        upstream = torch.randn_like(left)
                        left.backward(upstream)
                        right.backward(upstream)
                        torch.testing.assert_close(x.grad, y.grad)
                        torch.testing.assert_close(reference.weight.grad, actual.weight.grad)
                        torch.testing.assert_close(reference.bias.grad, actual.bias.grad)
                        for name, value in reference.state_dict().items():
                            torch.testing.assert_close(value, actual.state_dict()[name])
                    self.assertIs(actual.running_mean, mean_buffer)
                    self.assertIs(actual.running_var, var_buffer)
                    reference.eval()
                    actual.eval()
                    torch.testing.assert_close(reference(x), actual(x))

    def test_qat_freezes_custom_bn_statistics(self):
        model = nn.Sequential(QuantizableBatchNorm2d(4))
        set_train_mode_freeze_bn(model, verbose=False)
        self.assertFalse(model[0].training)
        before = model[0].running_var.clone()
        model(torch.randn(2, 4, 3, 5))
        torch.testing.assert_close(before, model[0].running_var)


if __name__ == '__main__':
    unittest.main()
