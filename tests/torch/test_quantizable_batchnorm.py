"""Training semantics of the BatchNorm used by the KWS example."""
import unittest

import torch
from torch import nn

from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from aimet_torch.utils_rx import (
    check_bn_params_status, freeze_quantizer_parameters, set_train_mode_freeze_bn,
)
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.v2.quantsim import QuantizationSimModel
from tests.support.paths import REPO_ROOT


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


class BatchNormFreezeTest(unittest.TestCase):
    def test_affine_and_existing_freezes_in_nested_modules(self):
        model = nn.Sequential(nn.Sequential(nn.BatchNorm1d(2), nn.BatchNorm3d(2)),
                              QuantizableBatchNorm2d(2), nn.BatchNorm2d(2, affine=False))
        model.add_module('post_bn', nn.Linear(2, 2))
        model[0][0].weight.requires_grad_(False)
        model.post_bn.bias.requires_grad_(False)

        result = freeze_quantizer_parameters(model, verbose=False)
        newly_frozen = {'0.0.bias', '0.1.weight', '0.1.bias', '1.weight', '1.bias'}
        self.assertEqual(set(result['frozen_bn_params']), newly_frozen)
        self.assertEqual(set(result['frozen_params']), newly_frozen)
        self.assertEqual(result['frozen_count'], len(newly_frozen))
        self.assertEqual(result['trainable_params'], ['post_bn.weight'])
        self.assertEqual([n for n, p in model.named_parameters() if p.requires_grad],
                         ['post_bn.weight'])
        self.assertEqual(freeze_quantizer_parameters(model, verbose=False)['frozen_count'], 0)

    def test_freeze_bn_affine_false_on_root_bn(self):
        model = nn.BatchNorm2d(2)
        model.bias.requires_grad_(False)
        result = freeze_quantizer_parameters(model, verbose=False, freeze_bn_affine=False)
        self.assertEqual(result['frozen_params'], [])
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(model.bias.requires_grad)
        self.assertEqual(freeze_quantizer_parameters(model, verbose=False)['frozen_bn_params'],
                         ['weight'])
        self.assertFalse(model.weight.requires_grad)

    def test_mode_preserves_dropout_in_non_bn_subtree(self):
        model = nn.Sequential(nn.SyncBatchNorm(2), QuantizableBatchNorm2d(2))
        model.add_module('pre_bn', nn.Sequential(nn.Linear(2, 2), nn.Dropout()))
        model.eval()
        count = set_train_mode_freeze_bn(model)
        self.assertTrue(model.pre_bn[1].training)
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        self.assertFalse(model[1].training)
        self.assertEqual(count, 2)

    def test_quantizer_freeze_follows_parameter_owner(self):
        model = nn.Module()
        model.ranges = QuantizeDequantize((), 8, False)
        model.quantizer_scale_projection = nn.Linear(2, 2)
        result = freeze_quantizer_parameters(model, verbose=False, freeze_bn_affine=False)
        self.assertEqual(set(result['frozen_quantizer_params']), {'ranges.min', 'ranges.max'})
        self.assertTrue(all(not p.requires_grad for p in model.ranges.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.quantizer_scale_projection.parameters()))

    def test_bn_status_matches_actual_affine_owners(self):
        model = nn.Sequential(nn.BatchNorm2d(2), nn.BatchNorm2d(2, affine=False))
        model.add_module('post_bn', nn.Linear(2, 2))
        model[0].bias.requires_grad_(False)
        status = check_bn_params_status(model, verbose=False)
        self.assertEqual(status['trainable_affine_params'], ['0.weight'])
        self.assertEqual(status['frozen_affine_params'], ['0.bias'])
        self.assertEqual(status['bn_module_count'], 2)
        self.assertEqual(status['total_affine_params'], 2)
        root = check_bn_params_status(model[0], verbose=False)
        self.assertEqual(root['trainable_affine_params'], ['weight'])
        self.assertEqual(root['frozen_affine_params'], ['bias'])

    def test_prepared_bn_affine_and_mode_in_quantsim(self):
        model = nn.Sequential(QuantizableBatchNorm2d(2))
        model.add_module('rnn2d_bn', QuantizableBatchNorm2d(2))
        model.add_module('post_bn', nn.Conv2d(2, 2, 1))
        config = REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'
        sim = QuantizationSimModel(prepare_model(model.eval()), torch.ones(1, 2, 3, 3),
                                   quant_scheme='tf', config_file=str(config))
        first, second = sim.model.get_submodule('0'), sim.model.rnn2d_bn
        self.assertNotIsInstance(second, QuantizableBatchNorm2d)
        count = set_train_mode_freeze_bn(sim.model)
        result = freeze_quantizer_parameters(sim.model, verbose=False)
        self.assertEqual(set(result['frozen_bn_params']),
                         {'0.weight', '0.bias', 'rnn2d_bn.weight', 'rnn2d_bn.bias'})
        self.assertTrue(all(not p.requires_grad for m in (first, second) for p in (m.weight, m.bias)))
        self.assertTrue(all(not child.training for m in (first, second) for child in m.modules()))
        self.assertTrue(sim.model.post_bn.training)
        self.assertTrue(sim.model.post_bn.weight.requires_grad)
        self.assertEqual(count, 2)
        status = check_bn_params_status(sim.model, verbose=False)
        self.assertEqual(status['bn_module_count'], count)
        self.assertEqual(status['frozen_affine_params'], result['frozen_bn_params'])
        self.assertEqual(status['trainable_affine_params'], [])


if __name__ == '__main__':
    unittest.main()
