"""GRU QAT carriers must share target outputs, gate caches, masks and gradients."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from quant_gru import QuantGRU


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class GruCarrierContractTest(unittest.TestCase):
    def calibrated(self, bits, pot2, bidirectional=False):
        torch.manual_seed(42)
        m = QuantGRU(4, 3, bidirectional=bidirectional).cuda()
        m.set_all_bitwidth(bits)
        if pot2:
            m.enable_pot2()
        x = torch.randn(11, 2, 4, device='cuda') * .5
        m.calibrating = True
        with torch.no_grad():
            m(x)
        m.calibrating = False
        m.finalize_calibration()
        m.use_quantization = True
        return m, x

    def test_forward_gate_masks_and_backward_match(self):
        for bits in (8, 16):
            for pot2 in (False, True):
                with self.subTest(bits=bits, pot2=pot2), tempfile.TemporaryDirectory() as directory:
                    floating, x = self.calibrated(bits, pot2)
                    path = str(Path(directory) / 'encodings.json')
                    floating.export_quant_params(path)
                    integer = QuantGRU(4, 3, quant_storage_dtype='int32').cuda()
                    integer.load_state_dict(floating.state_dict())
                    integer.load_quant_params(path)
                    integer.use_quantization = True
                    outputs = []
                    gradients = []
                    caches = []
                    for model in (floating, integer):
                        model.train()
                        value = x.clone().requires_grad_()
                        h0 = torch.full((1, 2, 3), .1, device='cuda', requires_grad=True)
                        y, h = model(value, h0)
                        caches.append([t.detach().clone() for t in y.grad_fn.saved_tensors])
                        outputs.append((y.detach(), h.detach()))
                        (y.square().sum() + h.square().sum()).backward()
                        gradients.append([p.grad for p in model.parameters()] + [value.grad, h0.grad])
                    for a, b in zip(outputs[0], outputs[1]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    for a, b in zip(caches[0], caches[1]):
                        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
                    for a, b in zip(gradients[0], gradients[1]):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_affine_to_pot2_then_lock_preserves_exported_runtime(self):
        for bidirectional in (False, True):
            with self.subTest(bidirectional=bidirectional), tempfile.TemporaryDirectory() as directory:
                source, x = self.calibrated(8, False, bidirectional)
                source.enable_pot2()
                # This is the real workflow order: switch, lock, then first QAT forward.
                source.set_quant_params_locked(True)
                source.eval()
                with torch.no_grad():
                    expected = source(x)[0]
                path = str(Path(directory) / 'native.json')
                source.export_quant_params(path)
                native = json.loads(Path(path).read_text())
                for field in ('operators', 'operators_reverse'):
                    for encoding in native.get(field, {}).values():
                        scales = encoding['scale']
                        for scale in scales if isinstance(scales, list) else [scales]:
                            self.assertEqual(math.frexp(scale)[0], .5)
                document = source.export_quant_params_to_aimet_format({}, module_name='gru')
                target = QuantGRU(4, 3, bidirectional=bidirectional).cuda().eval()
                target.load_state_dict(source.state_dict())
                self.assertTrue(target.load_quant_params_from_aimet_format(document, module_name='gru'))
                target.use_quantization = True
                with torch.no_grad():
                    torch.testing.assert_close(expected, target(x)[0], atol=0, rtol=0)

    def test_lock_rejects_dirty_imported_grid_without_calibration(self):
        source, _ = self.calibrated(8, False)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'native.json')
            source.export_quant_params(path)
            target = QuantGRU(4, 3).cuda()
            target.load_quant_params(path)
            target.enable_pot2()
            with self.assertRaisesRegex(RuntimeError, '校准数据已丢失'):
                target.set_quant_params_locked(True)
            self.assertFalse(target._quant_params_locked)

    def test_public_reload_recovers_affine_and_pot2_execution(self):
        for pot2 in (False, True):
            for bidirectional in (False, True):
                with self.subTest(pot2=pot2, bidirectional=bidirectional):
                    source, x = self.calibrated(8, pot2, bidirectional)
                    source.eval()
                    document = source.export_quant_params_to_aimet_format({}, module_name='gru')
                    target = QuantGRU(4, 3, bidirectional=bidirectional).cuda().eval()
                    target.load_state_dict(source.state_dict())
                    self.assertTrue(target.load_quant_params_from_aimet_format(copy.deepcopy(document), module_name='gru'))
                    target.use_quantization = True
                    self.assertEqual(target.use_pot2_scale, pot2)
                    with torch.no_grad():
                        torch.testing.assert_close(source(x)[0], target(x)[0], rtol=0, atol=0)
                    cloned = copy.deepcopy(target)
                    # deepcopy drops native grids; reloading must retain the same mode.
                    self.assertTrue(cloned.load_quant_params_from_aimet_format(copy.deepcopy(document), module_name='gru'))
                    with torch.no_grad():
                        torch.testing.assert_close(source(x)[0], cloned(x)[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
