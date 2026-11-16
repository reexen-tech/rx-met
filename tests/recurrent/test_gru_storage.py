"""Independent GRU carriers retain QAT caches and encoding lifecycle behavior."""

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
    def calibrated(self, bits, pot2, bidirectional=False, carrier='float32'):
        torch.manual_seed(42)
        m = QuantGRU(4, 3, bidirectional=bidirectional,
                     quant_storage_dtype=carrier).cuda()
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

    def test_each_carrier_preserves_caches_masks_and_backward(self):
        for bits in (8, 16):
            for pot2 in (False, True):
                for carrier in ('float32', 'int32'):
                    with self.subTest(bits=bits, pot2=pot2, carrier=carrier):
                        model, x = self.calibrated(bits, pot2, carrier=carrier)
                        model.train()
                        value = x.clone().requires_grad_()
                        h0 = torch.full((1, 2, 3), .1, device='cuda', requires_grad=True)
                        y, h = model(value, h0)
                        cache = y.grad_fn.saved_tensors
                        dtype = torch.float32 if carrier == 'float32' else torch.int32
                        for codes in cache[:5]:
                            self.assertEqual(codes.dtype, dtype)
                            self.assertTrue(torch.isfinite(codes).all())
                            torch.testing.assert_close(codes.float(), codes.float().round(),
                                                       rtol=0, atol=0)
                        for mask in cache[7:]:
                            self.assertEqual(mask.dtype, torch.uint8)
                            self.assertTrue(((mask == 0) | (mask == 1)).all())
                        for output in (y, h, cache[5], cache[6]):
                            self.assertEqual(output.dtype, torch.float32)
                            self.assertTrue(torch.isfinite(output).all())
                        (y.square().sum() + h.square().sum()).backward()
                        gradients = [p.grad for p in model.parameters()] + [value.grad, h0.grad]
                        for gradient in gradients:
                            self.assertIsNotNone(gradient)
                            self.assertTrue(torch.isfinite(gradient).all())
                        self.assertTrue(any(torch.any(gradient != 0) for gradient in gradients))
                        model.eval()
                        with torch.no_grad():
                            actual_y, actual_h = model(x, h0.detach())
                        torch.testing.assert_close(actual_y, y.detach(), rtol=0, atol=0)
                        torch.testing.assert_close(actual_h, h.detach(), rtol=0, atol=0)

    def test_affine_to_pot2_then_lock_preserves_exported_runtime(self):
        for carrier in ('float32', 'int32'):
            for bidirectional in (False, True):
                with self.subTest(carrier=carrier, bidirectional=bidirectional):
                    self.check_affine_to_pot2_then_lock(carrier, bidirectional)

    def check_affine_to_pot2_then_lock(self, carrier, bidirectional):
        with tempfile.TemporaryDirectory() as directory:
            source, x = self.calibrated(8, False, bidirectional, carrier)
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
            target = QuantGRU(4, 3, bidirectional=bidirectional,
                             quant_storage_dtype=carrier).cuda().eval()
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
        for carrier in ('float32', 'int32'):
            for pot2 in (False, True):
                for bidirectional in (False, True):
                    with self.subTest(carrier=carrier, pot2=pot2, bidirectional=bidirectional):
                        self.check_public_reload(carrier, pot2, bidirectional)

    def check_public_reload(self, carrier, pot2, bidirectional):
        source, x = self.calibrated(8, pot2, bidirectional, carrier)
        source.eval()
        document = source.export_quant_params_to_aimet_format({}, module_name='gru')
        target = QuantGRU(4, 3, bidirectional=bidirectional,
                         quant_storage_dtype=carrier).cuda().eval()
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


if __name__ == "__main__":
    unittest.main()
