"""Stage-file reload must preserve the execution grid, not just its real range."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantization.affine import QuantizeDequantize
from aimet_torch.staged_quantization_utils import save_quantizer_encodings, load_quantizer_encodings


def linear(bits=16, symmetric=True, shape=()):
    layer = QuantizationMixin.from_module(torch.nn.Linear(2, 2))
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(torch.tensor([.75, -.25]))
    layer.input_quantizers[0] = None
    layer.output_quantizers[0] = None
    layer.param_quantizers['weight'] = None
    q = QuantizeDequantize(shape, bits, symmetric)
    q.set_range(torch.full(shape, -1.), torch.full(shape, 1.))
    q.requires_grad_(False)
    layer.param_quantizers['bias'] = q
    return torch.nn.Sequential(layer)


class StageEncodingGridTest(unittest.TestCase):
    def round_trip(self, source, target):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stage.json'
            saved = save_quantizer_encodings(source, str(path), verbose=False)
            load_quantizer_encodings(target, str(path), skip_if_not_found=False,
                                     allow_overwrite=False, verbose=False)
            actual = save_quantizer_encodings(target, str(path), verbose=False)
        self.assertEqual(actual, saved)
        x = torch.zeros(3, 2)
        torch.testing.assert_close(source(x), target(x), rtol=0, atol=0)
        self.assertFalse(target[0].param_quantizers['bias']._allow_overwrite)

    def test_bias32_restores_into_fresh_bias16(self):
        for shape in [(), (2,)]:
            with self.subTest(shape=shape):
                source = linear(32, shape=shape)
                q = source[0].param_quantizers['bias']
                scale = torch.full(shape, 2.**-8)
                q.set_range(scale * q.qmin, scale * q.qmax)
                self.round_trip(source, linear(shape=shape))

    def test_affine_signed_unsigned_and_decimal_scale(self):
        for bits, symmetric in [(8, False), (16, True), (8, True)]:
            with self.subTest(bits=bits, symmetric=symmetric):
                source = linear(bits, symmetric, (2,))
                q = source[0].param_quantizers['bias']
                q._reparametrize_to_scale_offset()
                with torch.no_grad():
                    q.scale.copy_(torch.tensor([.013007, .027019]))
                    if not symmetric:
                        q.offset.copy_(torch.tensor([-37., -121.]))
                self.round_trip(source, linear(4, not symmetric, (2,)))

    def test_input_and_output_grids(self):
        source, target = linear(), linear()
        for attr in ['input_quantizers', 'output_quantizers']:
            q = QuantizeDequantize((), 8, False)
            q.set_range(torch.tensor(-.4), torch.tensor(1.3))
            q.requires_grad_(False)
            getattr(source[0], attr)[0] = q
            getattr(target[0], attr)[0] = QuantizeDequantize((), 16, True)
        self.round_trip(source, target)

    def test_invalid_encoding_is_not_silently_accepted(self):
        source = linear()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stage.json'
            saved = save_quantizer_encodings(source, str(path), verbose=False)
            key = '0.param_quantizers.bias'
            for field, value in [('scale', 0.), ('scale', float('nan')),
                                 ('offset', .5), ('qmax', -32768),
                                 ('symmetric', 'false'), ('shape', [3])]:
                with self.subTest(field=field, value=value):
                    bad = copy.deepcopy(saved)
                    bad[key][field] = value
                    path.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        load_quantizer_encodings(linear(), str(path),
                                                 skip_if_not_found=False, verbose=False)
