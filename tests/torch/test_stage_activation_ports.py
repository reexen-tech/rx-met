"""Ordinary activation port layouts must restore and lock the intended grid."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.v2.quantization.affine import QuantizeDequantize


def record(scale):
    return dict(bitwidth=8, dtype='INT8', scale=scale, zero_point=0,
                real_min=-128 * scale, real_max=127 * scale, is_symmetric='True')


def model(shape=()):
    layer = nn.Module()
    layer.input_quantizers = nn.ModuleList([QuantizeDequantize(shape, 16, True) for _ in range(2)])
    layer.output_quantizers = nn.ModuleList([QuantizeDequantize(shape, 16, True)])
    return nn.Sequential(layer)


def load(target, node, version=None, **kwargs):
    doc = {'activation_encodings': {'0': node}, 'param_encodings': {}}
    if version is not None:
        doc['schema_version'] = version
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.json'
        path.write_text(json.dumps(doc))
        return load_quantizer_encodings(target, str(path), allow_overwrite=False,
                                        verbose=False, **kwargs)


class StageActivationPortsTest(unittest.TestCase):
    def assert_grid(self, quantizer, scale):
        self.assertTrue(quantizer.is_initialized())
        for name in ('min', 'max'):
            self.assertFalse(quantizer.is_overwrite_allowed(name))
        self.assertEqual((quantizer.qmin, quantizer.qmax), (-128, 127))
        torch.testing.assert_close(quantizer.get_scale(), torch.tensor(scale), rtol=0, atol=0)

    def test_legacy_singleton_wrapper_preserves_each_outer_port(self):
        for version in (None, 1, 2, 3):
            with self.subTest(version=version):
                target = model()
                stats = load(target, {'input': {'1': {'0': record(.25)}, '0': {'0': record(.5)}},
                                      'output': {'0': {'0': record(.125)}}}, version)
                self.assertEqual((stats['loaded'], stats['skipped']), (3, 0))
                for quantizer, scale in zip(target[0].input_quantizers, [.5, .25]):
                    self.assert_grid(quantizer, scale)
                self.assert_grid(target[0].output_quantizers[0], .125)

    def test_list_hole_sparse_index_and_flat_record(self):
        for inputs in ([None, record(.25)], {'1': record(.25)}):
            with self.subTest(inputs=inputs):
                target = model()
                stats = load(target, {'input': inputs, 'output': record(.125)}, 4)
                self.assertEqual((stats['loaded'], stats['skipped']), (2, 0))
                self.assertFalse(target[0].input_quantizers[0].is_initialized())
                self.assertTrue(target[0].input_quantizers[0].is_overwrite_allowed('min'))
                self.assert_grid(target[0].input_quantizers[1], .25)
                self.assert_grid(target[0].output_quantizers[0], .125)

    def test_channel_records_and_compact_channel_arrays(self):
        compact = dict(record(.5), scale=[.5, .25], real_min=[-64., -32.],
                       real_max=[63.5, 31.75], zero_point=[0, 0])
        for payload in ([record(.5), record(.25)], compact):
            with self.subTest(payload=payload):
                target = model((2,))
                stats = load(target, {'input': {'1': payload}})
                self.assertEqual((stats['loaded'], stats['skipped']), (1, 0))
                self.assertFalse(target[0].input_quantizers[0].is_initialized())
                self.assert_grid(target[0].input_quantizers[1], [.5, .25])

    def test_invalid_slot_does_not_initialize_or_lock_quantizer(self):
        for payload in ({'metadata': record(.5)}, {'0': {'0': record(.5)}},
                        {'1': record(.5)}, {'0': record(.5), '2': record(.25)},
                        [], 'invalid', {'real_min': -1}, [record(.25), {}],
                        [record(.25), dict(record(.5), is_symmetric='False')]):
            with self.subTest(payload=payload):
                target = model()
                with self.assertRaisesRegex(ValueError, r'0\.input\.1'):
                    load(target, {'input': {'1': payload}}, skip_if_not_found=False)
                q = target[0].input_quantizers[1]
                self.assertFalse(q.is_initialized())
                self.assertTrue(q.is_overwrite_allowed('min'))
                self.assertEqual(q.bitwidth, 16)

    def test_invalid_io_container_is_reported(self):
        for ports in (42, 'bad'):
            with self.subTest(ports=ports):
                target = model()
                with self.assertRaisesRegex(ValueError, r'0\.output'):
                    load(target, {'output': ports})
                self.assertFalse(target[0].output_quantizers[0].is_initialized())

    def test_float_record_without_range_leaves_integer_quantizer_untouched(self):
        target = model()
        stats = load(target, {'input': {'1': {'bitwidth': 16, 'dtype': 'FLOAT16'}}}, 4)
        self.assertEqual((stats['loaded'], stats['skipped']), (0, 0))
        self.assertFalse(target[0].input_quantizers[1].is_initialized())
        self.assertTrue(target[0].input_quantizers[1].is_overwrite_allowed('min'))

    def test_reload_real_sdk_serializer_channel_list(self):
        from aimet_torch.v2.quantsim import QuantizationSimModel
        from tests.support.paths import REPO_ROOT

        config = REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'
        sim = QuantizationSimModel(nn.Sequential(nn.Linear(2, 2)), torch.ones(1, 2),
                                   quant_scheme='tf', config_file=str(config))
        layer = sim.model[0]
        q = QuantizeDequantize((2,), 8, True)
        q.set_range(torch.tensor([-64., -32.]), torch.tensor([63.5, 31.75]))
        layer.input_quantizers[0] = q
        layer.output_quantizers[0] = QuantizeDequantize((), 8, True)
        layer.output_quantizers[0].set_range(torch.tensor(-16.), torch.tensor(15.875))
        with tempfile.TemporaryDirectory() as directory:
            sim.save_encodings_to_json(directory, 'saved')
            path = Path(directory) / 'saved.json'
            saved = json.loads(path.read_text())['activation_encodings']['0']
            self.assertEqual(len(saved['input']['0']), 2)
            self.assertIn('scale', saved['output']['0'])
            layer.input_quantizers[0] = QuantizeDequantize((2,), 16, True)
            layer.output_quantizers[0] = QuantizeDequantize((), 16, True)
            load_quantizer_encodings(sim.model, str(path), skip_if_not_found=False,
                                     allow_overwrite=False, verbose=False)
            self.assert_grid(layer.input_quantizers[0], [.5, .25])
            self.assert_grid(layer.output_quantizers[0], .125)

    def test_rejected_record_preserves_existing_grid(self):
        bad_grids = (
            dict(record(.5), zero_point=1, real_min=-64.5, real_max=63.),
            dict(record(.5), scale=[.5, .25], real_min=[-64., -32.],
                 real_max=[63.5, 31.75], zero_point=[0, 0]),
            {'real_min': float('nan'), 'real_max': 1.},
            {'real_min': 1., 'real_max': -1.},
            {'0': record(.5)},  # A schema-4 record cannot have a legacy wrapper.
        )
        for payload in bad_grids:
            with self.subTest(payload=payload):
                target = model()
                q = target[0].input_quantizers[1]
                q.set_range(torch.tensor(-8.), torch.tensor(7.))
                before = q.get_scale().clone()
                with self.assertRaisesRegex(ValueError, r'0\.input\.1'):
                    load(target, {'input': {'1': payload}}, 4)
                self.assertEqual(q.bitwidth, 16)
                torch.testing.assert_close(q.get_scale(), before, rtol=0, atol=0)
                self.assertTrue(q.is_overwrite_allowed('min'))
