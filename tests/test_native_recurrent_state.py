"""Shared GRU/LSTM quantization-state and Po2 regression tests (CUDA)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import onnx
import torch
from torch import nn
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.staged_quantization_utils import save_quantizer_encodings, load_quantizer_encodings
from aimet_torch.rx_export.export_onnx_json import export_onnx_json

CONFIG = Path(__file__).resolve().parents[1] / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'


class RecurrentPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.gru = QuantGRU(4, 4, batch_first=True)
        self.lstm = QuantLSTM(4, 3, batch_first=True)
        self.head = nn.Linear(3, 2)

    def forward(self, x):
        x, _ = self.gru(self.proj(x))
        x, _ = self.lstm(x)
        return self.head(x)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class NativeRecurrentStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.x = torch.randn(2, 5, 4, device='cuda')

    def sim(self, gru_enabled, lstm_enabled):
        sim = QuantizationSimModel(prepare_model(RecurrentPair().cuda().eval()), self.x,
                                    quant_scheme='tf', config_file=str(CONFIG))
        config = self.directory / 'stage-config.json'
        config.write_text(json.dumps({
            'GRU_config': {'default_config': {'disable_quantization': not gru_enabled}},
            'LSTM_config': {'use_quantization': lstm_enabled},
        }))
        apply_mixed_precision_bitwidth(sim.model, str(config), verbose=False)
        self.assertEqual(sim.model.gru.use_quantization, gru_enabled)
        self.assertEqual(sim.model.lstm.use_quantization, lstm_enabled)
        return sim

    def assert_native_entries(self, encodings, flags):
        for name, enabled in flags.items():
            self.assertEqual(name in encodings.get('activation_encodings', {}), enabled)
            params = [key for key in encodings.get('param_encodings', {}) if key.startswith(name + '.')]
            self.assertEqual(bool(params), enabled)

    def test_disabled_stage_roundtrip_and_onnx(self):
        for flags in ({'gru': False, 'lstm': False}, {'gru': True, 'lstm': False},
                      {'gru': False, 'lstm': True}):
            for disable_after_calibration in (False, True):
                with self.subTest(flags=flags, disable_after_calibration=disable_after_calibration):
                    sim = self.sim(*(True if disable_after_calibration else enabled for enabled in flags.values()))
                    with torch.no_grad(), compute_encodings(sim.model):
                        sim.model(self.x)
                    for name, enabled in flags.items():
                        module = getattr(sim.model, name)
                        self.assertTrue(module.is_calibrated())
                        module.use_quantization = enabled
                    with torch.no_grad():
                        expected = sim.model(self.x)
                    stage = self.directory / 'stage.json'
                    saved = save_quantizer_encodings(sim.model, str(stage), verbose=False)
                    self.assert_native_entries(saved, flags)
                    fresh = self.sim(*flags.values())
                    fresh.model.load_state_dict(sim.model.state_dict())
                    load_quantizer_encodings(fresh.model, str(stage), skip_if_not_found=False,
                                             allow_overwrite=False, verbose=False)
                    for name, enabled in flags.items():
                        self.assertEqual(getattr(fresh.model, name).use_quantization, enabled)
                    with torch.no_grad():
                        torch.testing.assert_close(fresh.model(self.x), expected, rtol=1e-6, atol=1e-7)
                    graph_path, enc_path = export_onnx_json(
                        sim, str(self.directory), 'pair', tuple(self.x.shape))
                    graph = onnx.load(graph_path)
                    onnx.checker.check_model(graph)
                    for name in flags:
                        self.assertEqual(sum(node.name == name and node.op_type == name.upper()
                                             for node in graph.graph.node), 1)
                    self.assert_native_entries(json.loads(Path(enc_path).read_text()), flags)

    def test_strict_reload_requires_encodings_only_for_enabled_modules(self):
        path = self.directory / 'empty.json'
        path.write_text('{}')
        for name in ('gru', 'lstm'):
            with self.subTest(name=name):
                sim = self.sim(name == 'gru', name == 'lstm')
                with self.assertRaisesRegex(KeyError, name):
                    load_quantizer_encodings(sim.model, str(path), skip_if_not_found=False, verbose=False)

    def test_disabled_uncalibrated_layers_need_no_encodings(self):
        sim = self.sim(False, False)
        path = self.directory / 'disabled.json'
        saved = save_quantizer_encodings(sim.model, str(path), verbose=False)
        self.assert_native_entries(saved, {'gru': False, 'lstm': False})
        load_quantizer_encodings(sim.model, str(path), skip_if_not_found=False, verbose=False)
        for name in ('gru', 'lstm'):
            module = getattr(sim.model, name)
            self.assertFalse(module.use_quantization)
            self.assertFalse(module.is_calibrated())

    def test_valid_saved_encodings_still_enable_fresh_modules(self):
        sim = self.sim(True, True)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        path = self.directory / 'enabled.json'
        save_quantizer_encodings(sim.model, str(path), verbose=False)
        fresh = self.sim(False, False)
        fresh.model.load_state_dict(sim.model.state_dict())
        load_quantizer_encodings(fresh.model, str(path), skip_if_not_found=False, verbose=False)
        self.assertTrue(fresh.model.gru.use_quantization)
        self.assertTrue(fresh.model.lstm.use_quantization)
        with torch.no_grad():
            torch.testing.assert_close(fresh.model(self.x), sim.model(self.x))

    def test_gru_asymmetric_pot2_matches_direct_calibration(self):
        for bits in (8, 16):
            for unsigned in (False, True):
                with self.subTest(bits=bits, unsigned=unsigned):
                    config = self.directory / 'gru.json'
                    config.write_text(json.dumps({'GRU_config': {'operator_config': {
                        'input': {'bitwidth': bits, 'is_symmetric': False, 'is_unsigned': unsigned}}}}))
                    def make(pot2):
                        module = QuantGRU(1, 1, bidirectional=True, use_pot2_scale=pot2).cuda()
                        module.load_bitwidth_config(str(config))
                        return module
                    affine, native = make(False), make(True)
                    native.load_state_dict(affine.state_dict())
                    x = torch.tensor([-1., 9.], device='cuda').reshape(2, 1, 1)
                    for module in (affine, native):
                        with module.calibration_context():
                            module(x)
                    before = affine.export_quant_params_to_aimet_format({}, 'gru')['activation_encodings']['gru']['input'][0]
                    affine.enable_pot2()
                    actual = affine.export_quant_params_to_aimet_format({}, 'gru')['activation_encodings']['gru']
                    expected = native.export_quant_params_to_aimet_format({}, 'gru')['activation_encodings']['gru']
                    enc = actual['input'][0]
                    self.assertEqual(enc['bitwidth'], bits)
                    self.assertEqual(enc['is_symmetric'], 'False')
                    self.assertEqual(enc['dtype'], f"{'U' if unsigned else ''}INT{bits}")
                    self.assertEqual(math.log2(enc['scale']), round(math.log2(enc['scale'])))
                    self.assertNotEqual(enc['zero_point'], before['zero_point'])
                    self.assertEqual(actual, expected)
                    affine.use_quantization = True
                    self.assertTrue(torch.isfinite(affine(x)[0]).all())


if __name__ == '__main__':
    unittest.main()
