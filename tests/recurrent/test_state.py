"""Shared GRU/LSTM quantization-state and Po2 regression tests (CUDA)."""

import copy
import json
import importlib
from unittest.mock import patch
import math
import tempfile
import unittest
from pathlib import Path
import onnx
import numpy as np
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

from tests.support.paths import REPO_ROOT


CONFIG = REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'


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

    def test_default_and_sparse_configs_export_symmetric_grids(self):
        cases = {
            'constructor': None,
            'empty': {},
            'empty_sections': {'GRU_config': {}, 'LSTM_config': {}},
            'sparse': {
                'GRU_config': {'operator_config': {'input': {'bitwidth': 16}}},
                'LSTM_config': {'quant_config': {
                    'schema_version': 1, 'operators': {'input': {'bitwidth': 16}}}},
            },
        }
        for cls in (QuantGRU, QuantLSTM):
            for bidirectional in (False, True):
                for case, config in cases.items():
                    with self.subTest(operator=cls.__name__, bidirectional=bidirectional, case=case):
                        module = cls(4, 3, batch_first=True, bidirectional=bidirectional).cuda()
                        if config is not None:
                            path = self.directory / 'defaults.json'
                            path.write_text(json.dumps(config))
                            module.load_bitwidth_config(str(path))
                        with torch.no_grad(), module.calibration_context():
                            module(self.x + 2)
                        document = module.export_quant_params_to_aimet_format({}, module_name='rnn')
                        layer = document['activation_encodings']['rnn']
                        records = list(layer['input'].values()) + list(layer['output'].values())
                        self.assertEqual(layer['input']['0']['bitwidth'], 16 if case == 'sparse' else 8)
                        for direction in ('internal_ops', 'internal_ops_reverse'):
                            for name, operation in layer.get(direction, {}).items():
                                records.extend(operation['output'])
                                if name in ('update_gate_output', 'reset_gate_output',
                                            'input_gate_output', 'forget_gate_output', 'output_gate_output'):
                                    self.assertEqual(operation['output'][0]['dtype'], 'UINT8')
                        records.extend(document['param_encodings'].values())
                        for record in records:
                            self.assertEqual(record['is_symmetric'], 'True')
                            zero_points = record['zero_point']
                            if not isinstance(zero_points, list):
                                zero_points = [zero_points]
                            self.assertTrue(zero_points)
                            self.assertTrue(all(value == 0 for value in zero_points))

    def test_explicit_asymmetric_config_is_preserved(self):
        path = self.directory / 'asymmetric.json'
        path.write_text(json.dumps({
            'GRU_config': {'operator_config': {'input': {'is_symmetric': False}}},
            'LSTM_config': {'quant_config': {
                'schema_version': 1, 'operators': {'input': {'is_symmetric': False}}}},
        }))
        for cls in (QuantGRU, QuantLSTM):
            with self.subTest(operator=cls.__name__):
                module = cls(4, 3, batch_first=True).cuda()
                module.load_bitwidth_config(str(path))
                with torch.no_grad(), module.calibration_context():
                    module(self.x + 2)
                document = module.export_quant_params_to_aimet_format({}, module_name='rnn')
                layer = document['activation_encodings']['rnn']
                from aimet_common.compiler_encodings import validate_compiler_encodings
                validate_compiler_encodings(document)
                self.assertEqual(layer['input']['0']['is_symmetric'], 'False')
                self.assertEqual(layer['output']['0']['is_symmetric'], 'True')

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
                    document = json.loads(Path(enc_path).read_text())
                    self.assert_native_entries(document, flags)
                    excluded = set(document.get('excluded_layers') or [])
                    for name, enabled in flags.items():
                        self.assertEqual(name in excluded, not enabled)
                    if not any(flags.values()):
                        import onnxruntime as ort
                        floating = sim.get_original_model(sim.model, qdq_weights=False).cpu().eval()
                        for name, reference_type in (('gru', nn.GRU), ('lstm', nn.LSTM)):
                            native = getattr(floating, name)
                            reference_layer = reference_type(
                                native.input_size, native.hidden_size, num_layers=native.num_layers,
                                bias=native.bias, batch_first=native.batch_first,
                                bidirectional=native.bidirectional,
                            ).eval()
                            with torch.no_grad():
                                for param_name, parameter in reference_layer.named_parameters():
                                    parameter.copy_(getattr(native, param_name).detach().cpu())
                            setattr(floating, name, reference_layer)
                        with torch.no_grad():
                            reference = floating(self.x.cpu()).numpy()
                        session = ort.InferenceSession(graph_path, providers=['CPUExecutionProvider'])
                        actual = session.run(None, {session.get_inputs()[0].name: self.x.cpu().numpy()})[0]
                        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)

    def corrupt_recurrent_export(self, mutate):
        exporter = importlib.import_module('aimet_torch.rx_export.export_onnx_json')
        original = exporter.export_recurrent_encodings
        def corrupt(model, graph_path, enc_path):
            original(model, graph_path, enc_path)
            mutate(Path(graph_path), Path(enc_path))
        return patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt)

    def test_disabled_uncalibrated_recurrent_layers_can_export(self):
        sim = self.sim(False, False)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        for name in ('gru', 'lstm'):
            getattr(sim.model, name).reset_calibration()
            self.assertFalse(getattr(sim.model, name).is_calibrated())
        _, enc_path = export_onnx_json(sim, str(self.directory), 'uncalibrated', tuple(self.x.shape))
        document = json.loads(Path(enc_path).read_text())
        self.assert_native_entries(document, {'gru': False, 'lstm': False})
        self.assertTrue({'gru', 'lstm'} <= set(document['excluded_layers']))

    def test_disabled_layer_requires_exclusion_metadata(self):
        sim = self.sim(False, True)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        def remove(_graph_path, path):
            document = json.loads(path.read_text())
            document['excluded_layers'].remove('gru')
            path.write_text(json.dumps(document))
        with self.corrupt_recurrent_export(remove), self.assertRaisesRegex(RuntimeError, 'gru.*excluded_layers'):
            export_onnx_json(sim, str(self.directory), 'missing-exclusion', tuple(self.x.shape))

    def test_exclusion_metadata_cannot_disable_an_enabled_layer(self):
        for remove_encoding in (False, True):
            with self.subTest(remove_encoding=remove_encoding):
                sim = self.sim(True, True)
                with torch.no_grad(), compute_encodings(sim.model):
                    sim.model(self.x)
                def corrupt(_graph_path, path):
                    document = json.loads(path.read_text())
                    document['excluded_layers'] = ['gru']
                    if remove_encoding:
                        del document['activation_encodings']['gru']
                    path.write_text(json.dumps(document))
                with self.corrupt_recurrent_export(corrupt), self.assertRaisesRegex(RuntimeError, 'gru'):
                    export_onnx_json(sim, str(self.directory), 'fake-exclusion', tuple(self.x.shape))

    def test_disabled_layer_cannot_export_stale_quantization_encodings(self):
        sim = self.sim(True, True)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        from aimet_torch.rx_export.validation import collect_expected_encodings
        stale = collect_expected_encodings(sim.model)['activation_encodings']['gru']
        sim.model.gru.use_quantization = False
        def corrupt(_graph_path, path):
            document = json.loads(path.read_text())
            document['activation_encodings']['gru'] = stale
            path.write_text(json.dumps(document))
        with self.corrupt_recurrent_export(corrupt), self.assertRaisesRegex(RuntimeError, 'gru.*has quantization encodings'):
            export_onnx_json(sim, str(self.directory), 'stale-disabled', tuple(self.x.shape))

    def test_disabled_layer_still_requires_its_named_onnx_operator(self):
        sim = self.sim(False, False)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        def rename(path, _enc_path):
            graph = onnx.load(path)
            for node in graph.graph.node:
                if node.name == 'gru' and node.op_type == 'GRU':
                    node.name = 'renamed_gru'
            onnx.save(graph, path)
        with self.corrupt_recurrent_export(rename), self.assertRaisesRegex(RuntimeError, 'gru: expected one ONNX GRU'):
            export_onnx_json(sim, str(self.directory), 'renamed-disabled', tuple(self.x.shape))

    def test_export_rejects_missing_native_encodings(self):
        exporter = importlib.import_module('aimet_torch.rx_export.export_onnx_json')
        original = exporter.export_recurrent_encodings
        sim = self.sim(True, True)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        for name in ('gru', 'lstm'):
            for kind in ('internal_ops', 'weight'):
                with self.subTest(name=name, kind=kind):
                    def corrupt(model, graph_path, enc_path):
                        original(model, graph_path, enc_path)
                        path = Path(enc_path)
                        document = json.loads(path.read_text())
                        if kind == 'internal_ops':
                            del document['activation_encodings'][name]['internal_ops']
                        else:
                            del document['param_encodings'][name + '.weight_ih.weight']
                        path.write_text(json.dumps(document))
                    with patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt), \
                         self.assertRaisesRegex(RuntimeError, name):
                        export_onnx_json(sim, str(self.directory), 'missing', tuple(self.x.shape))

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
                    before = affine.export_quant_params_to_aimet_format({}, 'gru')['activation_encodings']['gru']['input']['0']
                    affine.enable_pot2()
                    actual = affine.export_quant_params_to_aimet_format({}, 'gru')['activation_encodings']['gru']
                    expected = native.export_quant_params_to_aimet_format({}, 'gru')['activation_encodings']['gru']
                    enc = actual['input']['0']
                    self.assertEqual(enc['bitwidth'], bits)
                    self.assertEqual(enc['is_symmetric'], 'False')
                    self.assertEqual(enc['dtype'], f"{'U' if unsigned else ''}INT{bits}")
                    self.assertEqual(math.log2(enc['scale']), round(math.log2(enc['scale'])))
                    self.assertNotEqual(enc['zero_point'], before['zero_point'])
                    self.assertEqual(actual, expected)
                    affine.use_quantization = True
                    self.assertTrue(torch.isfinite(affine(x)[0]).all())

    def test_public_schema_legacy_reload_and_invalid_exponents(self):
        def records(value):
            if isinstance(value, dict):
                if 'scale' in value:
                    yield value
                else:
                    for child in value.values():
                        yield from records(child)
            elif isinstance(value, list):
                for child in value:
                    yield from records(child)

        for module_type in (QuantGRU, QuantLSTM):
            for pot2 in (False, True):
                with self.subTest(module=module_type.__name__, pot2=pot2):
                    module = module_type(4, 3, batch_first=True, bidirectional=True).cuda()
                    with torch.no_grad(), module.calibration_context():
                        module(self.x)
                    if pot2:
                        module.enable_pot2()
                    public = module.export_quant_params_to_aimet_format({}, 'rnn')
                    layer = public['activation_encodings']['rnn']
                    self.assertEqual(set(layer['input']), {'0'})
                    self.assertEqual(set(layer['output']), {'0'})
                    for record in records(public):
                        self.assertIn('n', record)
                        self.assertIn('enc_type', record)
                        scales = record['scale'] if isinstance(record['scale'], list) else [record['scale']]
                        exponents = record['n'] if isinstance(record['n'], list) else [record['n']]
                        self.assertEqual(exponents, [round(-math.log2(value)) for value in scales])

                    fresh = module_type(4, 3, batch_first=True, bidirectional=True).cuda()
                    fresh.load_state_dict(module.state_dict())
                    fresh.load_quant_params_from_aimet_format(public, 'rnn')
                    before = fresh.export_quant_params_to_aimet_format({}, 'rnn')
                    legacy = copy.deepcopy(public)
                    for record in records(legacy):
                        record.pop('n')
                    old_layer = legacy['activation_encodings']['rnn']
                    for kind in ('input', 'output'):
                        old_layer[kind] = [old_layer[kind]['0']]
                    fresh.load_quant_params_from_aimet_format(legacy, 'rnn')
                    self.assertEqual(fresh.export_quant_params_to_aimet_format({}, 'rnn'), before)
                    module.use_quantization = fresh.use_quantization = True
                    with torch.no_grad():
                        torch.testing.assert_close(fresh(self.x), module(self.x))

                    for location in ('input', 'internal_ops_reverse', 'parameter'):
                        with self.subTest(location=location):
                            invalid = copy.deepcopy(public)
                            if location == 'parameter':
                                record = invalid['param_encodings']['rnn.weight_ih.weight']
                            elif location == 'input':
                                record = invalid['activation_encodings']['rnn']['input']['0']
                            else:
                                record = next(records(invalid['activation_encodings']['rnn'][location]))
                            record['n'] = 999
                            with self.assertRaisesRegex(ValueError, 'n.*scale'):
                                fresh.load_quant_params_from_aimet_format(invalid, 'rnn')
                            self.assertEqual(fresh.export_quant_params_to_aimet_format({}, 'rnn'), before)


if __name__ == "__main__":
    unittest.main()
