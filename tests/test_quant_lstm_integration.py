"""Real CUDA integration coverage; run in the rx-met build/runtime environment."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from quant_lstm import QuantLSTM
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow
from aimet_torch.staged_quantization_utils import save_quantizer_encodings, load_quantizer_encodings
from aimet_torch.rx_export.export_onnx_json import export_onnx_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'


def assert_recurrent_encoding_schema(test, layer):
    """One schema check applied to real QuantGRU and QuantLSTM exports."""
    fields = {'dtype', 'bitwidth', 'is_symmetric', 'enc_type',
              'scale', 'zero_point', 'real_min', 'real_max'}
    records = []
    for kind in ('input', 'output'):
        test.assertIsInstance(layer[kind], list)
        records.extend(layer[kind])
    for kind in ('internal_ops', 'internal_ops_reverse'):
        for operation in layer.get(kind, {}).values():
            test.assertIn('output', operation)
            test.assertIsInstance(operation['output'], list)
            records.extend(operation['output'])
    for record in records:
        test.assertEqual(set(record), fields)
        test.assertIsInstance(record['bitwidth'], int)
        test.assertIn(record['is_symmetric'], ('True', 'False'))
        for field in ('scale', 'zero_point', 'real_min', 'real_max'):
            values = record[field] if isinstance(record[field], list) else [record[field]]
            test.assertTrue(all(isinstance(value, (int, float)) for value in values))


class Model(nn.Module):
    def __init__(self, bidirectional=False, bias=True):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.lstm = QuantLSTM(4, 3, batch_first=True, bidirectional=bidirectional, bias=bias)
        self.head = nn.Linear(6 if bidirectional else 3, 2)

    def forward(self, x):
        sequence, _ = self.lstm(self.proj(x))
        return self.head(sequence)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class QuantLSTMIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.x = torch.randn(2, 5, 4, device='cuda')

    def sim(self, bidirectional=False, bias=True, scheme='tf'):
        model = prepare_model(Model(bidirectional, bias).cuda().eval())
        self.assertTrue(any(n.op == 'call_module' and n.target == 'lstm' for n in model.graph.nodes))
        sim = QuantizationSimModel(model, self.x, quant_scheme=scheme, config_file=str(CONFIG))
        self.assertIsInstance(sim.model.lstm, QuantLSTM)
        self.assertTrue(all(q is None for q in sim.model.lstm.param_quantizers.values()))
        self.assertTrue(all(q is None for q in sim.model.lstm.input_quantizers))
        self.assertTrue(all(q is None for q in sim.model.lstm.output_quantizers))
        path = self.directory / 'mixed.json'
        path.write_text(json.dumps({'LSTM_config': {'use_quantization': True,
            'quant_config': {'schema_version': 1, 'operators': {'cell_state': {'bitwidth': 16}}}}}))
        stats = apply_mixed_precision_bitwidth(sim.model, str(path), verbose=False)
        self.assertEqual(stats['quant_lstm_enabled'], 1)
        self.assertEqual(sim.model.lstm.get_quant_config('cell_state')['bitwidth'], 16)
        return sim

    def calibrate(self, sim):
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
            sim.model(self.x * 0.7)
        self.assertTrue(sim.model.lstm.is_calibrated())
        self.assertFalse(sim.model.lstm.calibrating)

    def test_ptq_qat_and_copy(self):
        for bidirectional in (False, True):
            with self.subTest(bidirectional=bidirectional):
                sim = self.sim(bidirectional)
                self.calibrate(sim)
                document = sim.model.lstm.export_quant_params()
                clone = copy.deepcopy(sim.model)
                torch.testing.assert_close(clone(self.x), sim.model(self.x))
                sim.model.train()
                value = self.x.clone().requires_grad_()
                sim.model(value).square().mean().backward()
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertIsNotNone(sim.model.lstm.weight_ih_l0.grad)
                self.assertEqual(document, sim.model.lstm.export_quant_params())

    def test_calibration_error_restores_mode(self):
        sim = self.sim()
        with self.assertRaisesRegex(ValueError, 'callback failed'):
            with compute_encodings(sim.model):
                sim.model(self.x)
                raise ValueError('callback failed')
        self.assertFalse(sim.model.lstm.calibrating)
        self.calibrate(sim)

    def test_percentile_and_reload_locked(self):
        sim = self.sim(True, scheme='percentile')
        sim.set_percentile_value(99.0)
        self.assertEqual(sim.model.lstm.percentile_value, 99.0)
        self.calibrate(sim)
        path = self.directory / 'stage.json'
        saved = save_quantizer_encodings(sim.model, str(path), verbose=False)
        fresh = self.sim(True)
        fresh.model.load_state_dict(sim.model.state_dict())
        loaded = load_quantizer_encodings(fresh.model, str(path), allow_overwrite=False, verbose=False)
        self.assertIn('QuantLSTM', loaded['loaded_types'])
        before = fresh.model.lstm.export_quant_params()
        with compute_encodings(fresh.model):
            fresh.model(self.x * 10)
        self.assertEqual(before, fresh.model.lstm.export_quant_params())
        self.assertIn('lstm', saved['quant_lstm_encodings'])
        torch.testing.assert_close(fresh.model(self.x), sim.model(self.x))
        self.assertIn('proj.input_quantizers.0', saved)

    def test_pot2_and_config_guard(self):
        sim = self.sim(True)
        self.calibrate(sim)
        apply_power_of_2_workflow(sim.model, verbose=False)
        self.assertTrue(sim.model.lstm.is_calibrated())
        for field in ('operators', 'operators_reverse'):
            for enc in sim.model.lstm.export_quant_params()[field].values():
                scales = torch.as_tensor(enc['scale'])
                torch.testing.assert_close(scales.log2(), scales.log2().round())
        self.assertTrue(torch.isfinite(sim.model(self.x)).all())
        with self.assertRaisesRegex(RuntimeError, 'Reset'):
            sim.model.lstm.load_quant_config({'schema_version': 1})

    def test_shared_gru_lstm_calibration_and_export(self):
        from quant_gru import QuantGRU
        import onnx

        class RecurrentPair(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = QuantGRU(4, 4, batch_first=True)
                self.lstm = QuantLSTM(4, 3, batch_first=True)

            def forward(self, x):
                x, _ = self.gru(x)
                x, _ = self.lstm(x)
                return x

        sim = QuantizationSimModel(prepare_model(RecurrentPair().cuda()), self.x,
                                   quant_scheme='tf', config_file=str(CONFIG))
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        for module in (sim.model.gru, sim.model.lstm):
            self.assertTrue(module.is_calibrated())
            module.use_quantization = True
        self.assertTrue(torch.isfinite(sim.model(self.x)).all())
        path, enc_path = export_onnx_json(sim, str(self.directory), 'pair', tuple(self.x.shape))
        graph = onnx.load(path)
        self.assertEqual(sum(n.op_type == 'GRU' for n in graph.graph.node), 1)
        self.assertEqual(sum(n.op_type == 'LSTM' for n in graph.graph.node), 1)
        enc = json.loads(Path(enc_path).read_text())
        self.assertIn('gru', enc['activation_encodings'])
        self.assertIn('lstm', enc['quant_lstm_encodings'])
        self.assertEqual(enc['schema_version'], 3)
        for name in ('gru', 'lstm'):
            assert_recurrent_encoding_schema(self, enc['activation_encodings'][name])
        self.assertEqual(set(enc['activation_encodings']['gru']['input'][0]),
                         set(enc['activation_encodings']['lstm']['input'][0]))
        for record in enc['param_encodings'].values():
            self.assertIn('bitwidth', record)
            self.assertIn('is_symmetric', record)
            self.assertNotIn('symmetric', record)
            if isinstance(record['scale'], list):
                self.assertTrue(all(isinstance(value, (int, float)) for value in record['scale']))

    def test_shared_lifecycle_failure_locks_and_stage_filter(self):
        from quant_gru import QuantGRU

        class RecurrentPair(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.gru = QuantGRU(4, 4, batch_first=True)
                self.lstm = QuantLSTM(4, 3, batch_first=True)

            def forward(self, x):
                x, _ = self.gru(self.proj(x))
                x, _ = self.lstm(x)
                return x

        def make_sim():
            return QuantizationSimModel(prepare_model(RecurrentPair().cuda()), self.x,
                                        quant_scheme='tf', config_file=str(CONFIG))

        sim = make_sim()
        with self.assertRaisesRegex(ValueError, 'callback failed'):
            with compute_encodings(sim.model):
                sim.model(self.x)
                raise ValueError('callback failed')
        for module in (sim.model.gru, sim.model.lstm):
            self.assertFalse(module.calibrating)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        for module in (sim.model.gru, sim.model.lstm):
            module.use_quantization = True
        path = self.directory / 'pair-stage.json'
        saved = save_quantizer_encodings(sim.model, str(path), verbose=False)
        self.assertIn('gru', saved['activation_encodings'])
        self.assertIn('lstm', saved['quant_lstm_encodings'])
        fresh = make_sim()
        fresh.model.load_state_dict(sim.model.state_dict())
        loaded = load_quantizer_encodings(fresh.model, str(path), allow_overwrite=False, verbose=False)
        self.assertTrue({'QuantGRU', 'QuantLSTM'}.issubset(loaded['loaded_types']))
        torch.testing.assert_close(fresh.model(self.x), sim.model(self.x))
        before = save_quantizer_encodings(fresh.model, str(path), verbose=False)
        with compute_encodings(fresh.model):
            fresh.model(self.x * 4)
        after = save_quantizer_encodings(fresh.model, str(path), verbose=False)
        self.assertEqual(before, after)
        filtered = make_sim()
        load_quantizer_encodings(filtered.model, str(path), layer_types=['QuantLSTM'], verbose=False)
        self.assertTrue(filtered.model.lstm.is_calibrated())
        self.assertFalse(filtered.model.gru.is_calibrated())
        self.assertFalse(filtered.model.proj.input_quantizers[0].is_initialized())
        with compute_encodings(filtered.model):
            pass  # Unexecuted branches need no calibration finalization.
        self.assertFalse(filtered.model.gru.is_calibrated())
        self.assertFalse(filtered.model.lstm.is_calibrated())

    def test_explicit_states_config_validation_and_reset(self):
        module = QuantLSTM(4, 3, batch_first=True, bidirectional=True).cuda()
        h = torch.randn(2, 2, 3, device='cuda', requires_grad=True)
        c = torch.randn_like(h, requires_grad=True)
        with torch.no_grad(), module.calibration_context():
            module(self.x, (h, c))
        module.use_quantization = True
        y, (hn, cn) = module(self.x, (h, c))
        (y.sum() + hn.sum() + cn.sum()).backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(c.grad).all())
        module.set_quant_params_locked()
        module.reset_calibration()
        self.assertFalse(module.quant_params_locked())
        with self.assertRaises(ValueError):
            module.load_bitwidth_config({'LSTM_config': {'use_quantization': 'false'}})
        with self.assertRaises(ValueError):
            module.load_bitwidth_config({'LSTM_config': {'unknown': 1}})
        self.assertTrue(module.use_quantization)

    def test_onnx_encodings_roundtrip(self):
        import onnx
        import onnxruntime as ort
        for bidirectional, bias in ((False, True), (True, True), (True, False)):
            with self.subTest(bidirectional=bidirectional, bias=bias):
                sim = self.sim(bidirectional, bias)
                self.calibrate(sim)
                quantized_output = sim.model(self.x).detach()
                path, enc_path = export_onnx_json(sim, str(self.directory), 'lstm_model', tuple(self.x.shape))
                graph = onnx.load(path)
                onnx.checker.check_model(graph)
                nodes = [n for n in graph.graph.node if n.op_type == 'LSTM']
                self.assertEqual(len(nodes), 1)
                self.assertEqual(nodes[0].name, 'lstm')
                self.assertFalse(any(n.op_type in ('QuantizeLinear','DequantizeLinear') for n in graph.graph.node))
                enc = json.loads(Path(enc_path).read_text())
                self.assertTrue(enc['activation_encodings']['lstm']['is_LSTM'])
                self.assertEqual(enc['schema_version'], 3)
                assert_recurrent_encoding_schema(self, enc['activation_encodings']['lstm'])
                self.assertEqual(len(enc['param_encodings']['lstm.weight_ih.weight']['scale']),
                                 (2 if bidirectional else 1) * 4 * sim.model.lstm.hidden_size)
                # Check deployment encoding lengths against actual ONNX W/R/B,
                # independently of quant_lstm_encodings used by native reload.
                initializers = {item.name: item for item in graph.graph.initializer}
                for suffix in ('weight_ih.weight', 'weight_hh.weight', 'bias'):
                    if suffix == 'bias' and not bias:
                        continue
                    key = 'lstm.' + suffix
                    tensor = initializers[key]
                    record = enc['param_encodings'][key]
                    self.assertEqual(record['enc_type'], 'PER_CHANNEL')
                    for field in ('scale', 'zero_point', 'real_min', 'real_max'):
                        self.assertEqual(len(record[field]), tensor.dims[0] * tensor.dims[1])
                        self.assertTrue(all(isinstance(value, (int, float)) for value in record[field]))
                fresh = self.sim(bidirectional, bias)
                fresh.model.load_state_dict(sim.model.state_dict())
                load_quantizer_encodings(fresh.model, enc_path, allow_overwrite=False, verbose=False)
                torch.testing.assert_close(fresh.model(self.x), quantized_output)
                float_model = sim.get_original_model(sim.model, qdq_weights=False).cuda().eval()
                float_model.lstm.use_quantization = False
                reference = float_model(self.x).detach().cpu().numpy()
                actual = ort.InferenceSession(path, providers=['CPUExecutionProvider']).run(None, {'input': self.x.cpu().numpy()})[0]
                torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(reference), atol=2e-5, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
