"""Native recurrent encoding readers preserve grids when optional n is absent."""

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from aimet_onnx.rx_ptq import pipeline
from aimet_onnx.rx_ptq.po2 import apply_power_of_2_workflow
from aimet_onnx.rx_ptq.recurrent import restore_native_encodings
from tests.support.paths import REPO_ROOT
from tests.support.recurrent import gru_graph, lstm_graph


CONFIG = REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'
PROVIDERS = ('CUDAExecutionProvider', 'CPUExecutionProvider')


def without_n(value):
    if isinstance(value, dict):
        return {key: without_n(child) for key, child in value.items() if key != 'n'}
    if isinstance(value, list):
        return [without_n(child) for child in value]
    return copy.deepcopy(value)


@unittest.skipUnless(torch.cuda.is_available(), 'native recurrent compatibility requires CUDA')
class NativeEncodingCompatibilityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='native-encoding-compat-')
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.index = 0

    def sim(self, model, feed):
        self.index += 1
        return pipeline.create_onnx_ptq_sim(copy.deepcopy(model), feed,
            config_file=str(CONFIG), quant_scheme='tf', providers=PROVIDERS,
            work_dir=self.path / str(self.index))

    def calibrated(self, kind):
        model, feed = (gru_graph() if kind == 'GRU' else lstm_graph())
        sim = self.sim(model, feed)
        sim.compute_encodings([feed, {name: value * 1.5 for name, value in feed.items()}])
        apply_power_of_2_workflow(sim, verbose=False)
        adapter = next(iter(sim._rx_native_recurrent.values()))
        return sim, adapter, model, feed

    def test_torch_gru_and_lstm_accept_missing_derived_exponents(self):
        for kind in ('GRU', 'LSTM'):
            with self.subTest(kind=kind):
                source, adapter, model, feed = self.calibrated(kind)
                payload = adapter.module.export_quant_params_to_aimet_format({}, module_name=adapter.name)
                legacy = without_n(payload)
                target = self.sim(model, feed)
                restored = next(iter(target._rx_native_recurrent.values()))
                self.assertTrue(restored.module.load_quant_params_from_aimet_format(
                    legacy, module_name=adapter.name, verbose=False))
                self.assertEqual(restored.module.export_quant_params_to_aimet_format({}, module_name=adapter.name),
                                 payload)
                for expected, actual in zip(source.session.run(None, feed), target.session.run(None, feed)):
                    np.testing.assert_array_equal(actual, expected)

    def test_onnx_gru_and_lstm_accept_missing_derived_exponents(self):
        for kind in ('GRU', 'LSTM'):
            with self.subTest(kind=kind):
                source, adapter, model, feed = self.calibrated(kind)
                payload = adapter.export_encodings()
                self.assertNotIn('schema_version', payload)
                legacy = without_n(payload)
                target = self.sim(model, feed)
                restore_native_encodings(target, {'native_recurrent_encodings': legacy}, allow_overwrite=False)
                restored = next(iter(target._rx_native_recurrent.values()))
                self.assertEqual(restored.export_encodings(), payload)
                self.assertEqual(legacy, without_n(payload))
                for expected, actual in zip(source.session.run(None, feed), target.session.run(None, feed)):
                    np.testing.assert_array_equal(actual, expected)

    def test_lstm_existing_single_record_port_lists_accept_missing_n(self):
        source, adapter, model, feed = self.calibrated('LSTM')
        payload = adapter.export_encodings()
        legacy = without_n(payload)
        entry = legacy['activation_encodings'][adapter.name]
        for kind in ('input', 'output'):
            entry[kind] = [entry[kind]['0']]
        before = copy.deepcopy(legacy)
        target = self.sim(model, feed)
        restore_native_encodings(target, {'native_recurrent_encodings': legacy}, allow_overwrite=False)
        restored = next(iter(target._rx_native_recurrent.values()))
        self.assertEqual(restored.export_encodings(), payload)
        self.assertEqual(legacy, before)
        for expected, actual in zip(source.session.run(None, feed), target.session.run(None, feed)):
            np.testing.assert_array_equal(actual, expected)

    def test_lstm_invalid_records_are_rejected_without_changing_loaded_state(self):
        _source, adapter, _model, _feed = self.calibrated('LSTM')
        payload = adapter.export_encodings()
        for kind in ('explicit_n', 'parameter_length', 'invalid_scale', 'extra_record_field', 'extra_entry_field'):
            with self.subTest(kind=kind):
                adapter.module.set_quant_params_locked(False)
                before = adapter.module.export_quant_params()
                flags = (adapter.module.use_quantization, adapter.module.quant_params_locked(),
                         adapter.module.calibrating)
                weights = {name: value.detach().clone() for name, value in adapter.module.named_parameters()}
                invalid = copy.deepcopy(payload)
                entry = invalid['activation_encodings'][adapter.name]
                if kind == 'explicit_n':
                    entry['input']['0']['n'] += 1
                elif kind == 'parameter_length':
                    invalid['param_encodings']['custom_W']['zero_point'].pop()
                elif kind == 'invalid_scale':
                    invalid['param_encodings']['custom_W']['scale'][0] = 0.
                elif kind == 'extra_record_field':
                    entry['input']['0']['unexpected'] = True
                else:
                    entry['unexpected'] = True
                with self.assertRaises(ValueError):
                    adapter.load_encodings(invalid, allow_overwrite=False)
                self.assertEqual(adapter.module.export_quant_params(), before)
                self.assertEqual((adapter.module.use_quantization, adapter.module.quant_params_locked(),
                                  adapter.module.calibrating), flags)
                for name, value in adapter.module.named_parameters():
                    self.assertTrue(torch.equal(value, weights[name]), name)


if __name__ == '__main__':
    unittest.main()
