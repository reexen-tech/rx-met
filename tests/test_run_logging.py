"""Detailed file output must not change quantization or console verbosity."""
import contextlib
import builtins
import io
import json
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from aimet_common import run_context as runtime
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow
from tests import test_onnx_ptq_alignment as fixtures


class RuntimeLoggingIntegrationTest(unittest.TestCase):
    def setUp(self):
        runtime.configure_output()
        self.addCleanup(runtime.configure_output)
        self.fixture = fixtures.OnnxAlignmentTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(runtime._finish_implicit)
        self.path = self.fixture.path

    def torch_sim(self):
        torch.manual_seed(41)
        sim = QuantizationSimModel(prepare_model(nn.Sequential(nn.Linear(4, 2)).eval()),
                                   torch.ones(1, 4), config_file=str(self.fixture.config), quant_scheme='tf')
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(torch.ones(1, 4))
            sim.model(torch.tensor([[-2., 1., 3., -1.]]))
        return sim

    def test_onnx_details_are_written_when_verbose_is_false(self):
        from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
        config = self.path / 'mixed.json'
        config.write_text(json.dumps({'layer_name_config': {'fc': {'output_bitwidth': 16}}}))
        console = io.StringIO()
        with contextlib.redirect_stdout(console), runtime.execution(self.path / 'onnx') as run:
            sim, _ = self.fixture.sim()
            report = apply_onnx_mixed_precision_bitwidth(sim, str(config), verbose=False)
        self.assertNotIn('应用混合精度位宽配置', console.getvalue())
        self.assertEqual(sim.qc_quantize_op_dict['y'].bitwidth, 16)
        self.assertEqual(report['name_matched'], 1)
        text = (run.root / 'run.log').read_text()
        self.assertIn('应用混合精度位宽配置', text)
        self.assertIn('[name] fc', text)
        self.assertTrue(list((run.root / 'config').glob('*.json')))

    def test_onnx_default_config_is_snapshotted_without_explicit_file(self):
        from aimet_common.quantsim_config.quantsim_config import _get_config_file
        from aimet_onnx.quantsim import QuantizationSimModel
        for config in (None, 'default'):
            with self.subTest(config=config), runtime.execution(self.path / 'onnx-default') as run:
                QuantizationSimModel(fixtures.make_graph(), dummy_input=self.fixture.feed,
                                      config_file=config, quant_scheme='tf')
            source = Path(_get_config_file(config))
            self.assertEqual(json.loads((run.root / 'config/quantsim.json').read_text()),
                             json.loads(source.read_text()))
            record = json.loads((run.root / 'run.json').read_text())
            self.assertEqual(record['config']['quantsim']['source'], str(source.resolve()))

    def test_torch_details_do_not_change_return_values_or_quantizer_settings(self):
        config = self.path / 'torch-mixed.json'
        config.write_text(json.dumps({'default_bitwidth': {'weight': 16, 'output': 16}}))
        observed = []
        for verbose in (False, True):
            console = io.StringIO()
            with contextlib.redirect_stdout(console), runtime.execution(self.path / 'torch') as run:
                sim = self.torch_sim()
                report = apply_mixed_precision_bitwidth(sim.model, str(config), verbose=verbose)
            layer = sim.model.get_submodule('0')
            observed.append((report, layer.param_quantizers['weight'].bitwidth,
                             layer.output_quantizers[0].bitwidth))
            self.assertIn('应用混合精度位宽配置', (run.root / 'run.log').read_text())
            if not verbose:
                self.assertNotIn('应用混合精度位宽配置', console.getvalue())
        self.assertEqual(observed[0], observed[1])

    def test_po2_file_comparison_preserves_quiet_return_contract_and_model_output(self):
        outputs = []
        for verbose in (False, True):
            console = io.StringIO()
            with contextlib.redirect_stdout(console), runtime.execution(self.path / 'po2') as run:
                sim = self.torch_sim()
                report = apply_power_of_2_workflow(sim.model, verbose=verbose)
                outputs.append(sim.model(torch.ones(1, 4)).detach())
            if not verbose:
                self.assertEqual(report['issues'], [])
                self.assertNotIn('Power-of-2 转换前后对比', console.getvalue())
            self.assertIn('Power-of-2 转换前后对比', (run.root / 'run.log').read_text())
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)

    def test_operator_configuration_is_logged_without_console_details(self):
        from quant_gru import QuantGRU
        from quant_lstm import QuantLSTM
        config = Path(__file__).resolve().parents[1] / 'examples/config/quick_start_full_quant.json'
        for kind in (QuantGRU, QuantLSTM):
            console = io.StringIO()
            with contextlib.redirect_stdout(console), runtime.execution(self.path / kind.__name__) as run:
                operator = kind(4, 2, batch_first=True)
                operator.load_bitwidth_config(str(config), verbose=False)
            text = (run.root / 'run.log').read_text()
            expected = 'GRU 量化配置' if kind is QuantGRU else '[QuantLSTM]'
            self.assertIn(expected, text)
            self.assertNotIn('configuration loaded', console.getvalue())
            self.assertNotIn('GRU 量化配置', console.getvalue())

    def test_complete_onnx_workflow_returns_grouped_outputs_and_format_versions(self):
        from aimet_onnx.rx_ptq.pipeline import run_onnx_ptq
        import onnx
        source = self.path / 'model.onnx'
        onnx.save(fixtures.make_graph(), source)
        calibration = self.path / 'calib'
        calibration.mkdir()
        np.save(calibration / 'sample.npy', self.fixture.feed['x'])
        root = self.path / 'workflow'
        result = run_onnx_ptq(model_path=source, calibration_dir=calibration,
                              output_dir=root, filename_prefix='model',
                              config_file=str(self.fixture.config), bitwidth_config_file='', quant_scheme='tf')
        directory = result['onnx'].parent.parent
        self.assertEqual(set(root.iterdir()), {directory})
        self.assertTrue(all(path.parent == directory / 'exports' for path in result.values()))
        record = json.loads((directory / 'run.json').read_text())
        encoding = json.loads(result['encodings'].read_text())
        self.assertEqual(record['outputs']['exports/model.encodings']['schema_version'], encoding['schema_version'])
        self.assertEqual(record['inputs']['model_path']['sha256'], runtime._sha256(source))
        self.assertEqual(record['status'], 'success')
        self.assertFalse((directory / '_work').exists())

    def test_fixed_onnx_workflow_repeats_without_removing_prepared_inputs(self):
        from aimet_onnx.rx_ptq.pipeline import run_onnx_ptq, write_onnx_ptq_metadata
        import onnx
        runtime.configure_output(timestamp_output=False)
        root = self.path / 'fixed-workflow'
        source = root / 'models/source.onnx'
        source.parent.mkdir(parents=True)
        onnx.save(fixtures.make_graph(), source)
        calibration = root / 'datasets/calib'
        calibration.mkdir(parents=True)
        sample = calibration / 'sample.npy'
        np.save(sample, self.fixture.feed['x'])
        source_hash, sample_hash = runtime._sha256(source), runtime._sha256(sample)
        ids = []
        for _ in range(2):
            result = run_onnx_ptq(model_path=source, calibration_dir=calibration,
                                  output_dir=root, filename_prefix='model',
                                  config_file=str(self.fixture.config), bitwidth_config_file='', quant_scheme='tf')
            self.assertTrue(all(path.parent == root / 'exports' for path in result.values()))
            record = json.loads((root / 'run.json').read_text())
            self.assertFalse(record['output_config']['timestamp_output'])
            self.assertEqual(record['status'], 'success')
            ids.append(record['execution_id'])
            self.assertFalse((root / record['execution_id']).exists())
            self.assertFalse((root / '_work').exists())
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(runtime._sha256(source), source_hash)
        self.assertEqual(runtime._sha256(sample), sample_hash)
        self.assertIn('Po2 method=', (root / 'run.log').read_text())
        metadata = write_onnx_ptq_metadata(root, 'model', {'after_export': True})
        self.assertEqual(metadata, root / 'exports/model.meta.json')
        record = json.loads((root / 'run.json').read_text())
        self.assertEqual(record['outputs']['exports/model.meta.json']['sha256'], runtime._sha256(metadata))

    def test_fixed_torch_export_uses_the_same_root_for_logs_and_models(self):
        from aimet_torch.rx_export import export_onnx_json
        import onnx
        runtime.configure_output(timestamp_output=False)
        root = self.path / 'fixed-torch'
        sim = self.torch_sim()
        stage = runtime.get_run(sim).root
        onnx_path, encodings_path = export_onnx_json(
            sim, str(root), 'model', dummy_input_shape=(1, 4))
        self.assertEqual(Path(onnx_path), root / 'exports/model.onnx')
        self.assertEqual(Path(encodings_path), root / 'exports/model.encodings')
        onnx.checker.check_model(onnx.load(onnx_path))
        record = json.loads((root / 'run.json').read_text())
        self.assertEqual(record['status'], 'success')
        self.assertFalse(record['output_config']['timestamp_output'])
        self.assertIn('model.encodings', (root / 'run.log').read_text())
        self.assertFalse(stage.exists())

    def test_standalone_gru_still_loads_without_aimet_logging(self):
        original = builtins.__import__

        def without_aimet(name, *args, **kwargs):
            if name.startswith('aimet_common'):
                raise ModuleNotFoundError(name='aimet_common')
            return original(name, *args, **kwargs)

        root = Path(__file__).resolve().parents[1]
        with patch('builtins.__import__', side_effect=without_aimet):
            module = runpy.run_path(str(root / 'operators/quant-gru/pytorch/quant_gru.py'))
        operator = module['QuantGRU'](4, 2, batch_first=True)
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            operator.load_bitwidth_config(str(root / 'examples/config/quick_start_full_quant.json'), verbose=False)
        self.assertEqual(console.getvalue(), '')
        self.assertTrue(operator.use_quantization)

    def test_standalone_lstm_integration_does_not_require_aimet(self):
        original = builtins.__import__

        def without_aimet(name, *args, **kwargs):
            if name.startswith('aimet_common'):
                raise ModuleNotFoundError(name='aimet_common')
            return original(name, *args, **kwargs)

        root = Path(__file__).resolve().parents[1]
        with patch('builtins.__import__', side_effect=without_aimet):
            module = runpy.run_path(str(root / 'operators/quant-lstm/pytorch/lstm_aimet.py'))

        class Operator(module['LSTMAimetIntegration']):
            use_quantization = False

        operator = Operator()
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            operator.load_bitwidth_config({'LSTM_config': {'use_quantization': True}}, verbose=False)
        self.assertEqual(console.getvalue(), '')
        self.assertTrue(operator.use_quantization)


if __name__ == '__main__':
    unittest.main()
