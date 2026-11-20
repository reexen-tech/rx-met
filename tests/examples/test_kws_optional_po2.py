"""Run the KWS entry point after deleting its optional Po2 stage from the AST."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import gru_interface_binding
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aimet_common import run_context as runtime
from aimet_torch.native_recurrent import named_native_recurrent
from aimet_torch.v2.nn import QuantizationMixin
from examples import quick_start_kws as kws


def load_example(test, function_name, *, skip_po2):
    """Compile the user's edit: remove the complete optional with-stage block."""
    tree = ast.parse(Path(kws.__file__).read_text())
    removed = []

    class RemoveStage(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name != function_name:
                return node
            return self.generic_visit(node)

        def visit_With(self, node):
            for item in node.items:
                call = item.context_expr
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == 'stage' and call.args
                        and isinstance(call.args[0], ast.Constant)
                        and 'Power-of-2' in str(call.args[0].value)):
                    removed.append(node)
                    return None
            return self.generic_visit(node)

    if skip_po2:
        tree = RemoveStage().visit(tree)
        test.assertEqual(len(removed), 1, f'{function_name} must have one removable Po2 stage')
    name = 'examples._kws_without_po2'
    module = types.ModuleType(name)
    module.__file__ = kws.__file__
    module.__package__ = 'examples'
    sys.modules[name] = module
    test.addCleanup(sys.modules.pop, name, None)
    exec(compile(ast.fix_missing_locations(tree), kws.__file__, 'exec'), module.__dict__)
    return module


class KwsOptionalPo2Test(unittest.TestCase):
    def setUp(self):
        if gru_interface_binding.cpu_only:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.skipTest('The CUDA recurrent wheel requires a CUDA device')
        runtime.configure_output(timestamp_output=False)
        self.addCleanup(runtime.configure_output)
        torch.manual_seed(67)

    def assert_bias_grid(self, model, *, rtol=1e-6):
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)) or not isinstance(module, QuantizationMixin):
                continue
            if module.bias is None:
                continue
            count += 1
            bias = module.param_quantizers['bias']
            self.assertEqual(bias.bitwidth, 32, name)
            self.assertTrue(bias.signed, name)
            self.assertTrue(bias.symmetric, name)
            expected = (module.input_quantizers[0].get_scale().reshape(())
                        * module.param_quantizers['weight'].get_scale().reshape(-1))
            # CPU transfer and JSON restore can change range-derived scales by FP32 ULPs.
            torch.testing.assert_close(bias.get_scale().reshape(-1), expected, rtol=rtol, atol=0,
                                       msg=lambda message: f'{name}: {message}')
        self.assertGreater(count, 0)

    def check_workflow(self, *, ptq_only, skip_po2=True):
        if not ptq_only and self.device.type != 'cuda':
            self.skipTest('Floating-point training and QAT require the CUDA recurrent wheel')
        example = load_example(self, '_run_kws_ptq' if ptq_only else '_run_kws', skip_po2=skip_po2)
        original_export = example.export_onnx_json
        original_evaluate = example.evaluate
        original_load = example.load_quantizer_encodings
        restored_models = set()
        waveforms = torch.randn(2, example.TARGET_LEN) * .02
        data = TensorDataset(waveforms, torch.tensor([2, 3]))
        loader = DataLoader(data, batch_size=2, shuffle=False, num_workers=0)
        loaders = {name: loader for name in ('train', 'val', 'test', 'calib')}

        def new_model(rnn_type):
            return example.AttMHRNN(rnn_type=rnn_type, cnn_filters=(2, 1), rnn_layers=1,
                                     rnn_units=4, heads=1, dense_units=(4,), dropout=0.)\
                .to(self.device)

        def evaluate(model, *args, **kwargs):
            if any(isinstance(module, QuantizationMixin) for module in model.modules()):
                self.assert_bias_grid(model, rtol=1e-6 if id(model) in restored_models else 0)
            return original_evaluate(model, *args, **kwargs)

        def load(model, *args, **kwargs):
            report = original_load(model, *args, **kwargs)
            restored_models.add(id(model))
            self.assert_bias_grid(model)
            return report

        exported = []

        def export(sim, *args, **kwargs):
            self.assert_bias_grid(sim.model)
            for name, module, operator in named_native_recurrent(sim.model):
                if operator.onnx_type == 'GRU':
                    self.assertEqual(module.use_pot2_scale, not skip_po2)
                document = module.export_quant_params_to_aimet_format(
                    {}, module_name=name, for_onnx=False)
                scales = [np.asarray(record['scale']).reshape(-1)
                          for record in document['param_encodings'].values()
                          if not record['dtype'].startswith('FLOAT')]
                logarithms = np.log2(np.concatenate(scales))
                self.assertEqual(bool(np.any(np.abs(logarithms - np.rint(logarithms)) > 1e-4)),
                                 skip_po2, name)
            result = original_export(sim, *args, **kwargs)
            exported.append(result)
            return result

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            calib, validation = directory / 'feeds/calib', directory / 'feeds/val'
            for folder in (calib, validation):
                folder.mkdir(parents=True)
                for index, waveform in enumerate(waveforms):
                    np.savez(folder / f'{index:05d}.npz', input=waveform.unsqueeze(0).numpy())
            (directory / 'feeds/manifest.json').write_text(json.dumps({
                f'{split}_{field}': value
                for split in ('calib', 'val')
                for field, value in (('files', ['00000.npz', '00001.npz']), ('labels', [2, 3]))
            }))
            for rnn_type in ('gru', 'lstm'):
                with self.subTest(rnn_type=rnn_type), contextlib.redirect_stdout(io.StringIO()) as output, \
                     patch.object(example, 'build_dataloaders', return_value=loaders), \
                     patch.object(example, '_new_model', side_effect=new_model), \
                     patch.object(example, 'evaluate', side_effect=evaluate), \
                     patch.object(example, 'load_quantizer_encodings', side_effect=load), \
                     patch.object(example, 'export_onnx_json', side_effect=export):
                    arguments = ['--rnn_type', rnn_type, '--device', str(self.device),
                                 '--output-dir', str(directory / 'runs')]
                    if ptq_only:
                        checkpoint = directory / f'{rnn_type}.pth'
                        example.save_checkpoint(new_model(rnn_type), checkpoint, rnn_type)
                        arguments += ['--ptq-only', '--fp-checkpoint', str(checkpoint),
                                      '--calib-dir', str(calib), '--validation-dir', str(validation)]
                    example.main(arguments)
                    root = directory / 'runs' / rnn_type
                    self.assertEqual(json.loads((root / 'run.json').read_text())['status'], 'success')
                    self.assertEqual('Power-of-2' in output.getvalue(), not skip_po2)
                    self.assertTrue(Path(exported[-1][0]).is_file())
                    self.assertTrue(Path(exported[-1][1]).is_file())
                    if ptq_only:
                        metadata = json.loads((root / 'exports/att_mh_rnn_kws_ptq.metadata.json').read_text())
                        if skip_po2:
                            self.assertNotIn('all_power_of_2', metadata)
                        else:
                            self.assertTrue(metadata['all_power_of_2'])
                        expected_stages = {'fp32', 'ptq', 'restored'}
                        if not skip_po2:
                            expected_stages.add('po2')
                        self.assertEqual(set(metadata['accuracy']), expected_stages)
                        self.assertLessEqual(metadata['restore_max_abs'], 1e-5)
                        with np.load(root / 'exports/outputs.npz') as values:
                            self.assertEqual(set(values.files), {
                                f'{name}_output0' for name in expected_stages if name != 'restored'})
                    else:
                        self.assertTrue((root / 'checkpoints/att_mh_rnn_kws_qat.pth').is_file())

    def test_ptq_only_without_optional_po2_exports_and_restores(self):
        self.check_workflow(ptq_only=True)

    def test_qat_without_optional_po2_exports_and_restores(self):
        self.check_workflow(ptq_only=False)

    def test_ptq_only_with_optional_po2_exports_and_restores(self):
        self.check_workflow(ptq_only=True, skip_po2=False)


if __name__ == '__main__':
    unittest.main()
