"""Po2 contracts and real recurrent workflows, on CPU and available CUDA devices."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx
import torch
from torch import nn
import gru_interface_binding

from aimet_common import run_context as runtime
from aimet_common.compiler_encodings import finalize_compiler_encodings
from aimet_torch._base.nn.modules.custom import Add
from aimet_torch.compare_before_after_po2 import collect_quantizer_info, compare_quantizers
from aimet_torch.model_preparer import prepare_model
from aimet_torch.native_recurrent import named_native_recurrent
from aimet_torch.power_of_2_quantization import named_generic_quantizers, verify_model_power_of_2
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth, apply_power_of_2_workflow
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.quantization.base import QuantizerBase
from aimet_torch.v2.quantsim import QuantizationSimModel
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'
MIXED_CONFIG = ROOT / 'examples/config/quick_start_full_quant.json'


def quantizer_record(name):
    return {'name': name, 'qmin': -127, 'qmax': 127, 'scale': 0.125,
            'offset': 0, 'min': -15.875, 'max': 15.875, 'symmetric': True}


class CompareQuantizersContractTest(unittest.TestCase):
    def compare(self, before, after):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_quantizers(before, after)

    def test_two_empty_lists_have_no_differences(self):
        self.assertEqual(self.compare([], []), [])

    def test_missing_or_replaced_quantizers_are_rejected(self):
        a, b = quantizer_record('a'), quantizer_record('b')
        for before, after in (([], [a]), ([a], []), ([a], [a, b]), ([a], [b])):
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                self.compare(before, after)

    def test_reordered_quantizers_are_matched_by_name(self):
        a, b = quantizer_record('a'), quantizer_record('b')
        self.assertEqual(self.compare([a, b], [b, a]), [])

    def test_duplicate_names_are_rejected(self):
        a, b = quantizer_record('a'), quantizer_record('b')
        for before, after in (([a, a], [a, b]), ([a, b], [a, a]), ([a, a], [a, a])):
            with self.subTest(before=before, after=after), self.assertRaises(ValueError):
                self.compare(before, after)

    def test_signed_integer_domains_do_not_report_false_symmetry_issues(self):
        for symmetric in (True, False):
            with self.subTest(symmetric=symmetric):
                record = quantizer_record('signed')
                record.update(qmin=-128, symmetric=symmetric, min=-16.0)
                self.assertEqual(self.compare([record], [copy.deepcopy(record)]), [])


class RecurrentModel(nn.Module):
    def __init__(self, kind, *, ordinary=False, bidirectional=False):
        super().__init__()
        self.ordinary = ordinary
        if ordinary:
            self.proj = nn.Linear(4, 4)
        if kind in ('gru', 'pair'):
            self.gru = QuantGRU(4, 4, batch_first=True, bidirectional=bidirectional)
        if kind in ('lstm', 'pair'):
            self.lstm = QuantLSTM(4, 4, batch_first=True, bidirectional=bidirectional)
        self.kind = kind

    def forward(self, x):
        if self.ordinary:
            x = self.proj(x)
        if self.kind in ('gru', 'pair'):
            x, _ = self.gru(x)
        if self.kind in ('lstm', 'pair'):
            x, _ = self.lstm(x)
        return x


class PowerOfTwoWorkflowTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        runtime.configure_output(timestamp_output=False)
        self.addCleanup(runtime.configure_output)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(runtime.execution(self.directory))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(torch.no_grad())
        if gru_interface_binding.cpu_only:
            self.devices = ('cpu',)
        elif torch.cuda.is_available():
            self.devices = ('cuda',)
        else:
            self.skipTest('The CUDA operator wheel requires a CUDA device')

    def sim(self, kind, device, *, ordinary=False, calibrate=True,
            mixed_config=MIXED_CONFIG, bidirectional=False):
        torch.manual_seed(37)
        x = torch.linspace(-1.3, 1.7, 12, device=device).reshape(1, 3, 4)
        model = RecurrentModel(kind, ordinary=ordinary, bidirectional=bidirectional)
        model = prepare_model(model.to(device).eval())
        sim = QuantizationSimModel(model, x, quant_scheme='tf', config_file=str(BASE_CONFIG))
        apply_mixed_precision_bitwidth(sim.model, str(mixed_config), verbose=False)
        if calibrate:
            with compute_encodings(sim.model):
                sim.model(x)
                sim.model(x * 0.6)
        return sim, x

    def assert_report(self, report, *, generic, native):
        self.assertTrue(report['all_power_of_2'])
        self.assertEqual(report['before_count'], report['after_count'])
        self.assertEqual(bool(report['before_count']), generic)
        verification = report['verification']
        expected = 'verified' if generic or native else 'not_applicable'
        self.assertEqual(verification['status'], expected)
        self.assertEqual(verification['generic']['status'], 'verified' if generic else 'not_applicable')
        self.assertEqual(verification['generic']['quantizer_count'], report['after_count'])
        self.assertEqual(verification['native']['status'], 'verified' if native else 'not_applicable')
        self.assertEqual(verification['native']['module_count'], native)
        self.assertEqual(len(verification['native']['modules']), native)

    def native_encodings(self, model):
        documents = {}
        for name, module, _ in named_native_recurrent(model):
            if module.use_quantization:
                documents[name] = module.export_quant_params_to_aimet_format(
                    {}, module_name=name, for_onnx=False)
        return documents

    def quantizer_state_keys(self, model):
        return {f'{name}.{key}' for name, module in model.named_modules()
                if isinstance(module, QuantizerBase) for key in module.state_dict()}

    def assert_native_po2(self, model):
        for name, document in self.native_encodings(model).items():
            with self.subTest(module=name):
                finalize_compiler_encodings(document, require_po2=True)
                self.assertTrue(document['activation_encodings'][name])

    def test_native_only_and_mixed_models_match_in_both_verbosity_modes(self):
        for device in self.devices:
            for kind, ordinary in (('gru', False), ('lstm', False), ('pair', False),
                                   ('gru', True), ('lstm', True), ('pair', True)):
                observed = []
                for verbose in (False, True):
                    with self.subTest(device=device, kind=kind, ordinary=ordinary, verbose=verbose):
                        sim, x = self.sim(kind, device, ordinary=ordinary)
                        report = apply_power_of_2_workflow(sim.model, verbose=verbose)
                        self.assert_report(report, generic=ordinary, native=2 if kind == 'pair' else 1)
                        self.assert_native_po2(sim.model)
                        output = sim.model(x).detach().clone()
                        self.assertTrue(torch.isfinite(output).all())
                        if not verbose:
                            self.assertEqual(report['issues'], [])
                        observed.append((output, collect_quantizer_info(sim.model),
                                         self.native_encodings(sim.model), report['verification']))
                torch.testing.assert_close(observed[0][0], observed[1][0], atol=0, rtol=0)
                self.assertEqual(observed[0][1:], observed[1][1:])

    def test_bidirectional_native_operators_verify_both_directions(self):
        for device in self.devices:
            for kind in ('gru', 'lstm'):
                with self.subTest(device=device, kind=kind):
                    sim, x = self.sim(kind, device, bidirectional=True)
                    report = apply_power_of_2_workflow(sim.model, verbose=False)
                    self.assert_report(report, generic=False, native=1)
                    self.assert_native_po2(sim.model)
                    document = self.native_encodings(sim.model)[kind]
                    entry = document['activation_encodings'][kind]
                    self.assertEqual(set(entry['internal_ops']), set(entry['internal_ops_reverse']))
                    output = sim.model(x)
                    self.assertEqual(tuple(output.shape), (1, 3, 8))
                    self.assertTrue(torch.isfinite(output).all())

    def test_native_verification_rejects_incomplete_or_non_po2_public_encodings(self):
        for device in self.devices:
            for kind in ('gru', 'lstm'):
                sim, _ = self.sim(kind, device, bidirectional=True)
                apply_power_of_2_workflow(sim.model, verbose=False)
                module = sim.model.get_submodule(kind)
                original = module.export_quant_params_to_aimet_format(
                    {}, module_name=kind, for_onnx=False)
                for defect in ('missing_internal_both_directions', 'missing_parameter',
                               'missing_reverse', 'non_po2_scale'):
                    with self.subTest(device=device, kind=kind, defect=defect):
                        document = copy.deepcopy(original)
                        entry = document['activation_encodings'][kind]
                        if defect == 'missing_internal_both_directions':
                            operation = next(name for name in entry['internal_ops']
                                             if name not in ('input', 'output'))
                            for field in ('internal_ops', 'internal_ops_reverse'):
                                del entry[field][operation]
                        elif defect == 'missing_parameter':
                            del document['param_encodings'][f'{kind}.weight_ih.weight']
                        elif defect == 'missing_reverse':
                            del entry['internal_ops_reverse']
                        else:
                            operation = next(iter(entry['internal_ops'].values()))
                            operation['output'][0]['scale'] = 0.3
                        with patch.object(module, 'export_quant_params_to_aimet_format',
                                          return_value=document):
                            with self.assertRaises(ValueError):
                                apply_power_of_2_workflow(sim.model, verbose=False)

    def test_ordinary_only_and_no_targets_report_the_verified_scope(self):
        for device in self.devices:
            for kind in ('ordinary', 'empty'):
                for verbose in (False, True):
                    with self.subTest(device=device, kind=kind, verbose=verbose):
                        if kind == 'ordinary':
                            sim, x = self.sim(kind, device, ordinary=True)
                            model = sim.model
                        else:
                            x = torch.ones(1, 3, 4, device=device)
                            model = nn.Identity().to(device).eval()
                        report = apply_power_of_2_workflow(model, verbose=verbose)
                        self.assert_report(report, generic=kind == 'ordinary', native=0)
                        self.assertTrue(torch.isfinite(model(x)).all())

    def test_literal_scalar_slots_do_not_hide_uncalibrated_tensor_inputs(self):
        class ScalarFirst(nn.Module):
            def forward(self, x):
                return torch.add(2., x)

        class KeywordScalarFirst(nn.Module):
            def __init__(self):
                super().__init__()
                self.add = Add()

            def forward(self, x):
                return self.add(x=2., y=x)

        for device in self.devices:
            for model_type in (ScalarFirst, KeywordScalarFirst):
                with self.subTest(device=device, model=model_type.__name__):
                    x = torch.linspace(-1.3, 1.7, 8, device=device).reshape(2, 4)
                    prepared = prepare_model(model_type().to(device).eval())
                    sim = QuantizationSimModel(prepared, x, quant_scheme='tf',
                                               config_file=str(BASE_CONFIG))
                    with self.assertRaisesRegex(RuntimeError, 'calibrat|initializ'):
                        apply_power_of_2_workflow(sim.model, verbose=False)
                    with compute_encodings(sim.model):
                        sim.model(x)
                        sim.model(x * 0.6)
                    name, module = next((name, module) for name, module in sim.model.named_modules()
                                        if isinstance(module, Add))
                    self.assertIsNotNone(module.input_quantizers[0])
                    self.assertFalse(module.input_quantizers[0].is_initialized())
                    self.assertIsNotNone(module.input_quantizers[1])
                    self.assertTrue(module.input_quantizers[1].is_initialized())
                    targets = dict(named_generic_quantizers(sim.model))
                    self.assertNotIn(f'{name}.input_quantizer[0]', targets)
                    self.assertIn(f'{name}.input_quantizer[1]', targets)
                    report = apply_power_of_2_workflow(sim.model, verbose=False)
                    self.assert_report(report, generic=True, native=0)
                    self.assertTrue(verify_model_power_of_2(sim.model, verbose=False))
                    for _, quantizer in named_generic_quantizers(sim.model):
                        scale = quantizer.get_scale()
                        torch.testing.assert_close(scale.log2(), scale.log2().round(), atol=1e-6, rtol=0)
                    self.assertTrue(torch.isfinite(sim.model(x)).all())

    def test_bare_native_roots_use_their_effective_encodings(self):
        for device in self.devices:
            for kind in (QuantGRU, QuantLSTM):
                with self.subTest(device=device, operator=kind.__name__):
                    module = kind(4, 4, batch_first=True).to(device).eval()
                    module.load_bitwidth_config(str(MIXED_CONFIG), verbose=False)
                    x = torch.linspace(-1.3, 1.7, 12, device=device).reshape(1, 3, 4)
                    with module.calibration_context():
                        module(x)
                        module(x * 0.6)
                    self.assertTrue(module.is_calibrated())
                    report = apply_power_of_2_workflow(module, verbose=False)
                    self.assert_report(report, generic=False, native=1)
                    document = module.export_quant_params_to_aimet_format(
                        {}, module_name='root', for_onnx=False)
                    finalize_compiler_encodings(document, require_po2=True)
                    output, _ = module(x)
                    self.assertTrue(torch.isfinite(output).all())

    def test_invalid_generic_scales_fail_before_native_or_generic_conversion(self):
        invalid_scales = ([], [float('nan')], [float('inf')], [float('-inf')],
                          [0.0], [-0.125], [0.125, float('nan')])
        for device in self.devices:
            sim, _ = self.sim('pair', device, ordinary=True)
            quantizer = sim.model.proj.param_quantizers['weight']
            native_before = self.native_encodings(sim.model)
            generic_before = collect_quantizer_info(sim.model)
            for values in invalid_scales:
                with self.subTest(device=device, scale=values):
                    scale = torch.tensor(values, device=device)
                    with patch.object(quantizer, 'get_scale', return_value=scale):
                        with self.assertRaisesRegex(ValueError, 'finite and positive'):
                            apply_power_of_2_workflow(sim.model, verbose=False)
                    self.assertEqual(native_before, self.native_encodings(sim.model))
                    self.assertEqual(generic_before, collect_quantizer_info(sim.model))

    def test_uncalibrated_targets_fail_before_any_conversion(self):
        for device in self.devices:
            for kind, ordinary in (('gru', False), ('lstm', False), ('pair', True),
                                   ('ordinary', True)):
                with self.subTest(device=device, kind=kind):
                    sim, _ = self.sim(kind, device, ordinary=ordinary, calibrate=False)
                    before = copy.deepcopy(sim.model.state_dict())
                    with self.assertRaisesRegex(RuntimeError, 'calibrat|initializ'):
                        apply_power_of_2_workflow(sim.model, verbose=False)
                    self.assertEqual(before.keys(), sim.model.state_dict().keys())
                    for name, value in before.items():
                        torch.testing.assert_close(value, sim.model.state_dict()[name], atol=0, rtol=0)

    def test_disabled_native_targets_need_no_calibration(self):
        for device in self.devices:
            with self.subTest(device=device):
                sim, x = self.sim('pair', device, calibrate=False)
                for _, module, _ in named_native_recurrent(sim.model):
                    module.use_quantization = False
                report = apply_power_of_2_workflow(sim.model, verbose=False)
                self.assert_report(report, generic=False, native=0)
                self.assertTrue(torch.isfinite(sim.model(x)).all())

    def affine_config(self):
        config = json.loads(MIXED_CONFIG.read_text())
        config['GRU_config']['default_config']['use_pot2_scale'] = False
        path = self.directory / 'affine.json'
        path.write_text(json.dumps(config))
        return path

    def assert_native_non_po2(self, model):
        def scales(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == 'scale':
                        yield torch.as_tensor(item)
                    else:
                        yield from scales(item)
            elif isinstance(value, list):
                for item in value:
                    yield from scales(item)

        for name, document in self.native_encodings(model).items():
            with self.subTest(module=name):
                self.assertTrue(any(not torch.allclose(scale.log2(), scale.log2().round(),
                                                       atol=1e-6, rtol=0)
                                    for scale in scales(document)),
                                'Affine calibration must exercise an actual Po2 conversion')

    def test_affine_native_grids_are_actually_converted_with_and_without_generic_layers(self):
        config = self.affine_config()
        for device in self.devices:
            for kind in ('gru', 'lstm'):
                for ordinary in (False, True):
                    with self.subTest(device=device, kind=kind, ordinary=ordinary):
                        sim, x = self.sim(kind, device, ordinary=ordinary, mixed_config=config)
                        self.assert_native_non_po2(sim.model)
                        report = apply_power_of_2_workflow(sim.model, verbose=False)
                        self.assert_report(report, generic=ordinary, native=1)
                        self.assert_native_po2(sim.model)
                        self.assertTrue(torch.isfinite(sim.model(x)).all())

    def test_generic_verification_requires_nonempty_finite_power_of_two_scales(self):
        cases = (([], False), ([float('nan')], False), ([float('inf')], False),
                 ([float('-inf')], False), ([0.0], False), ([-0.125], False),
                 ([0.125, 0.3], False), ([0.125, 0.25], True))
        for device in self.devices:
            sim, _ = self.sim('ordinary', device, ordinary=True)
            apply_power_of_2_workflow(sim.model, verbose=False)
            quantizer = sim.model.proj.param_quantizers['weight']
            for values, expected in cases:
                with self.subTest(device=device, scale=values):
                    scale = torch.tensor(values, device=device)
                    with patch.object(quantizer, 'get_scale', return_value=scale):
                        self.assertEqual(verify_model_power_of_2(sim.model, verbose=False), expected)

    def test_locked_native_conversion_fails_without_changing_generic_encodings(self):
        config = self.affine_config()
        for device in self.devices:
            for kind in ('gru', 'lstm'):
                with self.subTest(device=device, kind=kind):
                    sim, _ = self.sim(kind, device, ordinary=True, mixed_config=config)
                    before = collect_quantizer_info(sim.model)
                    for _, module, _ in named_native_recurrent(sim.model):
                        module.set_quant_params_locked(True)
                    with self.assertRaisesRegex(RuntimeError, 'lock'):
                        apply_power_of_2_workflow(sim.model, verbose=False)
                    self.assertEqual(before, collect_quantizer_info(sim.model))

    def test_repeated_and_locked_po2_workflows_are_idempotent(self):
        for device in self.devices:
            with self.subTest(device=device):
                sim, x = self.sim('pair', device, ordinary=True)
                first = apply_power_of_2_workflow(sim.model, verbose=False)
                before = self.native_encodings(sim.model)
                generic_before = collect_quantizer_info(sim.model)
                output = sim.model(x).detach().clone()
                for _, module, _ in named_native_recurrent(sim.model):
                    module.set_quant_params_locked(True)
                second = apply_power_of_2_workflow(sim.model, verbose=True)
                self.assertEqual(first['verification'], second['verification'])
                self.assertEqual(before, self.native_encodings(sim.model))
                self.assertEqual(generic_before, collect_quantizer_info(sim.model))
                torch.testing.assert_close(output, sim.model(x), atol=0, rtol=0)

    def test_conversion_and_native_verification_precede_comparison(self):
        for device in self.devices:
            with self.subTest(device=device):
                sim, _ = self.sim('pair', device, ordinary=True)
                with patch('aimet_torch.compare_before_after_po2.compare_quantizers',
                           side_effect=RuntimeError('report failure')):
                    with self.assertRaisesRegex(RuntimeError, 'report failure'):
                        apply_power_of_2_workflow(sim.model, verbose=False)
                self.assert_native_po2(sim.model)
                for record in collect_quantizer_info(sim.model):
                    scale = torch.as_tensor(record['scale'])
                    torch.testing.assert_close(scale.log2(), scale.log2().round(), atol=1e-6, rtol=0)

    def test_example_configuration_exports_and_restores_native_only_and_mixed_models(self):
        cases = [(kind, ordinary) for kind in ('gru', 'lstm', 'pair') for ordinary in (False, True)]
        for device in self.devices:
            for kind, ordinary in cases:
                with self.subTest(device=device, kind=kind, ordinary=ordinary):
                    sim, x = self.sim(kind, device, ordinary=ordinary)
                    report = apply_power_of_2_workflow(sim.model, verbose=False)
                    native_count = 2 if kind == 'pair' else 1
                    self.assert_report(report, generic=ordinary, native=native_count)
                    expected = sim.model(x).detach().clone()
                    prefix = f'{device}-{kind}-{"mixed" if ordinary else "native"}'
                    onnx_path, json_path = export_onnx_json(
                        sim, str(self.directory), prefix, tuple(x.shape))
                    graph = onnx.load(onnx_path)
                    onnx.checker.check_model(graph)
                    types = [node.op_type for node in graph.graph.node
                             if node.op_type in ('GRU', 'LSTM')]
                    self.assertEqual(types, ['GRU', 'LSTM'] if kind == 'pair' else [kind.upper()])
                    document = json.loads(Path(json_path).read_text())
                    finalize_compiler_encodings(document, require_po2=True)
                    for name, _, op in named_native_recurrent(sim.model):
                        entry = document['activation_encodings'][name]
                        self.assertTrue(entry[f'is_{op.onnx_type}'])
                        self.assertEqual(entry['input']['0']['bitwidth'], 8)
                        if op.onnx_type == 'LSTM':
                            self.assertEqual(entry['internal_ops']['cell_state']['output'][0]['bitwidth'], 16)
                    fresh, _ = self.sim(kind, device, ordinary=ordinary, calibrate=False)
                    quantizer_keys = self.quantizer_state_keys(sim.model)
                    weights = {name: value for name, value in sim.model.state_dict().items()
                               if name not in quantizer_keys}
                    incompatible = fresh.model.load_state_dict(weights, strict=False)
                    self.assertEqual(set(incompatible.missing_keys), self.quantizer_state_keys(fresh.model))
                    self.assertEqual(incompatible.unexpected_keys, [])
                    restored = load_quantizer_encodings(fresh.model, json_path,
                                                       allow_overwrite=False, verbose=False)
                    self.assertEqual(restored['loaded'], report['before_count'] + native_count)
                    torch.testing.assert_close(fresh.model(x), expected, atol=0, rtol=0)
                    rechecked = apply_power_of_2_workflow(fresh.model, verbose=False)
                    self.assert_report(rechecked, generic=ordinary, native=native_count)


if __name__ == '__main__':
    unittest.main()
