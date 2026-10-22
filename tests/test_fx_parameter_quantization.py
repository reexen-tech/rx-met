"""Exercise parameter QDQ through FX get_attr, as used by decomposed BN."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import onnx
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings, compute_param_encodings
from aimet_torch.staged_quantization_utils import save_quantizer_encodings, load_quantizer_encodings
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_torch.v2.quantization.encoding_analyzer import (
    MinMaxEncodingAnalyzer, PercentileEncodingAnalyzer, TfEnhancedEncodingAnalyzer,
)

CONFIG = Path('/repo/examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json')
if not CONFIG.exists():
    CONFIG = Path(__file__).resolve().parents[1] / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = QuantizableBatchNorm2d(3)
        with torch.no_grad():
            self.bn.weight.copy_(torch.tensor([.31, .74, .99]))
            self.bn.bias.copy_(torch.tensor([.13, -.21, .07]))
            self.bn.running_var.copy_(torch.tensor([.7, 1.2, 1.7]))
    def forward(self, x):
        return self.bn(x)


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.31, .74, .99]))
        self.bias = nn.Parameter(torch.tensor([.13, -.21, .07]))

    def forward(self, x):
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SchemeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = QuantizableBatchNorm2d(3)
        self.affine = Affine()

    def forward(self, x):
        return self.affine(self.bn(self.conv(x)))


class FxParameterQuantizationTest(unittest.TestCase):
    def test_auto_created_parameters_follow_selected_scheme(self):
        cases = (
            ('tf', MinMaxEncodingAnalyzer),
            (QuantScheme.min_max, MinMaxEncodingAnalyzer),
            ('percentile', PercentileEncodingAnalyzer),
            (QuantScheme.post_training_percentile, PercentileEncodingAnalyzer),
            ('tf_enhanced', TfEnhancedEncodingAnalyzer),
            (QuantScheme.training_range_learning_with_tf_init, MinMaxEncodingAnalyzer),
            (QuantScheme.training_range_learning_with_tf_enhanced_init, TfEnhancedEncodingAnalyzer),
        )
        devices = ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',)
        for device in devices:
            for scheme, analyzer in cases:
                with self.subTest(device=device, scheme=scheme):
                    torch.manual_seed(42)
                    x = torch.randn(1, 3, 4, 4, device=device)
                    sim = QuantizationSimModel(prepare_model(SchemeModel().to(device).eval()), x,
                        quant_scheme=scheme, config_file=str(CONFIG))
                    sim.set_percentile_value(95.)
                    for name in ('conv', 'bn', 'affine'):
                        module = sim.model.get_submodule(name)
                        for key in ('weight', 'bias'):
                            q = module.param_quantizers[key]
                            self.assertIsInstance(q.encoding_analyzer, analyzer, f'{name}.{key}')
                            if analyzer is PercentileEncodingAnalyzer:
                                self.assertEqual(q.encoding_analyzer.percentile, 95.)

    def test_percentile_setting_changes_expanded_parameter_calibration(self):
        sim = self.make_sim(quant_scheme='percentile')
        q = sim.model.bn.param_quantizers['weight']
        self.assertIsInstance(q.encoding_analyzer, PercentileEncodingAnalyzer)
        with torch.no_grad():
            sim.model.bn.weight.copy_(torch.tensor([1., 1., 100.]))
        sim.set_percentile_value(100.)
        compute_param_encodings(sim.model)
        untrimmed = q.get_scale().item()
        sim.set_percentile_value(50.)
        compute_param_encodings(sim.model)
        self.assertLess(q.get_scale().item(), untrimmed / 10)

    def test_auto_created_parameters_share_integer_and_float_creation_rules(self):
        for dtype, bits in ((QuantizationDataType.int, 16),
                            (QuantizationDataType.float, 8),
                            (QuantizationDataType.float, 16)):
            with self.subTest(dtype=dtype, bits=bits), patch('aimet_common.quantsim.ALLOW_EXPERIMENTAL', True):
                torch.manual_seed(42)
                x = torch.randn(1, 3, 4, 4)
                sim = QuantizationSimModel(prepare_model(SchemeModel().eval()), x,
                    quant_scheme='percentile', config_file=str(CONFIG),
                    default_param_bw=bits, default_output_bw=bits, default_data_type=dtype)
                normal = sim.model.conv.param_quantizers['weight']
                for name in ('bn', 'affine'):
                    for q in sim.model.get_submodule(name).param_quantizers.values():
                        self.assertIs(type(q), type(normal))
                        self.assertEqual(q.bitwidth, bits)
                        if dtype == QuantizationDataType.int or bits == 8:
                            self.assertIsInstance(q.encoding_analyzer, PercentileEncodingAnalyzer)
                        else:
                            self.assertIsNone(q.encoding_analyzer)
                        if dtype == QuantizationDataType.float:
                            self.assertTrue(all(not p.requires_grad for p in q.parameters()))

    def make_sim(self, quant_scheme='tf'):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 4, 4)
        self.fp = Model().eval()
        sim = QuantizationSimModel(prepare_model(copy.deepcopy(self.fp)), self.x,
                                  quant_scheme=quant_scheme, config_file=str(CONFIG))
        if quant_scheme == 'percentile':
            sim.set_percentile_value(99.99)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        return sim

    @staticmethod
    def disable_activation_quantizers(model):
        for m in model.modules():
            for attr in ['input_quantizers', 'output_quantizers']:
                qs = getattr(m, attr, [])
                for i in range(len(qs)):
                    qs[i] = None

    def test_parent_grid_executes_once_and_master_gradients_survive(self):
        sim = self.make_sim()
        self.disable_activation_quantizers(sim.model)
        b = sim.model.bn
        calls = {'weight': 0, 'bias': 0}
        handles = []
        for name, scale in [('weight', .25), ('bias', .125)]:
            q = b.param_quantizers[name]
            q.set_range(torch.tensor(q.qmin * scale), torch.tensor(q.qmax * scale))
            q.requires_grad_(False)
            def count(module, args, output, key=name):
                calls[key] += 1
            handles.append(q.register_forward_hook(count))
        x = self.x.clone().requires_grad_()
        w = b.weight.detach().clone().requires_grad_()
        bias = b.bias.detach().clone().requires_grad_()
        xr = self.x.clone().requires_grad_()
        def ste(v, scale):
            return v + (torch.round(v / scale).clamp(-128, 127) * scale - v).detach()
        normalized = self.fp.bn(xr)
        normalized = (normalized - self.fp.bn.bias.view(1, -1, 1, 1)) / self.fp.bn.weight.view(1, -1, 1, 1)
        expected = normalized * ste(w, .25).view(1, -1, 1, 1) + ste(bias, .125).view(1, -1, 1, 1)
        actual = sim.model(x)
        torch.testing.assert_close(actual, expected)
        upstream = torch.linspace(.1, 1.1, x.numel()).reshape_as(x)
        (actual * upstream).sum().backward()
        (expected * upstream).sum().backward()
        torch.testing.assert_close(b.weight.grad, w.grad)
        torch.testing.assert_close(b.bias.grad, bias.grad)
        torch.testing.assert_close(x.grad, xr.grad)
        self.assertEqual(calls, {'weight': 1, 'bias': 1})
        for h in handles:
            h.remove()
        b.param_quantizers['weight'] = None
        b.param_quantizers['bias'] = None
        torch.testing.assert_close(sim.model(self.x), self.fp(self.x))

    def test_parameter_edges_have_only_one_quantization_owner(self):
        sim = self.make_sim()
        for name, module in sim.model.bn.named_modules():
            if 'module_mul' in name or 'module_add' in name:
                if hasattr(module, 'input_quantizers'):
                    self.assertIsNone(module.input_quantizers[1], name)

    def test_recalibration_observes_updated_master_parameters(self):
        sim = self.make_sim()
        q = sim.model.bn.param_quantizers['weight']
        before = q.get_scale().clone()
        with torch.no_grad():
            sim.model.bn.weight.mul_(3)
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(self.x)
        self.assertGreater(q.get_scale().item(), before.item() * 2)

    def test_parameter_only_calibration_observes_master_and_respects_lock(self):
        sim = self.make_sim()
        q = sim.model.bn.param_quantizers['weight']
        before = q.get_scale().clone()
        with torch.no_grad():
            sim.model.bn.weight.mul_(3)
        compute_param_encodings(sim.model)
        self.assertGreater(q.get_scale().item(), before.item() * 2)
        locked_scale = q.get_scale().clone()
        q.allow_overwrite(False)
        with torch.no_grad():
            sim.model.bn.weight.mul_(3)
        compute_param_encodings(sim.model)
        torch.testing.assert_close(q.get_scale(), locked_scale, rtol=0, atol=0)

    def test_empty_layer_exclusion_preserves_parameter_qdq(self):
        sim = self.make_sim()
        self.disable_activation_quantizers(sim.model)
        q = sim.model.bn.param_quantizers['weight']
        q.set_range(torch.tensor(q.qmin * .25), torch.tensor(q.qmax * .25))
        expected = sim.model(self.x).detach()
        self.assertFalse(torch.equal(expected, self.fp(self.x)))
        sim.exclude_layers_from_quantization([])
        torch.testing.assert_close(sim.model(self.x), expected, rtol=0, atol=0)
        child = next(m for m in sim.model.bn.modules() if hasattr(m, 'input_quantizers'))
        sim.exclude_layers_from_quantization([child])
        torch.testing.assert_close(sim.model(self.x), expected, rtol=0, atol=0)
        sim.exclude_layers_from_quantization([sim.model.bn])
        torch.testing.assert_close(sim.model(self.x), self.fp(self.x))

    def test_qdq_weight_export_preserves_parameter_values_and_master(self):
        sim = self.make_sim()
        self.disable_activation_quantizers(sim.model)
        q = sim.model.bn.param_quantizers['weight']
        q.set_range(torch.tensor(q.qmin * .25), torch.tensor(q.qmax * .25))
        before = sim.model.bn.weight.detach().clone()
        expected = sim.model(self.x).detach()
        exported = sim.get_original_model(sim.model, qdq_weights=True)
        torch.testing.assert_close(exported(self.x), expected, rtol=0, atol=0)
        torch.testing.assert_close(sim.model.bn.weight, before, rtol=0, atol=0)

    def test_float_checkpoint_and_stage_reload_preserve_semantics(self):
        sim = self.make_sim()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'stage.json')
            save_quantizer_encodings(sim.model, path, verbose=False)
            expected = sim.model(self.x).detach()
            copied = copy.deepcopy(sim.model)
            torch.testing.assert_close(copied(self.x), expected, rtol=0, atol=0)
            original = sim.get_original_model(sim.model, qdq_weights=False)
            torch.testing.assert_close(original(self.x), self.fp(self.x))
            self.assertFalse(any('quantizer' in n for n, _ in original.named_modules()))
            fresh = QuantizationSimModel(original, self.x, quant_scheme='tf', config_file=str(CONFIG))
            load_quantizer_encodings(fresh.model, path, skip_if_not_found=False,
                                     allow_overwrite=False, verbose=False)
            torch.testing.assert_close(fresh.model(self.x), expected, rtol=0, atol=0)

    def test_export_preserves_parameter_encodings_and_reload(self):
        sim = self.make_sim()
        expected = sim.model(self.x).detach()
        with tempfile.TemporaryDirectory() as directory:
            graph_path, encoding_path = export_onnx_json(sim, directory, 'bn', tuple(self.x.shape))
            graph = onnx.load(graph_path)
            self.assertNotIn('rx_met.parameter_calibration', {entry.key for entry in graph.metadata_props})
            enc = json.loads(Path(encoding_path).read_text())
            onnx.checker.check_model(graph, full_check=True)
            self.assertIn('bn.weight', enc['param_encodings'])
            self.assertIn('bn.bias', enc['param_encodings'])
            for node in graph.graph.node:
                if node.op_type in ('Mul', 'Add') and len(node.input) > 1 and node.input[1] in ['bn.weight', 'bn.bias']:
                    self.assertNotIn('1', enc['activation_encodings'][node.name]['input'])
            fresh = self.make_sim()
            load_quantizer_encodings(fresh.model, encoding_path, skip_if_not_found=False,
                                     allow_overwrite=False, verbose=False)
            torch.testing.assert_close(fresh.model(self.x), expected, atol=1e-6, rtol=1e-5)

    def test_percentile_parameters_match_onnx_without_metadata_override(self):
        from aimet_onnx.rx_ptq import pipeline

        sim = self.make_sim(quant_scheme='percentile')
        with tempfile.TemporaryDirectory() as directory:
            onnx_path, _ = export_onnx_json(sim, directory, 'bn', tuple(self.x.shape))
            model = onnx.load(onnx_path)
            self.assertNotIn('rx_met.parameter_calibration', {entry.key for entry in model.metadata_props})
            feeds = {'input': self.x.numpy()}
            other = pipeline.create_onnx_ptq_sim(model, feeds, config_file=str(CONFIG), quant_scheme='percentile')
            other.set_percentile_value(99.99)
            other.compute_encodings([feeds])
            for key in ('weight', 'bias'):
                self.assertEqual(other.qc_quantize_op_dict[f'bn.{key}'].get_encodings()[0].delta,
                                 sim.model.bn.param_quantizers[key].get_scale().item())
            pipeline.apply_power_of_2_workflow(other, align_bias_scale=False, verbose=False)
            artifacts = pipeline.export_onnx_json(other, model, Path(directory) / 'second', 'bn')
            pipeline.verify_onnx_ptq_artifacts(artifacts, feeds, other.session.run(None, feeds),
                config_file=str(CONFIG), quant_scheme='percentile')
