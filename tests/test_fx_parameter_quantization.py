"""Exercise parameter QDQ through FX get_attr, as used by decomposed BN."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
import onnx
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantizable_batchnorm import QuantizableBatchNorm2d
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings, compute_param_encodings
from aimet_torch.staged_quantization_utils import save_quantizer_encodings, load_quantizer_encodings
from aimet_torch.rx_export.export_onnx_json import export_onnx_json

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


class FxParameterQuantizationTest(unittest.TestCase):
    def make_sim(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 4, 4)
        self.fp = Model().eval()
        sim = QuantizationSimModel(prepare_model(copy.deepcopy(self.fp)), self.x,
                                  quant_scheme='tf', config_file=str(CONFIG))
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
