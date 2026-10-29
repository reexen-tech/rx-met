"""Prepared GELU must calibrate the same elementary operators it exports."""
import json
from pathlib import Path
import tempfile
import unittest

import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from aimet_torch.model_preparer import prepare_model
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.v2.nn import compute_encodings
from aimet_torch.v2.quantsim import QuantizationSimModel
from tests.support.paths import REPO_ROOT

CONFIG = REPO_ROOT / 'examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json'


class GeluModel(nn.Module):
    def __init__(self, approximate, functional=False):
        super().__init__()
        self.approximate = approximate
        self.functional = functional
        if not functional:
            self.gelu = nn.GELU(approximate=approximate)

    def forward(self, x):
        if self.functional == 'positional':
            return F.gelu(x, approximate=self.approximate)
        if self.functional:
            return F.gelu(input=x, approximate=self.approximate)
        return self.gelu(x)


class QuantizableGeluTest(unittest.TestCase):
    def test_prepared_outputs_and_gradients_match_torch(self):
        for approximate in ('none', 'tanh'):
            for functional in (False, True, 'positional'):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(approximate=approximate, functional=functional, dtype=dtype):
                        model = GeluModel(approximate, functional).eval()
                        prepared = prepare_model(model)
                        x = torch.linspace(-8, 8, 120, dtype=dtype).reshape(2, 3, 20).requires_grad_()
                        y = x.detach().clone().requires_grad_()
                        expected, actual = model(x), prepared(y)
                        torch.testing.assert_close(actual, expected)
                        upstream = torch.linspace(-2, 3, x.numel(), dtype=dtype).reshape_as(x)
                        (expected * upstream).sum().backward()
                        (actual * upstream).sum().backward()
                        torch.testing.assert_close(y.grad, x.grad, atol=1e-6, rtol=1e-5)
                        self.assertFalse(any(isinstance(m, nn.GELU) for m in prepared.modules()))

    def test_qat_export_and_encoding_reload(self):
        for approximate in ('none', 'tanh'):
            for functional in (False, True, 'positional'):
                with self.subTest(approximate=approximate, functional=functional), tempfile.TemporaryDirectory() as directory:
                    torch.manual_seed(41)
                    model = nn.Sequential(nn.Linear(4, 4), GeluModel(approximate, functional)).eval()
                    x = torch.randn(2, 3, 4)
                    def new_sim():
                        return QuantizationSimModel(prepare_model(model), x, quant_scheme='tf',
                                                    config_file=str(CONFIG))
                    sim = new_sim()
                    with torch.no_grad(), compute_encodings(sim.model):
                        sim.model(x)
                    sim.model.train()
                    optimizer = torch.optim.SGD(sim.model.parameters(), lr=0.01)
                    before = sim.model.get_submodule('0').weight.detach().clone()
                    loss = sim.model(x).square().mean()
                    loss.backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in sim.model.parameters()))
                    optimizer.step()
                    self.assertFalse(torch.equal(before, sim.model.get_submodule('0').weight))
                    sim.model.eval()
                    expected_quantized = sim.model(x).detach()
                    float_model = sim.get_original_model(sim.model, qdq_weights=False).eval()
                    expected_float = float_model(x).detach()
                    graph_path, enc_path = export_onnx_json(sim, directory, 'gelu', tuple(x.shape))
                    graph = onnx.load(graph_path)
                    onnx.checker.check_model(graph, full_check=True)
                    enc = json.loads(Path(enc_path).read_text())
                    self.assertLessEqual(set(enc['activation_encodings']), {n.name for n in graph.graph.node})
                    self.assertLessEqual(set(enc['param_encodings']), {t.name for t in graph.graph.initializer})
                    activation_op = 'Erf' if approximate == 'none' else 'Tanh'
                    self.assertEqual(sum(n.op_type == activation_op for n in graph.graph.node), 1)
                    for node in graph.graph.node:
                        if node.op_type in ('Div', 'Erf', 'Add', 'Mul', 'Pow', 'Tanh', 'Gemm'):
                            entry = enc['activation_encodings'][node.name]
                            self.assertIn('0', entry['input'])
                            self.assertIn('0', entry['output'])
                    actual = ort.InferenceSession(graph_path, providers=['CPUExecutionProvider']).run(
                        None, {'input': x.numpy()})[0]
                    torch.testing.assert_close(torch.from_numpy(actual), expected_float)
                    fresh = new_sim()
                    fresh.model.load_state_dict(float_model.state_dict(), strict=False)
                    load_quantizer_encodings(fresh.model, enc_path, skip_if_not_found=False,
                                            allow_overwrite=False, verbose=False)
                    torch.testing.assert_close(fresh.model(x), expected_quantized, atol=1e-6, rtol=1e-5)

    def test_repeated_preparation_keeps_operator_names(self):
        from aimet_torch.quantizable_gelu import QuantizableGELU
        for approximate in ('none', 'tanh'):
            with self.subTest(approximate=approximate):
                model = GeluModel(approximate).eval()
                first = prepare_model(model)
                second = prepare_model(first)
                self.assertEqual(list(dict(first.named_modules())), list(dict(second.named_modules())))
                x = torch.randn(2, 4)
                torch.testing.assert_close(second(x), first(x), atol=0, rtol=0)
                # Direct use of the expanded module also remains preparable.
                root = prepare_model(QuantizableGELU(approximate).eval())
                sim = QuantizationSimModel(root, x, quant_scheme='tf', config_file=str(CONFIG))
                with torch.no_grad(), compute_encodings(sim.model):
                    sim.model(x)
                with tempfile.TemporaryDirectory() as directory:
                    export_onnx_json(sim, directory, 'root', tuple(x.shape))

    def test_reused_gelu_has_independent_calibration(self):
        class Reused(nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = nn.GELU()
            def forward(self, x):
                return self.gelu(x) + self.gelu(x * 10)
        x = torch.randn(2, 4)
        sim = QuantizationSimModel(prepare_model(Reused().eval()), x, quant_scheme='tf',
                                  config_file=str(CONFIG))
        with torch.no_grad(), compute_encodings(sim.model):
            sim.model(x)
        from aimet_torch.v2.nn.modules.custom import QuantizedErf
        modules = [m for m in sim.model.modules() if isinstance(m, QuantizedErf)]
        self.assertEqual(len(modules), 2)
        self.assertIsNot(modules[0].input_quantizers[0], modules[1].input_quantizers[0])
        self.assertFalse(torch.equal(modules[0].input_quantizers[0].get_scale(),
                                     modules[1].input_quantizers[0].get_scale()))
        with tempfile.TemporaryDirectory() as directory:
            export_onnx_json(sim, directory, 'reused', tuple(x.shape))


if __name__ == '__main__':
    unittest.main()
