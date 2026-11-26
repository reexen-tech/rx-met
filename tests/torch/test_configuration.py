"""Shared JSON1 operator names and built-in defaults work in Torch QuantSim."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from aimet_torch.nn.modules.custom import Mean
from aimet_torch.v2.quantsim import QuantizationSimModel
from tests.support.paths import REPO_ROOT


CONFIG = REPO_ROOT / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"


class Reduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = Mean()

    def forward(self, x):
        return self.reduce(x, dim=1, keepdim=True)


class TorchConfigurationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        temporary = tempfile.TemporaryDirectory(prefix="torch-config-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.json"
        self.x = torch.tensor([[-.75, -.125, .25, 1.], [.5, -.5, 1.5, -1.]])

    def config(self, **sections):
        document = json.loads(CONFIG.read_text(encoding="utf-8"))
        document.update(sections)
        self.path.write_text(json.dumps(document), encoding="utf-8")
        return str(self.path)

    def test_builtin_defaults_construct_calibrate_and_execute(self):
        cases = (
            {},
            {"config_file": None},
            {"config_file": "default"},
            {"config_file": str(REPO_ROOT / "src/aimet_common/quantsim_config/default_config.json")},
        )
        for kwargs in cases:
            with self.subTest(**kwargs):
                sim = QuantizationSimModel(
                    nn.Sequential(nn.Linear(4, 2)).eval(), self.x, quant_scheme="tf", **kwargs,
                )
                layer = sim.model[0]
                self.assertIsNone(layer.param_quantizers["bias"])
                sim.compute_encodings(lambda model: model(self.x))
                for quantizer in (layer.param_quantizers["weight"],
                                  layer.input_quantizers[0], layer.output_quantizers[0]):
                    self.assertIsNotNone(quantizer)
                    self.assertTrue(quantizer.is_initialized())
                    self.assertEqual(quantizer.bitwidth, 8)
                with torch.no_grad():
                    result = sim.model(self.x)
                self.assertEqual(result.shape, (2, 2))
                self.assertTrue(torch.isfinite(result).all())

    def test_standard_onnx_rules_preserve_mean_and_reduce_mean_distinction(self):
        for disable_reduction in (False, True):
            with self.subTest(disable_reduction=disable_reduction):
                rules = {
                    "Mean": {"is_output_quantized": "False"},
                    "If": {"is_output_quantized": "False"},
                }
                if disable_reduction:
                    rules["ReduceMean"] = {"is_output_quantized": "False"}
                sim = QuantizationSimModel(
                    Reduction().eval(), self.x, quant_scheme="tf",
                    config_file=self.config(op_type=rules),
                )
                self.assertEqual(sim.model.reduce.output_quantizers[0] is None, disable_reduction)
                sim.compute_encodings(lambda model: model(self.x))
                self.assertTrue(torch.isfinite(sim.model(self.x)).all())

    def test_unknown_operator_names_and_unsupported_supergroups_are_rejected(self):
        for name in ("Meen", "mean", "Convv"):
            with self.subTest(op_type=name):
                with self.assertRaisesRegex(AssertionError, f"Unsupported op type '{name}'"):
                    QuantizationSimModel(
                        nn.Sequential(nn.Linear(4, 2)).eval(), self.x, quant_scheme="tf",
                        config_file=self.config(op_type={name: {"is_output_quantized": "False"}}),
                    )
        for name in ("Mean", "If", "Convv"):
            with self.subTest(supergroup=name):
                with self.assertRaisesRegex(AssertionError, f"Unsupported op type '{name}'.*supergroups"):
                    QuantizationSimModel(
                        nn.Sequential(nn.Linear(4, 2)).eval(), self.x, quant_scheme="tf",
                        config_file=self.config(supergroups=[{"op_list": [name, "Relu"]}]),
                    )


if __name__ == "__main__":
    unittest.main()
