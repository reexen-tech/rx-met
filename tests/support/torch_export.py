"""Shared calibrated Torch export models and temporary output fixture."""
import importlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from torch import nn

from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import compute_encodings
from tests.support.paths import REPO_ROOT

exporter = importlib.import_module("aimet_torch.rx_export.export_onnx_json")
CONFIG = REPO_ROOT / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.sigmoid(self.linear(x) + x)


class TorchExportFixture:
    def __init__(self, testcase):
        torch.manual_seed(17)
        self.tmp = tempfile.TemporaryDirectory()
        testcase.addCleanup(self.tmp.cleanup)
        self.x = torch.randn(2, 4)

    def sim(self, model=None, calibrate=True, *, x=None):
        x = self.x if x is None else x
        sim = QuantizationSimModel(prepare_model((model or Model()).eval()), x,
                                  quant_scheme='tf', config_file=str(CONFIG))
        if calibrate:
            with torch.no_grad(), compute_encodings(sim.model):
                sim.model(x)
        return sim

    def export(self, sim, *, x=None):
        x = self.x if x is None else x
        return exporter.export_onnx_json(sim, self.tmp.name, 'model', tuple(x.shape))

    def corrupt_final_encodings(self, mutate):
        original = exporter.export_recurrent_encodings
        def corrupt(model, onnx_path, enc_path):
            original(model, onnx_path, enc_path)
            path = Path(enc_path)
            document = json.loads(path.read_text())
            mutate(document)
            path.write_text(json.dumps(document))
        return patch.object(exporter, 'export_recurrent_encodings', side_effect=corrupt)
