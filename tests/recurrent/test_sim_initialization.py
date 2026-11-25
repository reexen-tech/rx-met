"""QuantSim initializes native recurrent quantization without a stage JSON."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import onnx
import torch
from torch import nn
import quant_gru
import _quant_lstm
from quant_lstm import QuantLSTM

from aimet_common.defs import QuantizationDataType
from aimet_torch.model_preparer import prepare_model
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_torch.staged_quantization_utils import save_quantizer_encodings
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
from aimet_torch.v2.quantsim import QuantizationSimModel

from tests.support.paths import REPO_ROOT


CONFIG = REPO_ROOT / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"


class RecurrentModel(nn.Module):
    def __init__(self, operator_type, bidirectional=False):
        super().__init__()
        self.rnn = operator_type(3, 4, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(8 if bidirectional else 4, 2)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.head(output)


class NativeRecurrentSimInitializationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        temporary = tempfile.TemporaryDirectory(prefix="recurrent-sim-initialization-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def fixture(self, kind, *, bidirectional=False, config_file=str(CONFIG), **kwargs):
        if kind == "GRU":
            operator_type = quant_gru.QuantGRU
            cpu_only = quant_gru._CPU_ONLY
        else:
            operator_type = QuantLSTM
            cpu_only = not getattr(_quant_lstm, "supports_cuda", True)
        if not cpu_only and not torch.cuda.is_available():
            self.skipTest(f"{kind} execution requires a CPU wheel or CUDA")
        device = "cpu" if cpu_only else "cuda"
        x = torch.randn(2, 5, 3, device=device) * .3 + .15
        model = RecurrentModel(operator_type, bidirectional).to(device).eval()
        self.assertFalse(model.rnn.use_quantization)
        prepared = prepare_model(model)
        self.assertFalse(prepared.rnn.use_quantization)
        with torch.no_grad():
            baseline = prepared(x)
        sim = QuantizationSimModel(
            prepared, x, quant_scheme="tf", config_file=config_file, **kwargs,
        )
        return sim, prepared, x, baseline

    def native_forward(self, kind):
        if kind == "GRU":
            owner, name = quant_gru.gru_ops, "forward_quant_float_storage"
        else:
            owner, name = _quant_lstm, "lstm_forward_quantized"
        return patch.object(owner, name, wraps=getattr(owner, name))

    def apply_config(self, sim, document):
        path = self.directory / "stage.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        apply_mixed_precision_bitwidth(sim.model, str(path), verbose=False)

    def test_json1_only_quantizes_and_exports_native_layers(self):
        for kind in ("GRU", "LSTM"):
            for bidirectional in (False, True):
                with self.subTest(kind=kind, bidirectional=bidirectional):
                    sim, original, x, baseline = self.fixture(
                        kind, bidirectional=bidirectional,
                    )
                    native = sim.model.rnn
                    self.assertTrue(native.use_quantization)
                    self.assertFalse(native.is_calibrated())
                    self.assertTrue(all(q is None for q in native.input_quantizers))
                    self.assertTrue(all(q is None for q in native.output_quantizers))
                    self.assertTrue(all(q is None for q in native.param_quantizers.values()))

                    sim.compute_encodings(lambda model: model(x))
                    self.assertTrue(native.is_calibrated())
                    self.assertFalse(native.calibrating)
                    self.assertTrue(sim.model.head.param_quantizers["weight"].is_initialized())
                    with torch.no_grad(), self.native_forward(kind) as forward:
                        result = sim.model(x)
                    self.assertEqual(forward.call_count, 2 if bidirectional else 1)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertFalse(original.rnn.use_quantization)
                    self.assertFalse(original.rnn.is_calibrated())
                    with torch.no_grad():
                        torch.testing.assert_close(original(x), baseline, rtol=0, atol=0)

                    prefix = f"{kind.lower()}_{'bi' if bidirectional else 'uni'}"
                    graph_path, encodings_path = export_onnx_json(
                        sim, str(self.directory), prefix, tuple(x.shape),
                    )
                    graph = onnx.load(graph_path)
                    onnx.checker.check_model(graph)
                    self.assertEqual(sum(node.op_type == kind for node in graph.graph.node), 1)
                    document = json.loads(Path(encodings_path).read_text(encoding="utf-8"))
                    self.assertNotIn("rnn", document.get("excluded_layers", []))
                    entry = document["activation_encodings"]["rnn"]
                    self.assertTrue(entry["internal_ops"])
                    self.assertEqual(bool(entry.get("internal_ops_reverse")), bidirectional)
                    self.assertTrue(any(name.startswith("rnn.") for name in document["param_encodings"]))

    def test_stage_overrides_and_sparse_configs_survive_recalibration(self):
        for kind in ("GRU", "LSTM"):
            for enabled in (True, False):
                with self.subTest(kind=kind, enabled=enabled):
                    sim, _, x, _ = self.fixture(kind, config_file=str(CONFIG))
                    if not enabled:
                        self.apply_config(sim, {
                            "GRU_config": {"default_config": {"disable_quantization": True}},
                            "LSTM_config": {"use_quantization": False},
                        })
                    for document in (
                        {},
                        {"GRU_config": {}, "LSTM_config": {}},
                        {
                            "GRU_config": {"operator_config": {"input": {"bitwidth": 16}}},
                            "LSTM_config": {"quant_config": {
                                "schema_version": 1, "operators": {"input": {"bitwidth": 16}},
                            }},
                        },
                    ):
                        self.apply_config(sim, document)
                        self.assertEqual(sim.model.rnn.use_quantization, enabled)
                    for factor in (1.0, .7):
                        sim.compute_encodings(lambda model: model(x * factor))
                        self.assertEqual(sim.model.rnn.use_quantization, enabled)
                        with torch.no_grad(), self.native_forward(kind) as forward:
                            result = sim.model(x)
                        self.assertEqual(forward.call_count, int(enabled))
                        self.assertTrue(torch.isfinite(result).all())
                    document = save_quantizer_encodings(
                        sim.model, str(self.directory / "encodings.json"), verbose=False,
                    )
                    self.assertEqual("rnn" in document.get("activation_encodings", {}), enabled)

    def test_float_sim_does_not_enable_native_integer_quantization(self):
        for kind in ("GRU", "LSTM"):
            with self.subTest(kind=kind):
                sim, _, x, _ = self.fixture(
                    kind, config_file=str(CONFIG), default_data_type=QuantizationDataType.float,
                    default_output_bw=16, default_param_bw=16,
                )
                self.assertFalse(sim.model.rnn.use_quantization)
                sim.compute_encodings(lambda model: model(x))
                with torch.no_grad(), self.native_forward(kind) as forward:
                    result = sim.model(x)
                self.assertEqual(forward.call_count, 0)
                self.assertFalse(sim.model.rnn.use_quantization)
                self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
