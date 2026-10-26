"""CPU public API coverage; the CUDA target remains the numerical reference."""

import json
from unittest.mock import patch

import pytest
import torch

import gru_interface_binding as native
from quant_gru import QuantGRU

pytestmark = pytest.mark.skipif(not getattr(native, "cpu_only", False), reason="CPU wheel required")


@pytest.mark.parametrize("bidirectional,batch_first,bias", [(False, False, True), (True, True, False)])
def test_cpu_float_and_calibration_match_torch(bidirectional, batch_first, bias):
    torch.manual_seed(71)
    baseline = torch.nn.GRU(3, 4, bidirectional=bidirectional, batch_first=batch_first, bias=bias).eval()
    model = QuantGRU(3, 4, bidirectional=bidirectional, batch_first=batch_first, bias=bias).eval()
    model.load_state_dict(baseline.state_dict())
    x = torch.randn((2, 7, 3) if batch_first else (7, 2, 3))
    h0 = torch.randn(2 if bidirectional else 1, 2, 4) * 0.1
    with torch.no_grad():
        expected = baseline(x, h0)
        actual = model(x, h0)
        model.calibrating = True
        calibrated = model(x, h0)
    for reference, output, observed in zip(expected, actual, calibrated):
        torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(observed, output, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["minmax", "sqnr", "percentile"])
@pytest.mark.parametrize("bits,pot2", [(8, False), (8, True), (16, False), (16, True)])
def test_cpu_ptq_integer_carrier_and_encoding_roundtrip(method, bits, pot2, tmp_path):
    torch.manual_seed(39)
    model = QuantGRU(3, 4).eval()
    model.set_all_bitwidth(bits, is_symmetric=False)
    model.calibration_method = method
    if pot2:
        model.enable_pot2()
    x = torch.randn(9, 2, 3) * 0.3 + 0.15
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA was initialized")):
        with model.calibration_context(), torch.no_grad():
            model(x)
            model(x * 1.1)
        model.use_quantization = True
        with torch.no_grad():
            floating = model(x)
            model.quant_storage_dtype = "int32"
            integer = model(x)
        for a, b in zip(floating, integer):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

        # Integer input is already on the public input grid; no second quantization.
        meta = model.get_io_quant_meta()
        x_meta = meta["input"]
        lower = 0 if x_meta["is_unsigned"] else -(1 << (bits - 1))
        upper = (1 << bits) - 1 if x_meta["is_unsigned"] else (1 << (bits - 1)) - 1
        codes = torch.round(x / x_meta["scale"] + x_meta["zp"]).clamp(lower, upper).to(torch.int32)
        output_q, state_q = model.forward_quantized(codes)
        y_meta = meta["output"]
        for actual_q, reference in zip((output_q, state_q), integer):
            dequantized = (actual_q.float() - y_meta["zp"]) * y_meta["scale"]
            torch.testing.assert_close(dequantized, reference, atol=0, rtol=0)

        path = tmp_path / "encodings.json"
        model.export_quant_params(str(path))
        assert json.loads(path.read_text())["operators"]
        target = QuantGRU(3, 4).eval()
        target.load_state_dict(model.state_dict())
        target.load_quant_params(str(path))
        target.use_quantization = True
        with torch.no_grad():
            restored = target(x)
        for a, b in zip(restored, integer):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_cpu_rejects_training_and_qat():
    model = QuantGRU(3, 4)
    with pytest.raises(RuntimeError, match="call eval"):
        model(torch.randn(5, 2, 3))
    with pytest.raises(RuntimeError, match="QAT"):
        model.aimet_configure("int16_fixed_qat_sim")
    assert model.aimet_capabilities()["forward_io_device"] == "cpu"
    assert "int16_fixed_qat_sim" not in model.aimet_capabilities()["supported_modes"]


@pytest.mark.parametrize("calibrating", [False, True])
@pytest.mark.parametrize("misuse", ["training", "grad_enabled"])
def test_cpu_forward_rejects_training_and_grad_for_every_mode(calibrating, misuse):
    model = QuantGRU(3, 4).eval()
    model.calibrating = calibrating
    x = torch.randn(5, 2, 3)
    h0 = torch.zeros(1, 2, 4)
    if misuse == "training":
        model.train()
    context = torch.enable_grad() if misuse == "grad_enabled" else torch.no_grad()
    with context, pytest.raises(RuntimeError, match="call eval" if misuse == "training" else "backward"):
        model(x, h0)
    assert model.quant_ranges is None
    assert model.hist_collectors is None


@pytest.mark.parametrize("calibrating", [False, True])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_cpu_accepts_grad_tensors_when_autograd_is_disabled(calibrating, context):
    model = QuantGRU(3, 4).eval()
    model.calibrating = calibrating
    x = torch.randn(5, 2, 3, requires_grad=True)
    h0 = torch.zeros(1, 2, 4, requires_grad=True)
    with context():
        output, state = model(x, h0)
    assert output.shape == (5, 2, 4)
    assert state.shape == (1, 2, 4)
    assert not output.requires_grad
    assert not state.requires_grad
    assert (model.quant_ranges is not None) == calibrating


def test_cpu_export_path_bypasses_inference_context_checks(monkeypatch):
    model = QuantGRU(3, 4).train()
    model.export_mode = True
    model.calibrating = True
    x = torch.randn(5, 2, 3, requires_grad=True)
    expected = (torch.zeros(5, 2, 4), torch.zeros(1, 2, 4))
    monkeypatch.setattr(model, "_forward_onnx_gru", lambda value, state: expected)
    with torch.enable_grad():
        output, state = model(x)
    assert output is expected[0]
    assert state is expected[1]


@pytest.mark.parametrize("method", ["minmax", "sqnr", "percentile"])
@pytest.mark.parametrize("granularity", [0, 1, 2])
def test_cpu_weight_granularities_and_bidirectional_reload(method, granularity, tmp_path):
    torch.manual_seed(13)
    model = QuantGRU(3, 2, bidirectional=True, batch_first=True).eval()
    model.calibration_method = method
    for name in ("W", "R", "bw", "br"):
        setattr(model._bitwidth_config, name + "_granularity_", granularity)
    x = torch.randn(2, 5, 3) * 0.2
    with model.calibration_context(), torch.no_grad():
        model(x)
    model.use_quantization = True
    with torch.no_grad():
        expected = model(x)
    encoding = model.export_quant_params_to_aimet_format({}, module_name="gru")
    target = QuantGRU(3, 2, bidirectional=True, batch_first=True).eval()
    target.load_state_dict(model.state_dict())
    assert target.load_quant_params_from_aimet_format(encoding, module_name="gru")
    target.use_quantization = True
    with torch.no_grad():
        restored = target(x)
    for a, b in zip(expected, restored):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
