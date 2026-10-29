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
def test_cpu_forward_rejects_training_for_every_mode(calibrating):
    model = QuantGRU(3, 4).eval()
    model.calibrating = calibrating
    x = torch.randn(5, 2, 3)
    h0 = torch.zeros(1, 2, 4)
    model.train()
    with torch.no_grad(), pytest.raises(RuntimeError, match="call eval"):
        model(x, h0)
    assert model.quant_ranges is None
    assert model.hist_collectors is None


def _cpu_model_for_mode(mode, *, bidirectional=False, batch_first=False, bias=True):
    model = QuantGRU(3, 4, bidirectional=bidirectional, batch_first=batch_first, bias=bias).eval()
    x = torch.randn((2, 5, 3) if batch_first else (5, 2, 3)) * 0.2
    h0 = torch.randn(2 if bidirectional else 1, 2, 4) * 0.1
    if mode == "quant":
        with model.calibration_context(), torch.no_grad():
            model(x, h0)
        model.use_quantization = True
    else:
        model.calibrating = mode == "calibrate"
    return model, x, h0


@pytest.mark.parametrize("mode", ["float", "calibrate", "quant"])
@pytest.mark.parametrize("bidirectional,batch_first,bias", [
    (False, False, True), (False, True, False), (True, False, False), (True, True, True),
])
def test_cpu_eval_allows_grad_enabled_forward_and_rejects_backward(mode, bidirectional, batch_first, bias):
    torch.manual_seed(83)
    model, x, h0 = _cpu_model_for_mode(mode, bidirectional=bidirectional, batch_first=batch_first, bias=bias)
    with torch.no_grad():
        expected = model(x, h0)
    with torch.enable_grad():
        actual = model(x, h0)
    for reference, output in zip(expected, actual):
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        assert output.requires_grad
        assert output.grad_fn.saved_tensors == ()
        with pytest.raises(RuntimeError, match="CPU QuantGRU does not support backward"):
            output.sum().backward()
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("mode", ["float", "calibrate", "quant"])
@pytest.mark.parametrize("dependency", ["input", "h0", "parameter"])
@pytest.mark.parametrize("result_index", [0, 1])
def test_cpu_backward_guard_tracks_each_kind_of_dependency(mode, dependency, result_index):
    model, x, h0 = _cpu_model_for_mode(mode)
    model.requires_grad_(False)
    if dependency == "input":
        target = x.requires_grad_()
    elif dependency == "h0":
        target = h0.requires_grad_()
    else:
        target = model.weight_ih_l0.requires_grad_()
    with torch.enable_grad():
        result = model(x, h0)[result_index]
        with pytest.raises(RuntimeError, match="CPU QuantGRU does not support backward"):
            torch.autograd.grad(result.sum(), target)
    assert target.grad is None


@pytest.mark.parametrize("result_index", [0, 1])
def test_cpu_backward_guard_tracks_every_bidirectional_parameter(result_index):
    model, x, h0 = _cpu_model_for_mode("float", bidirectional=True)
    for _, parameter in model.named_parameters():
        model.requires_grad_(False)
        parameter.requires_grad_(True)
        with torch.enable_grad():
            result = model(x, h0)[result_index]
            with pytest.raises(RuntimeError, match="CPU QuantGRU does not support backward"):
                torch.autograd.grad(result.sum(), parameter)


@pytest.mark.parametrize("upstream_trainable", [False, True])
def test_cpu_backward_guard_prevents_silent_gradient_loss_in_mixed_network(upstream_trainable):
    model, x, _ = _cpu_model_for_mode("float")
    upstream = torch.nn.Linear(3, 3)
    downstream = torch.nn.Linear(4, 2)
    if upstream_trainable:
        model.requires_grad_(False)
    else:
        upstream.requires_grad_(False)
    with torch.enable_grad():
        result, _ = model(upstream(x))
        with pytest.raises(RuntimeError, match="CPU QuantGRU does not support backward"):
            downstream(result).sum().backward()


@pytest.mark.parametrize("mode", ["float", "calibrate", "quant"])
@pytest.mark.parametrize("bidirectional,batch_first", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("result_index", [0, 1])
def test_cpu_eval_allows_downstream_inplace_forward(mode, bidirectional, batch_first, result_index):
    model, x, h0 = _cpu_model_for_mode(mode, bidirectional=bidirectional, batch_first=batch_first)
    with torch.enable_grad():
        result = torch.nn.ReLU(inplace=True)(model(x, h0)[result_index])
        with pytest.raises(RuntimeError, match="CPU QuantGRU does not support backward"):
            result.sum().backward()


@pytest.mark.parametrize("boundary", ["detach", "frozen", "no_grad"])
def test_cpu_features_allow_explicitly_separate_trainable_head(boundary):
    model, x, h0 = _cpu_model_for_mode("float")
    head = torch.nn.Linear(4, 2)
    if boundary == "frozen":
        model.requires_grad_(False)
    with torch.enable_grad():
        if boundary == "no_grad":
            with torch.no_grad():
                features, _ = model(x, h0)
        else:
            features, _ = model(x, h0)
        if boundary == "detach":
            features = features.detach()
        assert not features.requires_grad
        assert features.grad_fn is None
        head(features).sum().backward()
    assert head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("mode", ["float", "calibrate", "quant"])
@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_cpu_accepts_grad_tensors_when_autograd_is_disabled(mode, context):
    model, x, h0 = _cpu_model_for_mode(mode)
    x.requires_grad_()
    h0.requires_grad_()
    with context():
        output, state = model(x, h0)
    assert output.shape == (5, 2, 4)
    assert state.shape == (1, 2, 4)
    assert not output.requires_grad
    assert not state.requires_grad
    assert output.grad_fn is None
    assert state.grad_fn is None
    assert (model.quant_ranges is not None) == (mode != "float")


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
