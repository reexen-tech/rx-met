"""The CPU ORT transport delegates calibration and PTQ to the native operator."""

from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest

import gru_interface_binding as native
from quant_gru_onnx import DOMAIN, OnnxGRU, runtime_library

pytestmark = pytest.mark.skipif(not getattr(native, "cpu_only", False), reason="CPU wheel required")


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("layout", [0, 1])
def test_cpu_onnx_transport_calibration_and_ptq(bidirectional, layout):
    import torch

    rng = np.random.default_rng(42)
    directions, hidden, inputs, batch, steps = 2 if bidirectional else 1, 3, 4, 2, 7
    weights = {
        "W": rng.normal(0, 0.2, (directions, 3 * hidden, inputs)).astype(np.float32),
        "R": rng.normal(0, 0.2, (directions, 3 * hidden, hidden)).astype(np.float32),
        "B": rng.normal(0, 0.1, (directions, 6 * hidden)).astype(np.float32),
    }
    initializers = {name: onnx.numpy_helper.from_array(value, name) for name, value in weights.items()}
    node = onnx.helper.make_node("GRU", ["x", "W", "R", "B", "", "h0"], ["y", "hn"],
        name="gru", hidden_size=hidden, linear_before_reset=1, layout=layout,
        direction="bidirectional" if bidirectional else "forward")
    adapter = OnnxGRU(node, initializers)
    assert adapter.module.weight_ih_l0.device.type == "cpu"
    x = rng.normal(0, 0.4, (steps, batch, inputs)).astype(np.float32)
    h0 = rng.normal(0, 0.1, (directions, batch, hidden)).astype(np.float32)
    if layout:
        x = x.transpose(1, 0, 2).copy()
        h0 = h0.transpose(1, 0, 2).copy()
    vi = lambda name, shape: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
    graph = onnx.helper.make_graph([adapter.runtime_node(node)], "native_gru",
        [vi("x", list(x.shape)), vi("h0", list(h0.shape))], [vi("y", None), vi("hn", None)])
    model = onnx.helper.make_model(graph, ir_version=10,
        opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid(DOMAIN, 1)])
    options = ort.SessionOptions()
    options.register_custom_ops_library(runtime_library())
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA was initialized")):
        with adapter.module.calibration_context():
            observed = session.run(None, {"x": x, "h0": h0})
        assert adapter.module.is_calibrated()
        assert adapter.export_encodings()["param_encodings"]
        assert all(np.isfinite(value).all() for value in observed)
        adapter.module.use_quantization = True
        expected = adapter.run(x, h0)
        actual = session.run(None, {"x": x, "h0": h0})
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
