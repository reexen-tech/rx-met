"""Small native recurrent ONNX models and deterministic GRU fixtures."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def gru_graph(directions=2, layout=0, initial_state=True):
    random = np.random.default_rng(42)
    weights = [numpy_helper.from_array(random.normal(0, .1, shape).astype(np.float32), name)
               for name, shape in (("custom_W", (directions, 6, 3)), ("custom_R", (directions, 6, 2)),
                                   ("custom_B", (directions, 12)))]
    xshape = [2, 4, 3] if layout else [4, 2, 3]
    hshape = [2, directions, 2] if layout else [directions, 2, 2]
    yshape = [2, 4, directions, 2] if layout else [4, directions, 2, 2]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, xshape)]
    feed = {"x": random.normal(0, .2, xshape).astype(np.float32)}
    if initial_state:
        inputs.append(helper.make_tensor_value_info("h0", TensorProto.FLOAT, hshape))
        feed["h0"] = random.normal(0, .1, hshape).astype(np.float32)
    node = helper.make_node("GRU", ["x", "custom_W", "custom_R", "custom_B", "", "h0" if initial_state else ""],
                            ["y", "state"], name="user/gru", hidden_size=2, linear_before_reset=1,
                            direction="bidirectional" if directions == 2 else "forward", layout=layout)
    model = helper.make_model(helper.make_graph([node], "native-gru", inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, yshape),
         helper.make_tensor_value_info("state", TensorProto.FLOAT, hshape)], weights),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    return model, feed


def lstm_graph(directions=2, layout=0, states=(True, True), bias=True):
    rng = np.random.default_rng(42)
    shapes = [("custom_W", (directions, 8, 3)), ("custom_R", (directions, 8, 2))]
    if bias:
        shapes.append(("custom_B", (directions, 16)))
    weights = [numpy_helper.from_array(rng.normal(0, .1, shape).astype(np.float32), name)
               for name, shape in shapes]
    xshape = [2, 4, 3] if layout else [4, 2, 3]
    hshape = [2, directions, 2] if layout else [directions, 2, 2]
    yshape = [2, 4, directions, 2] if layout else [4, directions, 2, 2]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, xshape)]
    feed = {"x": rng.normal(0, .2, xshape).astype(np.float32)}
    for name, enabled in zip(("h0", "c0"), states):
        if enabled:
            inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, hshape))
            feed[name] = rng.normal(0, .1, hshape).astype(np.float32)
    node = helper.make_node("LSTM", ["x", "custom_W", "custom_R", "custom_B" if bias else "", "",
                                    "h0" if states[0] else "", "c0" if states[1] else ""],
                            ["y", "hidden", "cell"], name="user/lstm", hidden_size=2,
                            direction="bidirectional" if directions == 2 else "forward", layout=layout)
    model = helper.make_model(helper.make_graph([node], "native-lstm", inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, yshape),
         helper.make_tensor_value_info("hidden", TensorProto.FLOAT, hshape),
         helper.make_tensor_value_info("cell", TensorProto.FLOAT, hshape)], weights),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    return model, feed


def fixed_gru_model(directory, clipped=None):
    import torch
    from quant_gru import QuantGRU

    m=QuantGRU(1,1).cuda();m.set_all_bitwidth(8)
    with torch.no_grad():
        m.weight_ih_l0.copy_(torch.tensor([[.2],[.1],[.3]],device='cuda'))
        m.weight_hh_l0.copy_(torch.tensor([[.1],[.1],[.2]],device='cuda'))
        m.bias_ih_l0.copy_(torch.tensor([.1,.1,.2],device='cuda'))
        m.bias_hh_l0.copy_(torch.tensor([.1,.1,.4],device='cuda'))
    m.calibrating=True
    with torch.no_grad():m(torch.tensor([[[.5]]],device='cuda'))
    m.calibrating=False;m.finalize_calibration()
    path=Path(directory)/'encodings.json';m.export_quant_params(str(path));data=json.loads(path.read_text())
    data['model_info']['use_pot2_scale']=True
    for name,e in data['operators'].items():
        scale=1/32 if name.endswith('_input') or name.endswith('_linear') else 1/128
        if name in ('update_gate_output','reset_gate_output'):scale=1/256
        if name==clipped:scale=1/4096
        channels=len(e['scale']) if isinstance(e['scale'],list) else None
        lo,hi=(0,255) if e['dtype'].startswith('U') else (-128,127)
        e.update(scale=[scale]*channels if channels else scale,zero_point=[0]*channels if channels else 0,
                 symmetric=True,real_min=[scale*lo]*channels if channels else scale*lo,
                 real_max=[scale*hi]*channels if channels else scale*hi)
    path.write_text(json.dumps(data));m.load_quant_params(str(path));m.use_quantization=True;m.train()
    return m,data
