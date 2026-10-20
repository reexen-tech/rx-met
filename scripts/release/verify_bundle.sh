#!/usr/bin/env bash
# Verify the delivered installation in a clean runtime with a read-only application mount.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/lib/installed_bundle.sh"
VERIFY_GPU="${RX_MET_VERIFY_GPU:-0}"
GPU_DEVICE="${RX_MET_VERIFY_GPU_DEVICE:-0}"
TARGETS=()
while (($#)); do
    case "$1" in
        --gpu) VERIFY_GPU=1; shift ;;
        --no-gpu) VERIFY_GPU=0; shift ;;
        --gpu-device|--install-prefix)
            (($# >= 2)) || { echo "$1 缺少参数" >&2; exit 1; }
            if [[ "$1" == --gpu-device ]]; then GPU_DEVICE="$2"; else INSTALL_PREFIX="$2"; fi
            shift 2 ;;
        cu118|cu126|cu130) TARGETS+=("$1"); shift ;;
        -h|--help) echo '用法: verify_bundle.sh [--gpu] [--gpu-device ID] [--install-prefix DIR] [cu118|cu126|cu130 ...]'; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done
((${#TARGETS[@]})) || TARGETS=(cu118 cu126 cu130)
for target in "${TARGETS[@]}"; do
    rx_met_resolve_installation "$target"
    values="$(python3 - "$RX_MET_INSTALLED_DIR/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
c = m['compatibility']
p = c['packages']
print('\t'.join([m['variant'], c['cuda_version'], *[p[n].split('+')[0] for n in ('torch','torchvision','torchaudio')], p['onnxruntime-gpu'], c['python_version']]))
PY
)"
    IFS=$'\t' read -r variant cuda_version torch_version vision_version audio_version onnxruntime_version python_version <<< "$values"
    docker_args=(--rm -i --network none --read-only --tmpfs /tmp:rw,exec,nosuid,size=2g
        --user "$(id -u):$(id -g)" -e HOME=/tmp -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e USER=rx-met-validator -e LOGNAME=rx-met-validator -e PYTHONPATH=/opt/rx-met/python
        -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,src=$RX_MET_INSTALLED_DIR,dst=/opt/rx-met,readonly"
        -e "EXPECTED_VARIANT=$variant" -e "EXPECTED_CUDA=$cuda_version"
        -e "EXPECTED_TORCH=$torch_version" -e "EXPECTED_TORCHVISION=$vision_version"
        -e "EXPECTED_TORCHAUDIO=$audio_version" -e "EXPECTED_ONNXRUNTIME=$onnxruntime_version"
        -e "EXPECTED_PYTHON=$python_version" -e "VERIFY_GPU=$VERIFY_GPU")
    if ((VERIFY_GPU)); then docker_args+=(--gpus "device=$GPU_DEVICE"); fi
    printf 'Verify %s GPU=%s\n' "$RX_MET_INSTALLED_DIR" "$VERIFY_GPU"
    docker run "${docker_args[@]}" "$RX_MET_RUNTIME_ID" python3 - <<'PY'
import importlib.metadata
import os
import runpy
import subprocess
import sys
from pathlib import Path
import site

import aimet_onnx
import aimet_torch
import numpy as np
import onnx
import onnxruntime
import quant_gru
import quant_lstm
import torch
import torchaudio
import torchvision

for module in (aimet_onnx, aimet_torch, quant_gru, quant_lstm):
    assert Path(module.__file__).resolve().is_relative_to(Path("/opt/rx-met/python")), module.__file__
expected = {
    "torch": os.environ["EXPECTED_TORCH"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "torchaudio": os.environ["EXPECTED_TORCHAUDIO"],
}
for package, version in expected.items():
    actual = importlib.metadata.version(package).split("+", 1)[0]
    assert actual == version, (package, actual, version)
assert torch.version.cuda == os.environ["EXPECTED_CUDA"], torch.version.cuda
environment = dict(line.split("=", 1) for line in Path("/opt/rx-quant-env/environment.env").read_text().splitlines())
assert environment["cuda_variant"] == os.environ["EXPECTED_VARIANT"]
assert importlib.metadata.version("quant-lstm").endswith("+" + os.environ["EXPECTED_VARIANT"])
assert importlib.metadata.version("onnxruntime-gpu") == os.environ["EXPECTED_ONNXRUNTIME"]
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
assert f"{sys.version_info.major}.{sys.version_info.minor}" == os.environ["EXPECTED_PYTHON"]

runpy.run_path("/opt/rx-met/examples/quick_start_kws.py", run_name="verify_rx_met")
runpy.run_path("/opt/rx-met/examples/quick_start_onnx_ptq.py", run_name="verify_rx_met")

# 使用真实 AIMET ONNX custom op 跑一个最小校准和推理。
input_info = onnx.helper.make_tensor_value_info(
    "input", onnx.TensorProto.FLOAT, [1, 4]
)
output_info = onnx.helper.make_tensor_value_info(
    "output", onnx.TensorProto.FLOAT, [1, 4]
)
graph = onnx.helper.make_graph(
    [onnx.helper.make_node("Relu", ["input"], ["output"])],
    "rx_met_smoke",
    [input_info],
    [output_info],
)
model = onnx.helper.make_model(
    graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
)
model.ir_version = 8
# QuantizationSimModel 会原地插入 custom op；普通 CUDA session 使用原始模型。
plain_model_bytes = model.SerializeToString()
feed = {"input": np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)}
onnx_provider = (
    "CUDAExecutionProvider" if os.environ["VERIFY_GPU"] == "1"
    else "CPUExecutionProvider"
)
onnx_sim = aimet_onnx.QuantizationSimModel(
    model, dummy_input=feed, providers=[onnx_provider]
)
onnx_sim.compute_encodings([feed])
assert onnx_sim.session.run(None, feed)[0].shape == (1, 4)
assert onnx_sim.session.get_providers()[0] == onnx_provider

site_roots = [Path("/opt/rx-met/python"), *[Path(value) for value in site.getsitepackages()]]
patterns = [
    (site_roots, "aimet_common/_libpymo*.so"),
    (site_roots, "aimet_common/libquant_info*.so"),
    (site_roots, "aimet_common/libaimet_onnxrt_ops.so"),
    (site_roots, "onnxruntime/capi/libonnxruntime_providers_cuda.so"),
    (site_roots, "gru_interface_binding*.so"),
    (site_roots, "_quant_lstm*.so"),
    (site_roots, "lib/libgru_quant_shared.so"),
]
libraries = []
for roots, relative_pattern in patterns:
    matches = [match for root in roots for match in root.glob(relative_pattern)]
    assert matches, f"缺少原生库: {relative_pattern}"
    libraries.extend(matches)
for library in libraries:
    output = subprocess.run(
        ["ldd", str(library)], check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    unresolved = [
        line.strip()
        for line in output.splitlines()
        if "not found" in line and "libcuda.so.1" not in line
    ]
    assert not unresolved, (library, unresolved)

if os.environ["VERIFY_GPU"] == "1":
    assert torch.cuda.is_available(), "容器内不可见 CUDA GPU"
    session_options = onnxruntime.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    cuda_session = onnxruntime.InferenceSession(
        plain_model_bytes,
        sess_options=session_options,
        providers=["CUDAExecutionProvider"],
    )
    assert cuda_session.get_providers()[0] == "CUDAExecutionProvider"
    assert cuda_session.run(None, feed)[0].shape == (1, 4)
    import torch.nn as nn
    import aimet_torch.v2 as aimet_v2

    dummy = torch.randn(2, 4, device="cuda")
    torch_sim = aimet_v2.quantsim.QuantizationSimModel(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()).cuda().eval(),
        dummy,
        config_file=(
            "/opt/rx-met/examples/config/"
            "mrnn_quantsim_config_custom_mixed_precision_v2.json"
        ),
    )
    with aimet_v2.nn.compute_encodings(torch_sim.model):
        torch_sim.model(dummy)
    assert torch.isfinite(torch_sim.model(dummy)).all()

    from quant_gru import QuantGRU

    model = QuantGRU(4, 8, batch_first=True).cuda()
    value = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    output, hidden = model(value)
    (output.square().mean() + hidden.square().mean()).backward()
    torch.cuda.synchronize()
    assert value.grad is not None
    print("GPU", torch.cuda.get_device_name(0), "QuantGRU forward/backward OK")

    from quant_lstm import QuantLSTM
    from aimet_torch.model_preparer import prepare_model

    class LstmSmoke(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = QuantLSTM(4, 8, batch_first=True, bidirectional=True)
            self.head = torch.nn.Linear(16, 2)

        def forward(self, value):
            sequence, _ = self.lstm(value)
            return self.head(sequence)

    lstm_sim = aimet_v2.quantsim.QuantizationSimModel(
        prepare_model(LstmSmoke().cuda()), value.detach(), quant_scheme="tf_enhanced",
        config_file=(
            "/opt/rx-met/examples/config/"
            "mrnn_quantsim_config_custom_mixed_precision_v2.json"
        ),
    )
    lstm_sim.model.lstm.use_quantization = True
    with torch.no_grad(), aimet_v2.nn.compute_encodings(lstm_sim.model):
        lstm_sim.model(value.detach())
    assert lstm_sim.model.lstm.is_calibrated()
    value = value.detach().requires_grad_()
    lstm_sim.model(value).square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    print("QuantLSTM QuantSim calibration/forward/backward OK")

print("image verification passed", torch.__version__, torch.version.cuda)
PY
    docker run "${docker_args[@]}" "$RX_MET_RUNTIME_ID" python3 -m pip check
done
