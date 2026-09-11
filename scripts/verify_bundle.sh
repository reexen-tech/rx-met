#!/usr/bin/env bash
# 验证最终 rx-met 镜像；--gpu 额外运行真实 CUDA/QuantGRU 冒烟测试。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAKE_FILE="${ROOT}/docker/docker-bake.hcl"
VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
IMAGE_REPOSITORY="${RX_MET_IMAGE_REPOSITORY:-rx-met}"
VERIFY_GPU="${RX_MET_VERIFY_GPU:-0}"
TARGETS=()

log() { printf '[verify_bundle] %s\n' "$*"; }
die() { printf '[verify_bundle] ERROR: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --gpu) VERIFY_GPU=1 ;;
        --no-gpu) VERIFY_GPU=0 ;;
        -h|--help)
            printf '用法: ./scripts/verify_bundle.sh [--gpu|--no-gpu] [cu118|cu126|cu130 ...]\n'
            exit 0
            ;;
        cu118|cu126|cu130) TARGETS+=("$1") ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done
if ((${#TARGETS[@]} == 0)); then
    TARGETS=(cu118 cu126 cu130)
fi
declare -A seen_targets=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen_targets[${target}]:-}" ]] || die "目标重复: ${target}"
    seen_targets["${target}"]=1
done
[[ "${VERIFY_GPU}" == "0" || "${VERIFY_GPU}" == "1" ]] \
    || die "RX_MET_VERIFY_GPU 必须是 0 或 1"
[[ "${IMAGE_REPOSITORY}" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || die "无效的镜像仓库名: ${IMAGE_REPOSITORY}"
for command in docker git python3; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done

BAKE_JSON="$(mktemp -t rx-met-verify-bake.XXXXXX.json)"
cleanup() { rm -f "${BAKE_JSON}"; }
trap cleanup EXIT
VERSION="${VERSION}" IMAGE_REPOSITORY="${IMAGE_REPOSITORY}" \
SOURCE_REVISION="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)" \
docker buildx bake -f "${BAKE_FILE}" --print "${TARGETS[@]}" > "${BAKE_JSON}"

for target in "${TARGETS[@]}"; do
    values="$(python3 - "${BAKE_JSON}" "${target}" <<'PY'
import json
import sys

target = json.load(open(sys.argv[1], encoding="utf-8"))["target"][sys.argv[2]]
args = target["args"]
print(
    "\t".join(
        [
            target["tags"][0],
            args["CUDA_VARIANT"],
            args["RX_MET_CUDA_VERSION"],
            args["TORCH_VERSION"],
            args["TORCHVISION_VERSION"],
            args["TORCHAUDIO_VERSION"],
        ]
    )
)
PY
)"
    IFS=$'\t' read -r image variant cuda_version torch_version vision_version audio_version <<< "${values}"
    docker image inspect "${image}" >/dev/null 2>&1 || die "本地镜像不存在: ${image}"
    log "验证 ${image}（GPU=${VERIFY_GPU}）"

    docker_args=(-i --rm)
    if ((VERIFY_GPU)); then
        docker_args+=(--gpus all)
    fi
    docker_args+=(
        -e "EXPECTED_VARIANT=${variant}"
        -e "EXPECTED_CUDA=${cuda_version}"
        -e "EXPECTED_TORCH=${torch_version}"
        -e "EXPECTED_TORCHVISION=${vision_version}"
        -e "EXPECTED_TORCHAUDIO=${audio_version}"
        -e "VERIFY_GPU=${VERIFY_GPU}"
    )

    docker run "${docker_args[@]}" "${image}" python3 - <<'PY'
import importlib.metadata
import os
import runpy
import subprocess
from pathlib import Path

import aimet_onnx
import aimet_torch
import numpy as np
import onnx
import onnxruntime
import quant_gru
import torch
import torchaudio
import torchvision

expected = {
    "torch": os.environ["EXPECTED_TORCH"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "torchaudio": os.environ["EXPECTED_TORCHAUDIO"],
}
for package, version in expected.items():
    actual = importlib.metadata.version(package).split("+", 1)[0]
    assert actual == version, (package, actual, version)
assert torch.version.cuda == os.environ["EXPECTED_CUDA"], torch.version.cuda
assert os.environ["CUDA_VARIANT"] == os.environ["EXPECTED_VARIANT"]

runpy.run_path("/opt/rx-met/examples/quick_start_kws.py", run_name="verify_rx_met")
runpy.run_path("/opt/rx-met/examples/onnx_ptq_quick_start.py", run_name="verify_rx_met")
assert "CPUExecutionProvider" in onnxruntime.get_available_providers()

# 使用真实 AIMET ONNX custom op 跑一个最小 CPU 校准和推理。
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
feed = {"input": np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)}
onnx_sim = aimet_onnx.QuantizationSimModel(
    model, dummy_input=feed, providers=["CPUExecutionProvider"]
)
onnx_sim.compute_encodings([feed])
assert onnx_sim.session.run(None, feed)[0].shape == (1, 4)

site = Path("/usr/local/lib/python3.10/dist-packages")
patterns = [
    site / "aimet_common" / "_libpymo*.so",
    site / "aimet_common" / "libquant_info*.so",
    site / "aimet_common" / "libaimet_onnxrt_ops.so",
    site / "gru_interface_binding*.so",
    site / "lib" / "libgru_quant_shared.so",
]
libraries = []
for pattern in patterns:
    matches = list(pattern.parent.glob(pattern.name))
    assert matches, f"缺少原生库: {pattern}"
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

print("image verification passed", torch.__version__, torch.version.cuda)
PY
    docker run --rm "${image}" python3 -m pip check
done

log "全部目标验证通过"
