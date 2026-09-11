#!/usr/bin/env bash
# 验证稳定 build-env/runtime-env 的身份、工具链和锁定依赖。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAKE_FILE="${ROOT}/docker/docker-bake.hcl"
source "${ROOT}/scripts/lib/environment_images.sh"
TARGETS=()

log() { printf '[verify_environment_images] %s\n' "$*"; }
die() { printf '[verify_environment_images] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
用法: ./scripts/verify_environment_images.sh [cu118|cu126|cu130 ...]

默认验证全部三个 CUDA 变体。该检查不使用 GPU。
EOF
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        cu118|cu126|cu130) TARGETS+=("$1") ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done
if ((${#TARGETS[@]} == 0)); then
    TARGETS=(cu118 cu126 cu130)
fi
rx_met_validate_environment_config || die "环境镜像名称或版本无效"
for command in docker python3; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done

export ENV_VERSION="${RX_MET_ENV_VERSION}"
export BUILD_ENV_REPOSITORY="${RX_MET_BUILD_ENV_REPOSITORY}"
export RUNTIME_ENV_REPOSITORY="${RX_MET_RUNTIME_ENV_REPOSITORY}"
BAKE_JSON="$(mktemp -t rx-met-environment-verify.XXXXXX.json)"
cleanup() { rm -f -- "${BAKE_JSON}"; }
trap cleanup EXIT
docker buildx bake -f "${BAKE_FILE}" --print "${TARGETS[@]}" > "${BAKE_JSON}"

declare -A seen=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen[${target}]:-}" ]] || die "目标重复: ${target}"
    seen["${target}"]=1
    values="$(python3 - "${BAKE_JSON}" "${target}" <<'PY'
import json
import sys

args = json.load(open(sys.argv[1], encoding="utf-8"))["target"][sys.argv[2]]["args"]
print("\t".join([
    args["RX_MET_CUDA_VERSION"],
    args["TORCH_VERSION"],
    args["TORCHVISION_VERSION"],
    args["TORCHAUDIO_VERSION"],
]))
PY
)"
    IFS=$'\t' read -r cuda_version torch_version vision_version audio_version \
        <<< "${values}"
    build_image="$(rx_met_build_environment_image "${target}")"
    runtime_image="$(rx_met_runtime_environment_image "${target}")"

    for image in "${build_image}" "${runtime_image}"; do
        docker image inspect "${image}" >/dev/null 2>&1 \
            || die "环境镜像不存在: ${image}"
        platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${image}")"
        [[ "${platform}" == "linux/amd64" ]] \
            || die "环境镜像平台错误: ${image}=${platform}"
    done

    log "验证 ${build_image}"
    docker run --rm \
        -e "EXPECTED_ROLE=build" \
        -e "EXPECTED_ENV_VERSION=${RX_MET_ENV_VERSION}" \
        -e "EXPECTED_VARIANT=${target}" \
        -e "EXPECTED_CUDA=${cuda_version}" \
        -e "EXPECTED_TORCH=${torch_version}" \
        -e "EXPECTED_TORCHVISION=${vision_version}" \
        -e "EXPECTED_TORCHAUDIO=${audio_version}" \
        "${build_image}" bash -e -u -o pipefail -c '
            command -v nvcc >/dev/null
            command -v cmake >/dev/null
            python3 -m pip check
            python3 - <<"PY"
import importlib.metadata
import os
from pathlib import Path
import torch

metadata = dict(
    line.split("=", 1)
    for line in Path("/opt/rx-met-env/environment.env").read_text().splitlines()
)
assert metadata["environment_role"] == os.environ["EXPECTED_ROLE"]
assert metadata["environment_version"] == os.environ["EXPECTED_ENV_VERSION"]
assert metadata["cuda_variant"] == os.environ["EXPECTED_VARIANT"]
assert metadata["cuda_version"] == os.environ["EXPECTED_CUDA"]
for package, expected in {
    "torch": os.environ["EXPECTED_TORCH"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "torchaudio": os.environ["EXPECTED_TORCHAUDIO"],
}.items():
    actual = importlib.metadata.version(package).split("+", 1)[0]
    assert actual == expected, (package, actual, expected)
assert torch.version.cuda == os.environ["EXPECTED_CUDA"]
PY
        '

    log "验证 ${runtime_image}"
    docker run --rm \
        -e "EXPECTED_ROLE=runtime" \
        -e "EXPECTED_ENV_VERSION=${RX_MET_ENV_VERSION}" \
        -e "EXPECTED_VARIANT=${target}" \
        -e "EXPECTED_CUDA=${cuda_version}" \
        -e "EXPECTED_TORCH=${torch_version}" \
        -e "EXPECTED_TORCHVISION=${vision_version}" \
        -e "EXPECTED_TORCHAUDIO=${audio_version}" \
        "${runtime_image}" bash -e -u -o pipefail -c '
            ! command -v nvcc >/dev/null
            python3 -m pip check
            python3 - <<"PY"
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import torch

metadata = dict(
    line.split("=", 1)
    for line in Path("/opt/rx-met-env/environment.env").read_text().splitlines()
)
assert metadata["environment_role"] == os.environ["EXPECTED_ROLE"]
assert metadata["environment_version"] == os.environ["EXPECTED_ENV_VERSION"]
assert metadata["cuda_variant"] == os.environ["EXPECTED_VARIANT"]
assert metadata["cuda_version"] == os.environ["EXPECTED_CUDA"]
for package, expected in {
    "torch": os.environ["EXPECTED_TORCH"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "torchaudio": os.environ["EXPECTED_TORCHAUDIO"],
}.items():
    actual = importlib.metadata.version(package).split("+", 1)[0]
    assert actual == expected, (package, actual, expected)
assert torch.version.cuda == os.environ["EXPECTED_CUDA"]
assert importlib.util.find_spec("aimet_torch") is None
assert importlib.util.find_spec("quant_gru") is None
PY
        '
done

log "全部环境镜像验证通过"
