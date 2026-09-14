#!/usr/bin/env bash
# 在最终产品镜像中使用真实 Speech Commands 数据完整运行 KWS/QAT 示例。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
IMAGE_REPOSITORY="${RX_MET_IMAGE_REPOSITORY:-rx-met}"
DATASET_DIR=""
OUTPUT_ROOT="${RX_MET_KWS_VALIDATION_OUTPUT:-${ROOT}/.release/example-validation/v${VERSION}/quick_start_kws}"
GPU_DEVICE="${RX_MET_VERIFY_GPU_DEVICE:-0}"
TARGETS=()

log() { printf '[verify_kws_example] %s\n' "$*"; }
die() { printf '[verify_kws_example] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
用法:
  ./scripts/verify_kws_example.sh --dataset-dir DIR [选项] [cu118|cu126|cu130 ...]

必填参数:
  --dataset-dir DIR      Speech Commands v0.02 解压后的数据集目录

可选参数:
  --output-dir DIR       输出根目录
                         默认: ${OUTPUT_ROOT}
  --gpu-device ID        仅向容器开放指定 GPU，默认: ${GPU_DEVICE}
  -h, --help             显示帮助

不指定 CUDA 变体时依次验证 cu118、cu126、cu130。每个变体的产物和日志写入
输出根目录下对应的 cu118、cu126 或 cu130 子目录。KWS 示例会自行训练模型，
不需要传入已有模型。
EOF
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || die "${option} 缺少参数值"
    printf '%s' "${value}"
}

while (($#)); do
    case "$1" in
        --dataset-dir|--speech-commands-dir)
            DATASET_DIR="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --output-dir)
            OUTPUT_ROOT="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --gpu-device)
            GPU_DEVICE="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        cu118|cu126|cu130)
            TARGETS+=("$1")
            shift
            ;;
        *) die "未知参数: $1" ;;
    esac
done

[[ -n "${DATASET_DIR}" ]] || die "必须传入 --dataset-dir"
[[ -n "${GPU_DEVICE}" ]] || die "--gpu-device 不能为空"
[[ "${IMAGE_REPOSITORY}" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || die "无效的镜像仓库名: ${IMAGE_REPOSITORY}"
for command in docker realpath stat tee; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done

DATASET_DIR="$(realpath -e -- "${DATASET_DIR}")"
[[ -d "${DATASET_DIR}" ]] || die "数据集目录不存在: ${DATASET_DIR}"
[[ -d "${DATASET_DIR}/_background_noise_" ]] \
    || die "不是有效的 Speech Commands 数据目录（缺少 _background_noise_）: ${DATASET_DIR}"
mkdir -p -- "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(realpath -- "${OUTPUT_ROOT}")"
[[ -w "${OUTPUT_ROOT}" ]] || die "输出目录不可写: ${OUTPUT_ROOT}"

if ((${#TARGETS[@]} == 0)); then
    TARGETS=(cu118 cu126 cu130)
fi
declare -A seen_targets=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen_targets[${target}]:-}" ]] || die "目标重复: ${target}"
    seen_targets["${target}"]=1
done

user_id="$(id -u)"
group_id="$(id -g)"
dataset_group_id="$(stat -c '%g' "${DATASET_DIR}")"

for target in "${TARGETS[@]}"; do
    image="${IMAGE_REPOSITORY}:${VERSION}-${target}"
    docker image inspect "${image}" >/dev/null 2>&1 \
        || die "本地产品镜像不存在: ${image}"

    target_output="${OUTPUT_ROOT}/${target}"
    mkdir -p -- "${target_output}"
    [[ -w "${target_output}" ]] || die "输出目录不可写: ${target_output}"
    log "运行 ${image}，GPU=${GPU_DEVICE}"
    log "输出目录: ${target_output}"

    docker run --rm \
        --gpus "device=${GPU_DEVICE}" \
        --shm-size=2g \
        --user "${user_id}:${group_id}" \
        --group-add "${dataset_group_id}" \
        -e HOME=/tmp \
        -e USER=rx-met-validator \
        -e LOGNAME=rx-met-validator \
        -e "RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands" \
        -e "RX_MET_KWS_OUTPUT_DIR=/output" \
        -e "RX_MET_KWS_FP_MODEL=/output/model_fp_kws.pth" \
        -v "${DATASET_DIR}:/datasets/speech_commands:ro" \
        -v "${target_output}:/output" \
        "${image}" \
        python3 /opt/rx-met/examples/quick_start_kws.py \
        2>&1 | tee "${target_output}/verify.log"
done

log "全部 KWS example 验证通过: ${OUTPUT_ROOT}"
