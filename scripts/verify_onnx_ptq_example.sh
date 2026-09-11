#!/usr/bin/env bash
# 在最终产品镜像中使用指定 ONNX 模型和真实校准数据运行 ONNX PTQ 示例。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
IMAGE_REPOSITORY="${RX_MET_IMAGE_REPOSITORY:-rx-met}"
MODEL_PATH=""
DATASET_DIR=""
OUTPUT_ROOT="${RX_MET_ONNX_PTQ_VALIDATION_OUTPUT:-${ROOT}/.release/example-validation/v${VERSION}/onnx_ptq}"
TARGETS=()

log() { printf '[verify_onnx_ptq_example] %s\n' "$*"; }
die() { printf '[verify_onnx_ptq_example] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
用法:
  ./scripts/verify_onnx_ptq_example.sh --model FILE --dataset-dir DIR [选项] [cu118|cu126|cu130 ...]

必填参数:
  --model FILE           待量化的 ONNX 模型
  --dataset-dir DIR      含 *.npy 或 *.npz 的校准数据目录

可选参数:
  --output-dir DIR       输出根目录
                         默认: ${OUTPUT_ROOT}
  -h, --help             显示帮助

不指定 CUDA 变体时依次验证 cu118、cu126、cu130。每个变体的产物和日志写入
输出根目录下对应的 cu118、cu126 或 cu130 子目录。当前 ONNX Runtime 为 CPU
版本，因此此示例不占用 GPU；KWS 脚本负责验证真实 CUDA 执行。
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
        --model|--model-path)
            MODEL_PATH="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --dataset-dir|--calib-dir)
            DATASET_DIR="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --output-dir)
            OUTPUT_ROOT="$(require_value "$1" "${2:-}")"
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

[[ -n "${MODEL_PATH}" ]] || die "必须传入 --model"
[[ -n "${DATASET_DIR}" ]] || die "必须传入 --dataset-dir"
[[ "${IMAGE_REPOSITORY}" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || die "无效的镜像仓库名: ${IMAGE_REPOSITORY}"
for command in docker realpath stat tee; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done

MODEL_PATH="$(realpath -e -- "${MODEL_PATH}")"
DATASET_DIR="$(realpath -e -- "${DATASET_DIR}")"
[[ -f "${MODEL_PATH}" ]] || die "模型文件不存在: ${MODEL_PATH}"
[[ -d "${DATASET_DIR}" ]] || die "校准数据目录不存在: ${DATASET_DIR}"
shopt -s nullglob
calibration_files=("${DATASET_DIR}"/*.npy "${DATASET_DIR}"/*.npz)
shopt -u nullglob
((${#calibration_files[@]} > 0)) \
    || die "校准目录中没有 *.npy 或 *.npz: ${DATASET_DIR}"
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
model_dir="$(dirname "${MODEL_PATH}")"
model_name="$(basename "${MODEL_PATH}")"
model_group_id="$(stat -c '%g' "${MODEL_PATH}")"
dataset_group_id="$(stat -c '%g' "${DATASET_DIR}")"

for target in "${TARGETS[@]}"; do
    image="${IMAGE_REPOSITORY}:${VERSION}-${target}"
    docker image inspect "${image}" >/dev/null 2>&1 \
        || die "本地产品镜像不存在: ${image}"

    target_output="${OUTPUT_ROOT}/${target}"
    mkdir -p -- "${target_output}"
    [[ -w "${target_output}" ]] || die "输出目录不可写: ${target_output}"
    log "运行 ${image}（ONNX Runtime CPUExecutionProvider）"
    log "输出目录: ${target_output}"

    docker run --rm \
        --shm-size=2g \
        --user "${user_id}:${group_id}" \
        --group-add "${model_group_id}" \
        --group-add "${dataset_group_id}" \
        -e HOME=/tmp \
        -e "RX_MET_ONNX_PTQ_MODEL=/model/${model_name}" \
        -e "RX_MET_ONNX_PTQ_CALIB=/datasets/calib" \
        -e "RX_MET_ONNX_PTQ_OUTPUT_DIR=/output" \
        -v "${model_dir}:/model:ro" \
        -v "${DATASET_DIR}:/datasets/calib:ro" \
        -v "${target_output}:/output" \
        "${image}" \
        python3 /opt/rx-met/examples/onnx_ptq_quick_start.py \
        2>&1 | tee "${target_output}/verify.log"
done

log "全部 ONNX PTQ example 验证通过: ${OUTPUT_ROOT}"
