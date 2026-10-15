#!/usr/bin/env bash
# 在纯 runtime + 只读应用挂载中使用真实 Speech Commands 数据完整运行 KWS/QAT 示例。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
source "$ROOT/scripts/lib/installed_bundle.sh"
DATASET_DIR=""
OUTPUT_ROOT="${RX_MET_KWS_VALIDATION_OUTPUT:-${ROOT}/.release/example-validation/v${VERSION}/quick_start_kws}"
GPU_DEVICE="${RX_MET_VERIFY_GPU_DEVICE:-0}"
RNN_TYPE="both"
TARGETS=()

log() { printf '[verify_kws_example] %s\n' "$*"; }
die() { printf '[verify_kws_example] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
用法:
  ./scripts/release/verify_kws_example.sh --dataset-dir DIR [选项] [cu118|cu126|cu130 ...]

必填参数:
  --dataset-dir DIR      Speech Commands v0.02 解压后的数据集目录

可选参数:
  --install-prefix DIR  应用安装根目录（默认 .release/installed）
  --output-dir DIR       输出根目录
                         默认: ${OUTPUT_ROOT}
  --rnn-type TYPE       gru、lstm 或 both（默认 both）
  --gpu-device ID        仅向容器开放指定 GPU，默认: ${GPU_DEVICE}
  -h, --help             显示帮助

不指定 CUDA 变体时依次验证 cu118、cu126、cu130，每个变体默认完整运行 GRU 和 LSTM。
产物和日志写入 <输出根目录>/<CUDA 变体>/<gru|lstm>/。KWS 示例会自行训练模型，
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
        --install-prefix)
            INSTALL_PREFIX="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --output-dir)
            OUTPUT_ROOT="$(require_value "$1" "${2:-}")"
            shift 2
            ;;
        --rnn-type)
            RNN_TYPE="$(require_value "$1" "${2:-}")"
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
case "${RNN_TYPE}" in
    both) RNN_TYPES=(gru lstm) ;;
    gru|lstm) RNN_TYPES=("${RNN_TYPE}") ;;
    *) die "--rnn-type 必须是 gru、lstm 或 both" ;;
esac
for command in docker python3 realpath stat tee; do
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
    rx_met_resolve_installation "$target" || die "缺少已安装的应用或 runtime: $target"
    image="$RX_MET_RUNTIME_ID"

    target_output="${OUTPUT_ROOT}/${target}"
    mkdir -p -- "${target_output}"
    [[ -w "${target_output}" ]] || die "输出目录不可写: ${target_output}"
    for rnn_type in "${RNN_TYPES[@]}"; do
        mkdir -p -- "${target_output}/${rnn_type}"
        log "运行 ${image}，RNN=${rnn_type}，GPU=${GPU_DEVICE}"
        log "输出目录: ${target_output}/${rnn_type}"

        docker run --rm -i --network none --read-only --tmpfs /tmp:rw,exec,nosuid,size=4g \
            --gpus "device=${GPU_DEVICE}" \
            --shm-size=2g \
            --user "${user_id}:${group_id}" \
            --group-add "${dataset_group_id}" \
            -e HOME=/tmp -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 \
            -e PYTHONPATH=/opt/rx-met/python \
            -e PYTHONDONTWRITEBYTECODE=1 \
            --mount "type=bind,src=$RX_MET_INSTALLED_DIR,dst=/opt/rx-met,readonly" \
            -e USER=rx-met-validator \
            -e LOGNAME=rx-met-validator \
            -e "RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands" \
            -e "RX_MET_KWS_OUTPUT_DIR=/output" \
            -v "${DATASET_DIR}:/datasets/speech_commands:ro" \
            -v "${target_output}:/output" \
            "${image}" \
            python3 - "${rnn_type}" <<'PYTHON' 2>&1 | tee "${target_output}/${rnn_type}/verify.log"
import json
from pathlib import Path
import runpy
import sys

import onnx

rnn_type = sys.argv[1]
entry = "/opt/rx-met/examples/quick_start_kws.py"
sys.argv = [entry, "--rnn_type", rnn_type]
runpy.run_path(entry, run_name="__main__")
output = Path("/output") / rnn_type
model = onnx.load(output / "att_mh_rnn_kws.onnx")
onnx.checker.check_model(model)
nodes = [node for node in model.graph.node if node.op_type in ("GRU", "LSTM")]
assert len(nodes) == 2 and all(node.op_type == rnn_type.upper() for node in nodes)
encodings = json.loads((output / "att_mh_rnn_kws.encodings").read_text())
for node in nodes:
    layer = encodings["activation_encodings"][node.name]
    assert layer["is_" + rnn_type.upper()]
    records = []
    for kind in ("input", "output"):
        assert len(layer[kind]) == 1
        records.extend(layer[kind])
    for kind in ("internal_ops", "internal_ops_reverse"):
        for operation in layer.get(kind, {}).values():
            assert len(operation["output"]) == 1
            records.extend(operation["output"])
    for record in records:
        assert record["enc_type"] == "PER_TENSOR"
        assert all(isinstance(record[field], (int, float))
                   for field in ("scale", "zero_point", "real_min", "real_max"))
    if rnn_type == "lstm":
        assert "quant_lstm_encodings" not in encodings
        assert layer["internal_ops"]["cell_state"]["output"][0]["dtype"] == "INT16"
        assert "model_info" not in layer and "execution_metadata" not in layer
assert (output / "att_mh_rnn_kws_qat.pth").is_file()
print(f"KWS {rnn_type.upper()} 模型、编码及完整示例验证通过")
PYTHON
    done
done

log "全部 KWS example 验证通过: ${OUTPUT_ROOT}"
