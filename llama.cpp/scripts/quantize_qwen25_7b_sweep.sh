#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-models/q64test_qwen25_7b_instruct}"
NAME="${NAME:-Qwen2.5-7B-Instruct}"
BASE="${BASE:-${OUT}/${NAME}-f16.gguf}"
QUANT_BIN="${QUANT_BIN:-./build_cuda_no_lut/bin/llama-quantize}"

LEGACY_TYPES=(Q8_0_64 Q8_1_64 Q5_0_64 Q5_1_64 Q4_0_64 Q4_1_64)
K_TYPES=(Q6_K_64 Q5_K_64 Q5_K_64S Q4_K_64 Q4_K_64S Q3_K_64 Q2_K_64S Q2_K_64)

mkdir -p "${OUT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] missing: ${path}" >&2
    exit 1
  fi
}

run_one_type() {
  local type="$1"
  local model="${OUT}/${NAME}-${type}.gguf"
  local quant_log="${OUT}/quant_${type}.log"
  local blocked="${OUT}/${type}.quant.blocked"

  rm -f "${blocked}"

  if [[ -f "${model}" ]]; then
    echo "[SKIP] quantized model exists: ${model}"
    return 0
  fi

  echo "========== quantize ${type} =========="
  if ! "${QUANT_BIN}" "${BASE}" "${model}" "${type}" 2>&1 | tee "${quant_log}"; then
    echo "[BLOCKED] quantize failed for ${type}; see ${quant_log}" | tee "${blocked}"
    return 0
  fi
}

run_group() {
  local group_name="$1"
  shift

  echo "========== ${group_name} =========="
  for type in "$@"; do
    run_one_type "${type}"
  done
}

print_status() {
  echo
  echo "========== quantization status =========="
  for type in "${LEGACY_TYPES[@]}" "${K_TYPES[@]}"; do
    local model="${OUT}/${NAME}-${type}.gguf"
    local blocked="${OUT}/${type}.quant.blocked"
    if [[ -f "${model}" ]]; then
      echo "[DONE]    ${type}"
    elif [[ -f "${blocked}" ]]; then
      echo "[BLOCKED] ${type}"
    else
      echo "[PENDING] ${type}"
    fi
  done
}

require_file "${BASE}"
require_file "${QUANT_BIN}"

case "${PHASE}" in
  legacy)
    run_group "legacy" "${LEGACY_TYPES[@]}"
    ;;
  k)
    run_group "k-quant" "${K_TYPES[@]}"
    ;;
  all)
    run_group "legacy" "${LEGACY_TYPES[@]}"
    run_group "k-quant" "${K_TYPES[@]}"
    ;;
  status)
    ;;
  *)
    echo "Usage: $0 {legacy|k|all|status}" >&2
    exit 2
    ;;
esac

print_status
