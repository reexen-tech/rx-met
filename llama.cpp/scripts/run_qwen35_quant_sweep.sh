#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-models/q64test_qwen35}"
NAME="${NAME:-Qwen3.5-35B-A3B}"
BASE="${BASE:-${OUT}/${NAME}-f16.gguf}"
F="${F:-models/q64test/wikitext-2-raw/wiki.test.raw}"

QUANT_BIN="${QUANT_BIN:-./build_cuda_no_lut/bin/llama-quantize}"
PPL_BIN="${PPL_BIN:-./build_cuda_no_lut/bin/llama-perplexity}"

NGL="${NGL:-99}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:--1}"

LEGACY_TYPES=(Q8_0_64 Q5_0_64 Q4_0_64 Q3_0_64 Q2_0_64)
K_TYPES=(Q6_K_64 Q5_K_64 Q4_K_64 Q3_K_64 Q2_K_64)
IQ_TYPES=(IQ3_XXS IQ2_S IQ2_XXS IQ1_M IQ1_S)

mkdir -p "${OUT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] missing: ${path}" >&2
    exit 1
  fi
}

has_final_estimate() {
  local log="$1"
  [[ -f "${log}" ]] && grep -q "Final estimate" "${log}"
}

run_base_ppl() {
  local log="${OUT}/ppl_base.log"

  if has_final_estimate "${log}"; then
    echo "[SKIP] baseline PPL already complete: ${log}"
    return 0
  fi

  echo "[RUN] baseline PPL"
  "${PPL_BIN}" \
    -m "${BASE}" \
    -ngl "${NGL}" \
    -f "${F}" \
    -c "${CTX}" \
    --chunks "${CHUNKS}" \
    2>&1 | tee "${log}"
}

run_one_type() {
  local type="$1"
  local model="${OUT}/${NAME}-${type}.gguf"
  local quant_log="${OUT}/quant_${type}.log"
  local ppl_log="${OUT}/ppl_${type}.log"
  local blocked="${OUT}/${type}.blocked"

  rm -f "${blocked}"

  if [[ -f "${model}" ]]; then
    echo "[SKIP] quantized model exists: ${model}"
  else
    echo "[RUN] quantize ${type}"
    if ! "${QUANT_BIN}" "${BASE}" "${model}" "${type}" 2>&1 | tee "${quant_log}"; then
      echo "[BLOCKED] quantize failed for ${type}; see ${quant_log}" | tee "${blocked}"
      return 0
    fi
  fi

  if has_final_estimate "${ppl_log}"; then
    echo "[SKIP] PPL already complete: ${ppl_log}"
    return 0
  fi

  echo "[RUN] PPL ${type}"
  if ! "${PPL_BIN}" \
    -m "${model}" \
    -ngl "${NGL}" \
    -f "${F}" \
    -c "${CTX}" \
    --chunks "${CHUNKS}" \
    2>&1 | tee "${ppl_log}"; then
    echo "[BLOCKED] PPL failed for ${type}; see ${ppl_log}" | tee "${blocked}"
    return 0
  fi
}

run_group() {
  local group_name="$1"
  shift

  echo "========== ${group_name} =========="
  run_base_ppl

  for type in "$@"; do
    run_one_type "${type}"
  done
}

print_status() {
  echo
  echo "========== status =========="
  for type in "${LEGACY_TYPES[@]}" "${K_TYPES[@]}" "${IQ_TYPES[@]}"; do
    local ppl_log="${OUT}/ppl_${type}.log"
    local blocked="${OUT}/${type}.blocked"
    if has_final_estimate "${ppl_log}"; then
      echo "[DONE]    ${type}"
    elif [[ -f "${blocked}" ]]; then
      echo "[BLOCKED] ${type}"
    else
      echo "[PENDING] ${type}"
    fi
  done
}

require_file "${BASE}"
require_file "${F}"
require_file "${QUANT_BIN}"
require_file "${PPL_BIN}"

case "${PHASE}" in
  legacy0)
    run_group "legacy0" "${LEGACY_TYPES[@]}"
    ;;
  k)
    run_group "k-quant" "${K_TYPES[@]}"
    ;;
  iq)
    run_group "iq" "${IQ_TYPES[@]}"
    ;;
  all)
    run_group "legacy0" "${LEGACY_TYPES[@]}"
    run_group "k-quant" "${K_TYPES[@]}"
    run_group "iq" "${IQ_TYPES[@]}"
    ;;
  status)
    ;;
  *)
    echo "Usage: $0 {legacy0|k|iq|all|status}" >&2
    exit 2
    ;;
esac

print_status
