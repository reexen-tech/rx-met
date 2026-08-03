#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
DATASET_FILTER="${2:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-models/q64test_qwen25_7b_instruct}"
NAME="${NAME:-Qwen2.5-7B-Instruct}"
BASE="${BASE:-${OUT}/${NAME}-f16.gguf}"
PPL_BIN="${PPL_BIN:-./build_cuda_no_lut/bin/llama-perplexity}"

NGL="${NGL:-99}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:--1}"

LEGACY_TYPES=(Q8_0_64 Q8_1_64 Q5_0_64 Q5_1_64 Q4_0_64 Q4_1_64)
K_TYPES=(Q6_K_64 Q5_K_64 Q5_K_64S Q4_K_64 Q4_K_64S Q3_K_64 Q2_K_64S Q2_K_64)

DATASET_KEYS=(wikitext2 gsm8k math500 mbpp)
declare -A DATASET_FILES=(
  [wikitext2]="models/q64test/wikitext-2-raw/wiki.test.raw"
  [gsm8k]="models/q64test/gsm8k-raw/gsm8k.test.raw"
  [math500]="models/q64test/math500-raw/math500.test.raw"
  [mbpp]="models/q64test/mbpp-raw/mbpp.test.raw"
)

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

should_run_dataset() {
  local dataset="$1"
  [[ "${DATASET_FILTER}" == "all" || "${DATASET_FILTER}" == "${dataset}" ]]
}

run_one_ppl() {
  local dataset="$1"
  local type="$2"
  local model="$3"
  local raw_file="${DATASET_FILES[${dataset}]}"
  local log="${OUT}/ppl_${dataset}_${type}.log"
  local blocked="${OUT}/ppl_${dataset}_${type}.blocked"

  rm -f "${blocked}"

  require_file "${raw_file}"

  if [[ ! -f "${model}" ]]; then
    echo "[BLOCKED] missing model for ${dataset} ${type}: ${model}" | tee "${blocked}"
    return 0
  fi

  if has_final_estimate "${log}"; then
    echo "[SKIP] PPL already complete: ${log}"
    return 0
  fi

  echo "========== PPL ${dataset} ${type} =========="
  if ! "${PPL_BIN}" \
    -m "${model}" \
    -ngl "${NGL}" \
    -f "${raw_file}" \
    -c "${CTX}" \
    --chunks "${CHUNKS}" \
    2>&1 | tee "${log}"; then
    echo "[BLOCKED] PPL failed for ${dataset} ${type}; see ${log}" | tee "${blocked}"
    return 0
  fi
}

run_dataset() {
  local dataset="$1"
  shift

  if ! should_run_dataset "${dataset}"; then
    return 0
  fi

  echo "========== dataset ${dataset} =========="
  run_one_ppl "${dataset}" "base" "${BASE}"

  for type in "$@"; do
    run_one_ppl "${dataset}" "${type}" "${OUT}/${NAME}-${type}.gguf"
  done
}

print_status() {
  echo
  echo "========== PPL status =========="
  local types=(base "${LEGACY_TYPES[@]}" "${K_TYPES[@]}")
  for dataset in "${DATASET_KEYS[@]}"; do
    if ! should_run_dataset "${dataset}"; then
      continue
    fi
    for type in "${types[@]}"; do
      local log="${OUT}/ppl_${dataset}_${type}.log"
      local blocked="${OUT}/ppl_${dataset}_${type}.blocked"
      if has_final_estimate "${log}"; then
        echo "[DONE]    ${dataset} ${type}"
      elif [[ -f "${blocked}" ]]; then
        echo "[BLOCKED] ${dataset} ${type}"
      else
        echo "[PENDING] ${dataset} ${type}"
      fi
    done
  done
}

require_file "${BASE}"
require_file "${PPL_BIN}"

case "${PHASE}" in
  legacy)
    TYPES=("${LEGACY_TYPES[@]}")
    ;;
  k)
    TYPES=("${K_TYPES[@]}")
    ;;
  all)
    TYPES=("${LEGACY_TYPES[@]}" "${K_TYPES[@]}")
    ;;
  base)
    TYPES=()
    ;;
  status)
    print_status
    exit 0
    ;;
  *)
    echo "Usage: $0 {base|legacy|k|all|status} [all|wikitext2|gsm8k|math500|mbpp]" >&2
    exit 2
    ;;
esac

for dataset in "${DATASET_KEYS[@]}"; do
  run_dataset "${dataset}" "${TYPES[@]}"
done

print_status
