#!/usr/bin/env bash
# 单独补跑 Qwen3.5-35B-A3B 的 MBPP PPL（block64 / q64test_qwen35 目录）。
#
# 背景：该目录此前只跑了 wikitext2 / gsm8k / math500 三个数据集，MBPP 缺失。
# 本脚本只补 MBPP，日志命名为 ppl_mbpp_<type>.log（与已有 ppl_gsm8k_* / ppl_math500_* 前缀风格一致）。
#
# 量化类型清单与既有 qwen35 sweep 一致：14 个 Q*_64 + 5 个 IQ*。
# 权重文件已存在（19 个 .gguf 已生成），本脚本只跑 PPL。
#
# 幂等：已含 "Final estimate" 的日志自动跳过，可重复运行。
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${OUT:-models/q64test_qwen35}"
NAME="${NAME:-Qwen3.5-35B-A3B}"
BASE="${BASE:-${OUT}/${NAME}-f16.gguf}"
F="${F:-models/q64test/mbpp-raw/mbpp.test.raw}"
PPL_BIN="${PPL_BIN:-./build_cuda_no_lut/bin/llama-perplexity}"

NGL="${NGL:-99}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:--1}"

# 14 个 Q*_64 + 5 个 IQ*，与已有 sweep 完全一致。
Q_TYPES=(Q8_0_64 Q8_1_64 Q6_K_64 Q5_1_64 Q5_K_64 Q5_0_64 Q5_K_64S Q4_1_64 Q4_K_64 Q4_0_64 Q4_K_64S Q3_K_64 Q2_K_64 Q2_K_64S)
IQ_TYPES=(IQ3_XXS IQ2_XXS IQ2_S IQ1_M IQ1_S)

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

run_one_ppl() {
  local label="$1"      # 日志里的标签，如 base / Q4_K_64 / IQ2_S
  local model="$2"
  local log="${OUT}/ppl_mbpp_${label}.log"
  local blocked="${OUT}/ppl_mbpp_${label}.blocked"

  rm -f "${blocked}"

  if [[ ! -f "${model}" ]]; then
    echo "[BLOCKED] missing model for mbpp ${label}: ${model}" | tee "${blocked}"
    return 0
  fi

  if has_final_estimate "${log}"; then
    echo "[SKIP] PPL already complete: mbpp ${label}"
    return 0
  fi

  echo "========== PPL mbpp ${label} =========="
  if (( DRY_RUN )); then
    echo "[DRY-RUN] ${PPL_BIN} -m ${model} -ngl ${NGL} -f ${F} -c ${CTX} --chunks ${CHUNKS}"
    return 0
  fi
  if ! "${PPL_BIN}" \
    -m "${model}" \
    -ngl "${NGL}" \
    -f "${F}" \
    -c "${CTX}" \
    --chunks "${CHUNKS}" \
    2>&1 | tee "${log}"; then
    echo "[BLOCKED] PPL failed for mbpp ${label}; see ${log}" | tee "${blocked}"
    return 0
  fi
}

print_status() {
  echo
  echo "========== MBPP PPL status =========="
  local types=("base" "${Q_TYPES[@]}" "${IQ_TYPES[@]}")
  local ok=0 bad=0
  for t in "${types[@]}"; do
    local log="${OUT}/ppl_mbpp_${t}.log"
    local blocked="${OUT}/ppl_mbpp_${t}.blocked"
    if has_final_estimate "${log}"; then
      echo "[DONE]    ${t}"; ok=$((ok+1))
    elif [[ -f "${blocked}" ]]; then
      echo "[BLOCKED] ${t}"; bad=$((bad+1))
    else
      echo "[PENDING] ${t}"; bad=$((bad+1))
    fi
  done
  echo "---- summary: ${ok} done / ${bad} not-done (of ${#types[@]}) ----"
}

require_file "${BASE}"
require_file "${F}"
require_file "${PPL_BIN}"

# base (FP16)
run_one_ppl "base" "${BASE}"

# 14 个 Q*_64
for t in "${Q_TYPES[@]}"; do
  run_one_ppl "${t}" "${OUT}/${NAME}-${t}.gguf"
done

# 5 个 IQ*
for t in "${IQ_TYPES[@]}"; do
  run_one_ppl "${t}" "${OUT}/${NAME}-${t}.gguf"
done

print_status
