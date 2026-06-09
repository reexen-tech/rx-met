#!/usr/bin/env bash
# 端到端 Wikitext PPL + prompt tok/s：
#   1) cuda_only
#   2) cuda_reex_gemm_q8     — GEMM Q8 + USE_REEX（PWNL/LUT），KV 默认 f16
#   3) cuda_reex_full_q8     — 再加 REEX FP16 管线；下列为同一构建上的 KV 运行时组合：
#        - KV 全 q8_0
#        - K=q8_0, V=q4_0（K8 + V4）
#        - K/V=q4_0（KV Q4）
#   量化 V 会走 Flash Attention。K8V4（-ctk q8_0 -ctv q4_0）本脚本固定追加 --flash-attn on：
#   默认 flash-attn=auto 时若 FA 张量落到 CPU 会与「量化 V 必须 FA」冲突并可能 139。
#   VERIFY_Q8_E2E_KV_EXTRA 仍可追加其它 flag（--flash-attn on 放在该行最后，会覆盖 EXTRA 里的 -fa）。
#
#   【K8+V4】CMake 未开 GGML_CUDA_FA_ALL_QUANTS 时，CUDA FA 在「K/V 类型不同」下会选不到内核（见 fattn.cu），
#   调度器会把 48 层 FLASH_ATTN 全部回退到 CPU；此时即使 --flash-attn on 不崩，也会出现极低 tok/s。
#   解决：VERIFY_GEMM_Q8=1 ./scripts/verify_build_variants.sh 已为 gemm_q8/full_q8 打开 FA_ALL_QUANTS；
#   或 cmake -DGGML_CUDA_FA_ALL_QUANTS=ON 后重编。
#
# 前置：VERIFY_GEMM_Q8=1 ./scripts/verify_build_variants.sh（gemm_q8 + full_q8）
#
# 用法:
#   ./scripts/verify_reex_gemm_q8_e2e.sh
#   VERIFY_Q8_E2E_CHUNKS=500 ./scripts/verify_reex_gemm_q8_e2e.sh
#   VERIFY_Q8_E2E_KV_EXTRA="-fa" ./scripts/verify_reex_gemm_q8_e2e.sh
#
set -euo pipefail

export LLAMA_ROOT="${LLAMA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MODEL="${MODEL:-$LLAMA_ROOT/models/Qwen/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf}"
export WIKI="${WIKI:-$LLAMA_ROOT/datasets/model_evaluation/language_modeling/wikitext/converted/wikitext-2-raw-v1/test.txt}"
REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-full}"

VERIFY_CTX="${VERIFY_CTX:-512}"
VERIFY_BATCH="${VERIFY_BATCH:-512}"
VERIFY_CHUNKS="${VERIFY_Q8_E2E_CHUNKS:-${VERIFY_CHUNKS:-3}}"
VERIFY_NGL="${VERIFY_NGL_CUDA:-99}"
VERIFY_THREADS="${VERIFY_THREADS:-4}"
VERIFY_Q8_E2E_KV_EXTRA="${VERIFY_Q8_E2E_KV_EXTRA:-}"
VERIFY_Q8_E2E_VARIANTS="${VERIFY_Q8_E2E_VARIANTS:-cuda_only cuda_reex_gemm_q8 cuda_reex_full_q8+K8V8 cuda_reex_full_q8+K8V4}"
VERIFY_Q8_E2E_RUN_Q8_TEST="${VERIFY_Q8_E2E_RUN_Q8_TEST:-1}"
# 未重编带 FA_ALL_QUANTS 时 K8V4 会崩，可设 1 跳过该行
VERIFY_Q8_E2E_SKIP_K8V4="${VERIFY_Q8_E2E_SKIP_K8V4:-0}"

CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
cd "$LLAMA_ROOT"

lp_ld() { echo "$1/bin:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"; }

format_duration() {
  local total_sec="${1:-0}"
  local h=$(( total_sec / 3600 ))
  local m=$(( (total_sec % 3600) / 60 ))
  local s=$(( total_sec % 60 ))
  if (( h > 0 )); then
    printf '%dh%02dm%02ds' "$h" "$m" "$s"
  elif (( m > 0 )); then
    printf '%dm%02ds' "$m" "$s"
  else
    printf '%ds' "$s"
  fi
}

cache_bool_is() {
  local cache="$1"
  local key="$2"
  local expect="$3"
  [[ -f "$cache" ]] || return 1
  grep -q "^${key}:BOOL=${expect}$" "$cache" 2>/dev/null
}

# 只取「prompt eval」行的 tok/s。llama_perf_context_print 还会打一行「eval time」（decode）；
# perplexity 末尾常 n_eval=1：若该行耗时非零会得到 ~100 tok/s，误用「最后一个 tokens per second)」
# 会把 K8V4 等配置错算成两个数量级偏慢（其它变体若 eval 为 0 打 inf 则不会中招）。
extract_tok_s() {
  python3 - "$1" <<'PY'
import re, sys
try:
    text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
except OSError:
    print("")
    sys.exit(0)
val = ""
for line in text.splitlines():
    if "prompt eval time" not in line:
        continue
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second\)", line)
    if m:
        val = m.group(1)
print(val)
PY
}

extract_ppl() {
  python3 - "$1" <<'PY'
import re, sys
try:
    text = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
except OSError:
    print("")
    sys.exit(0)
m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", text)
print(m.group(1) if m else "")
PY
}

run_ppl_to_log() {
  local root="$1"
  local log="$2"
  shift 2
  local exe="$root/bin/llama-perplexity"
  [[ -x "$exe" ]] || { echo "SKIP_NO_BIN"; return 1; }
  export LD_LIBRARY_PATH="$(lp_ld "$root")"
  "$exe" -m "$MODEL" -f "$WIKI" \
    --ctx-size "$VERIFY_CTX" -b "$VERIFY_BATCH" \
    --chunks "$VERIFY_CHUNKS" \
    -ngl "$VERIFY_NGL" -t "$VERIFY_THREADS" \
    "$@" \
    2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

ONLY_ROOT="$LLAMA_ROOT/build_verify_cuda_only"
Q8_ROOT="$LLAMA_ROOT/build_verify_cuda_reex_gemm_q8"
FULL_Q8_ROOT="$LLAMA_ROOT/build_verify_cuda_reex_full_q8"

if [[ ! -x "$ONLY_ROOT/bin/llama-perplexity" ]]; then
  echo "错误: 缺少 $ONLY_ROOT/bin/llama-perplexity（cuda_only）"
  exit 2
fi
if [[ ! -x "$Q8_ROOT/bin/llama-perplexity" ]]; then
  echo "错误: 缺少 $Q8_ROOT/bin/llama-perplexity（请先 VERIFY_GEMM_Q8=1 ./scripts/verify_build_variants.sh）"
  exit 2
fi
if [[ ! -x "$FULL_Q8_ROOT/bin/llama-perplexity" ]]; then
  echo "错误: 缺少 $FULL_Q8_ROOT/bin/llama-perplexity（请先 VERIFY_GEMM_Q8=1 ./scripts/verify_build_variants.sh）"
  exit 2
fi
if [[ ! -f "$MODEL" ]] || [[ ! -f "$WIKI" ]]; then
  echo "错误: MODEL 或 WIKI 不存在"
  echo "  MODEL=$MODEL"
  echo "  WIKI=$WIKI"
  exit 2
fi

echo "================================================================"
echo "  cuda_only | gemm_q8 | full_q8+K8V8 | full_q8+K8V4 | full_q8+KVq4"
echo "================================================================"
echo "  MODEL=$MODEL"
echo "  WIKI=$WIKI"
echo "  ctx=$VERIFY_CTX batch=$VERIFY_BATCH chunks=$VERIFY_CHUNKS ngl=$VERIFY_NGL threads=$VERIFY_THREADS"
echo "  REEX_GATE_PROFILE=$REEX_GATE_PROFILE"
echo "  VERIFY_Q8_E2E_VARIANTS=$VERIFY_Q8_E2E_VARIANTS"
echo "  VERIFY_Q8_E2E_RUN_Q8_TEST=$VERIFY_Q8_E2E_RUN_Q8_TEST"
echo "  full_q8 KV 行追加: ${VERIFY_Q8_E2E_KV_EXTRA:-<无>}（K8V4 另固定 --flash-attn on）"
if cache_bool_is "$FULL_Q8_ROOT/CMakeCache.txt" GGML_CUDA_FA_ALL_QUANTS ON; then
  echo "  full_q8 CMakeCache: GGML_CUDA_FA_ALL_QUANTS=ON"
else
  echo "  full_q8 CMakeCache: GGML_CUDA_FA_ALL_QUANTS!=ON（K8V4 将跳过，避免误跑到 CPU Flash Attention）"
fi
echo ""

declare -a ROW_TAGS=() ROW_TOK=() ROW_PPL=() ROW_RC=()

bench_one() {
  local v="$1"
  shift
  local root="$1"
  shift
  local log
  local start_epoch
  log="$(mktemp)"
  start_epoch="$(date +%s)"
  echo "---------- ${v} ----------"
  local rc=0
  run_ppl_to_log "$root" "$log" "$@" || rc=$?
  local t p
  t="$(extract_tok_s "$log")"
  p="$(extract_ppl "$log")"
  ROW_TAGS+=("$v"); ROW_TOK+=("${t:-N/A}"); ROW_PPL+=("${p:-N/A}"); ROW_RC+=("$rc")
  if [[ "$rc" -ne 0 ]]; then
    echo "  [失败] llama-perplexity 退出码 $rc"
  else
    echo "  [OK] prompt tok/s ≈ ${t:-?}  PPL ≈ ${p:-?}  耗时: $(format_duration $(( $(date +%s) - start_epoch )))"
  fi
  rm -f "$log"
  echo ""
}

for variant in $VERIFY_Q8_E2E_VARIANTS; do
  case "$variant" in
    cuda_only)
      bench_one "cuda_only" "$ONLY_ROOT"
      ;;
    cuda_reex_gemm_q8)
      bench_one "cuda_reex_gemm_q8" "$Q8_ROOT"
      ;;
    cuda_reex_full_q8+K8V8)
      # shellcheck disable=SC2086
      bench_one "cuda_reex_full_q8+K8V8" "$FULL_Q8_ROOT" -ctk q8_0 -ctv q8_0 ${VERIFY_Q8_E2E_KV_EXTRA}
      ;;
    cuda_reex_full_q8+K8V4)
      if [[ "${VERIFY_Q8_E2E_SKIP_K8V4}" == "1" ]]; then
        echo "---------- cuda_reex_full_q8+K8V4 ----------"
        echo "  [跳过] VERIFY_Q8_E2E_SKIP_K8V4=1（或重编: -DGGML_CUDA_FA_ALL_QUANTS=ON）"
        ROW_TAGS+=("cuda_reex_full_q8+K8V4"); ROW_TOK+=("(skip)"); ROW_PPL+=("(skip)"); ROW_RC+=("skip")
        echo ""
      elif ! cache_bool_is "$FULL_Q8_ROOT/CMakeCache.txt" GGML_CUDA_FA_ALL_QUANTS ON; then
        echo "---------- cuda_reex_full_q8+K8V4 ----------"
        echo "  [跳过] build_verify_cuda_reex_full_q8 未启用 GGML_CUDA_FA_ALL_QUANTS=ON"
        echo "         否则 K!=V 的 CUDA Flash Attention 会回退 CPU，结果会严重失真"
        ROW_TAGS+=("cuda_reex_full_q8+K8V4"); ROW_TOK+=("(skip:no_fa_all_quants)"); ROW_PPL+=("(skip)"); ROW_RC+=("skip")
        echo ""
      else
        # shellcheck disable=SC2086
        bench_one "cuda_reex_full_q8+K8V4" "$FULL_Q8_ROOT" -ctk q8_0 -ctv q4_0 ${VERIFY_Q8_E2E_KV_EXTRA} --flash-attn on
      fi
      ;;
    cuda_reex_full_q8+KVq4)
      # shellcheck disable=SC2086
      bench_one "cuda_reex_full_q8+KVq4" "$FULL_Q8_ROOT" -ctk q4_0 -ctv q4_0 ${VERIFY_Q8_E2E_KV_EXTRA}
      ;;
    *)
      echo "错误: 未知 VERIFY_Q8_E2E_VARIANTS 项: $variant"
      exit 2
      ;;
  esac
done

echo "================================================================"
echo "  汇总"
echo "================================================================"
printf "  %-34s  %14s  %12s  %s\n" "variant" "tok/s" "PPL" "rc"
for i in "${!ROW_TAGS[@]}"; do
  printf "  %-34s  %14s  %12s  %s\n" "${ROW_TAGS[$i]}" "${ROW_TOK[$i]}" "${ROW_PPL[$i]}" "${ROW_RC[$i]}"
done

base="${ROW_TOK[0]:-}"
if [[ "$base" =~ ^[0-9.]+$ ]]; then
  echo ""
  echo "  相对 cuda_only（tok/s）："
  for i in $(seq 1 $((${#ROW_TOK[@]} - 1))); do
    t="${ROW_TOK[$i]}"
    tag="${ROW_TAGS[$i]}"
    [[ "$t" =~ ^[0-9.]+$ ]] || continue
    awk -v b="$base" -v x="$t" -v lbl="$tag" 'BEGIN { printf "    %-34s  %6.1f%%\n", lbl, 100.0*x/b }'
  done
fi

T8="$Q8_ROOT/bin/test-reex-cuda-q8"
if [[ "$VERIFY_Q8_E2E_RUN_Q8_TEST" != "1" ]]; then
  echo ""
  echo "[提示] 已跳过 test-reex-cuda-q8（VERIFY_Q8_E2E_RUN_Q8_TEST=0）"
elif [[ -x "$T8" ]]; then
  echo ""
  echo "================================================================"
  echo "  test-reex-cuda-q8（gemm_q8 构建）"
  echo "================================================================"
  start_epoch="$(date +%s)"
  export LD_LIBRARY_PATH="$(lp_ld "$Q8_ROOT")"
  set +o pipefail
  "$T8" 2>&1 | tee /tmp/test-reex-cuda-q8_e2e.log
  t8_rc="${PIPESTATUS[0]}"
  set -o pipefail
  if [[ "$t8_rc" -ne 0 ]]; then
    echo "错误: test-reex-cuda-q8 退出码 $t8_rc"
    exit 1
  fi
  echo "  [test-reex-cuda-q8] 耗时: $(format_duration $(( $(date +%s) - start_epoch )))"
else
  echo ""
  echo "[提示] 未找到 test-reex-cuda-q8: cmake --build build_verify_cuda_reex_gemm_q8 --target test-reex-cuda-q8 -j"
fi

for i in "${!ROW_RC[@]}"; do
  r="${ROW_RC[$i]}"
  [[ "$r" == "skip" ]] && continue
  [[ "$r" == "0" ]] || exit 1
done
exit 0
