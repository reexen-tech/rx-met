#!/usr/bin/env bash
# CUDA REEX 变体 A/B、batch 缩放；可选跑 test-reex-cuda-q16 / run_reex_cuda_trace_sweep
# Q8：VERIFY_GEMM_Q8=1 ./scripts/verify_build_variants.sh 可编出 gemm_q8 / full_q8；存在则纳入 A/B（full_q8 仍为默认 KV f16，KV q8 见 verify_reex_gemm_q8_e2e.sh）
# 依赖：bash（勿用 dash）；镜像可无 /usr/bin/time，本脚本用 bash 内建 time
set -euo pipefail

# ========== 按需修改 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLAMA_ROOT="${LLAMA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export MODEL="${MODEL:-$LLAMA_ROOT/models/Qwen/Qwen3-30B-A3B-Instruct-2507/Qwen3-30B-A3B-Instruct-2507-Q4_K.gguf}"
export WIKI="${WIKI:-$LLAMA_ROOT/datasets/model_evaluation/language_modeling/wikitext/converted/wikitext-2-raw-v1/test.txt}"
REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-full}"
case "$REEX_GATE_PROFILE" in
  project|full) ;;
  *)
    echo "错误: 未知 REEX_GATE_PROFILE=$REEX_GATE_PROFILE（仅支持 project/full）" >&2
    exit 2
    ;;
esac
VERIFY_THREADS="${VERIFY_THREADS:-4}"
VERIFY_PPL_CTX="${VERIFY_PPL_CTX:-512}"
VERIFY_PPL_BATCH="${VERIFY_PPL_BATCH:-512}"
VERIFY_PPL_CHUNKS="${VERIFY_PPL_CHUNKS:-5}"
VERIFY_NGL_CUDA="${VERIFY_NGL_CUDA:-99}"
VERIFY_CUDA_AB_VARIANTS="${VERIFY_CUDA_AB_VARIANTS:-cuda_only cuda_reex_fp16 cuda_reex_full cuda_reex_gemm}"
VERIFY_CUDA_AB_BATCHES="${VERIFY_CUDA_AB_BATCHES:-128 8}"
VERIFY_CUDA_Q16_TEST="${VERIFY_CUDA_Q16_TEST:-1}"
VERIFY_CUDA_TRACE_SWEEP="${VERIFY_CUDA_TRACE_SWEEP:-1}"
# ==============================

cd "$LLAMA_ROOT"

if [[ ! -f "$MODEL" ]] || [[ ! -f "$WIKI" ]]; then
  echo "错误: MODEL 或 WIKI 不存在"
  echo "  MODEL=$MODEL"
  echo "  WIKI=$WIKI"
  exit 2
fi

echo "=== verify_build_cuda_reex.sh ==="
echo "  REEX_GATE_PROFILE=$REEX_GATE_PROFILE"
echo "  MODEL=$MODEL"
echo "  WIKI=$WIKI"
echo "  ctx=$VERIFY_PPL_CTX batch=$VERIFY_PPL_BATCH chunks=$VERIFY_PPL_CHUNKS ngl=$VERIFY_NGL_CUDA threads=$VERIFY_THREADS"
echo "  VERIFY_CUDA_AB_VARIANTS=$VERIFY_CUDA_AB_VARIANTS"
echo "  VERIFY_CUDA_AB_BATCHES=$VERIFY_CUDA_AB_BATCHES"
echo "  VERIFY_CUDA_Q16_TEST=$VERIFY_CUDA_Q16_TEST  VERIFY_CUDA_TRACE_SWEEP=$VERIFY_CUDA_TRACE_SWEEP"
echo ""

export CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
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

# 用 bash 内建 time（勿写 env ... time，否则无 /usr/bin/time 时会报 No such file）
run_ppl() {
  local root="$1"
  shift
  local exe="$root/bin/llama-perplexity"
  export LD_LIBRARY_PATH="$(lp_ld "$root")"
  TIMEFORMAT=$'  wall %R s (user %U sys %S)\n'
  time "$exe" "$@"
}

run_ppl_grep() {
  local root="$1"
  shift
  local exe="$root/bin/llama-perplexity"
  export LD_LIBRARY_PATH="$(lp_ld "$root")"
  # 兼容不同版本日志关键词
  "$exe" "$@" 2>&1 | grep -E 'second.*per pass|prompt eval time|tokens per second|tok/s|Error|error|fatal' || true
}

is_fit_or_oom_failure() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  grep -qE \
    'failed to fit params to free device memory|n_gpu_layers already set by user to .* abort|cudaMalloc failed: out of memory|unable to allocate CUDA0 buffer' \
    "$log"
}

append_unique() {
  local value="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$value" ]] && return 0
  done
  printf '%s\n' "$value"
}

candidate_ngls() {
  local primary="${1:-$VERIFY_NGL_CUDA}"
  local -a values=("$primary" 80 64 48 40 32 24 16 8 0)
  local -a out=()
  local v
  for v in "${values[@]}"; do
    [[ "$v" =~ ^[0-9]+$ ]] || continue
    if ((${#out[@]} == 0)); then
      out+=("$v")
      continue
    fi
    if ! printf '%s\n' "${out[@]}" | grep -qx "$v"; then
      out+=("$v")
    fi
  done
  printf '%s\n' "${out[@]}"
}

batch_for_variant() {
  local label="$1"
  local batch="$VERIFY_PPL_BATCH"
  case "$label" in
    cuda_reex_gemm|cuda_reex_full)
      if (( batch > 8 )); then
        batch=8
      fi
      ;;
  esac
  echo "$batch"
}

run_variant_ppl_resilient() {
  local root="$1"
  local label="$2"
  local log="$3"
  shift 3
  local ngl
  local rc=1
  mapfile -t _ngl_candidates < <(candidate_ngls "$VERIFY_NGL_CUDA")
  local attempt=1
  for ngl in "${_ngl_candidates[@]}"; do
    echo "  [$label] 尝试 ngl=$ngl (第 $attempt/${#_ngl_candidates[@]} 次)"
    set +e
    run_ppl "$root" "$@" -ngl "$ngl" -t "$VERIFY_THREADS" 2>&1 | tee "$log" | tail -n 30
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" -eq 0 ]]; then
      echo "  [$label] 成功，ngl=$ngl"
      return 0
    fi
    echo "  [$label] 失败，exit_code=$rc"
    if ! is_fit_or_oom_failure "$log"; then
      return "$rc"
    fi
    sleep 3
    attempt=$((attempt + 1))
  done
  return "$rc"
}

run_variant_ppl_grep_resilient() {
  local root="$1"
  local label="$2"
  local log="$3"
  shift 3
  local exe="$root/bin/llama-perplexity"
  local ldpath
  ldpath="$(lp_ld "$root")"
  local ngl
  local rc=1
  mapfile -t _ngl_candidates < <(candidate_ngls "$VERIFY_NGL_CUDA")
  local attempt=1
  for ngl in "${_ngl_candidates[@]}"; do
    echo "  [$label] 尝试 ngl=$ngl (第 $attempt/${#_ngl_candidates[@]} 次)"
    set +e
    env LD_LIBRARY_PATH="$ldpath" "$exe" "$@" -ngl "$ngl" -t "$VERIFY_THREADS" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e
    grep -E 'second.*per pass|prompt eval time|tokens per second|tok/s|Error|error|fatal' "$log" || true
    if [[ "$rc" -eq 0 ]]; then
      echo "  [$label] 成功，ngl=$ngl"
      return 0
    fi
    echo "  [$label] 失败，exit_code=$rc"
    if ! is_fit_or_oom_failure "$log"; then
      return "$rc"
    fi
    sleep 3
    attempt=$((attempt + 1))
  done
  return "$rc"
}

#----------------------------------------------------------------
# 1) A/B：对比变体（务必看 /tmp/ppl_*.log 全文与下面 time 行）
#----------------------------------------------------------------
read -r -a VARIANTS <<< "$VERIFY_CUDA_AB_VARIANTS"

for v in "${VARIANTS[@]}"; do
  root="$LLAMA_ROOT/build_verify_${v}"
  [[ -x "$root/bin/llama-perplexity" ]] || { echo "跳过: 无 $root/bin/llama-perplexity"; continue; }
  batch="$(batch_for_variant "$v")"
  start_epoch="$(date +%s)"
  echo "========== $v =========="
  run_variant_ppl_resilient "$root" "$v" "/tmp/ppl_${v}.log" \
    -m "$MODEL" -f "$WIKI" --ctx-size "$VERIFY_PPL_CTX" -b "$batch" --chunks "$VERIFY_PPL_CHUNKS"
  echo "  [$v] 耗时: $(format_duration $(( $(date +%s) - start_epoch )))"
  echo ""
done

if [[ "$REEX_GATE_PROFILE" == "full" ]]; then
  #----------------------------------------------------------------
  # 2) batch 缩放：cuda_reex_full
  #----------------------------------------------------------------
  root="$LLAMA_ROOT/build_verify_cuda_reex_full"
  if [[ -x "$root/bin/llama-perplexity" ]]; then
    for b in $VERIFY_CUDA_AB_BATCHES; do
      start_epoch="$(date +%s)"
      echo "========== cuda_reex_full batch=$b =========="
      run_variant_ppl_grep_resilient "$root" "cuda_reex_full batch=$b" "/tmp/ppl_cuda_reex_full_batch_${b}.log" \
        -m "$MODEL" -f "$WIKI" --ctx-size "$VERIFY_PPL_CTX" -b "$b" --chunks "$VERIFY_PPL_CHUNKS"
      echo "  [cuda_reex_full batch=$b] 耗时: $(format_duration $(( $(date +%s) - start_epoch )))"
      echo ""
    done
  else
    echo "跳过 batch 缩放: 无 $root/bin/llama-perplexity"
  fi

  #----------------------------------------------------------------
  # 3) test-reex-cuda-q16（需已编译该 target）
  #----------------------------------------------------------------
  T16BIN="$LLAMA_ROOT/build_verify_cuda_reex_full/bin/test-reex-cuda-q16"
  [[ -x "$T16BIN" ]] || T16BIN="$LLAMA_ROOT/build_verify_cuda_reex_gemm/bin/test-reex-cuda-q16"
  if [[ "$VERIFY_CUDA_Q16_TEST" != "1" ]]; then
    echo "[full] 已跳过 test-reex-cuda-q16（VERIFY_CUDA_Q16_TEST=0）"
  elif [[ -x "$T16BIN" ]]; then
    start_epoch="$(date +%s)"
    export LD_LIBRARY_PATH="$(lp_ld "$(dirname "$T16BIN")/..")"
    "$T16BIN" 2>&1 | tee /tmp/test-reex-cuda-q16.log
    echo "  [test-reex-cuda-q16] 耗时: $(format_duration $(( $(date +%s) - start_epoch )))"
  else
    echo "未找到 test-reex-cuda-q16，可在 REEX CUDA 构建目录执行:"
    echo "  cmake --build build_verify_cuda_reex_full --target test-reex-cuda-q16 -j"
  fi

  #----------------------------------------------------------------
  # 4) run_reex_cuda_trace_sweep（需 eval_single_token）
  #    子脚本默认 BUILD_DIR=build，必须显式 export EVAL_BIN / BUILD_DIR
  #----------------------------------------------------------------
  find_eval_single_token() {
    local d p
    for d in \
      "$LLAMA_ROOT/build_verify_cuda_reex_full" \
      "$LLAMA_ROOT/build_verify_cuda_reex_gemm" \
      "$LLAMA_ROOT/build_verify_cuda_reex_gemm_q8" \
      "$LLAMA_ROOT/build_verify_cuda_reex_full_q8" \
      "$LLAMA_ROOT/build"; do
      for p in "$d/bin/eval_single_token" "$d/tests/eval_single_token"; do
        if [[ -x "$p" ]]; then
          echo "$p"
          return 0
        fi
      done
    done
    return 1
  }

  EVAL="$(find_eval_single_token || true)"
  if [[ "$VERIFY_CUDA_TRACE_SWEEP" != "1" ]]; then
    echo "[full] 已跳过 run_reex_cuda_trace_sweep（VERIFY_CUDA_TRACE_SWEEP=0）"
  elif [[ -n "${EVAL:-}" ]]; then
    start_epoch="$(date +%s)"
    export EVAL_BIN="$EVAL"
    # eval 常在 <build>/bin/eval_single_token，BUILD_DIR 应为该 build 根目录
    export BUILD_DIR="$(cd "$(dirname "$EVAL")/.." && pwd)"
    export MODEL="$MODEL"
    bash "$LLAMA_ROOT/tests/test-reex/run_reex_cuda_trace_sweep.sh"
    echo "  [run_reex_cuda_trace_sweep] 耗时: $(format_duration $(( $(date +%s) - start_epoch )))"
  else
    echo "未找到 eval_single_token，可在 REEX CUDA 构建目录执行:"
    echo "  cmake --build build_verify_cuda_reex_full --target eval_single_token -j"
    echo "  cmake --build build_verify_cuda_reex_full --target test-reex-cuda-q16 -j   # 算子级 Q16 命中统计"
    exit 0
  fi
else
  echo "[project] 已跳过 batch 缩放、test-reex-cuda-q16 与 trace sweep；这些属于 full-only 深度专项。"
fi
