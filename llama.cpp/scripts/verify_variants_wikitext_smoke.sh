#!/usr/bin/env bash
# 对已存在的 build_verify_* 目录中的 llama-perplexity 做小批量 Wikitext PPL 冒烟
#（不重新编译）。通常先跑 ./scripts/verify_build_variants.sh，再跑本脚本。
#
# 用法（在 llama.cpp 根目录）:
#   export VERIFY_MODEL=/path/to/model.gguf
#   export VERIFY_WIKITEXT=/path/to/validation.txt   # 或 .raw，纯文本一行一段
#   ./scripts/verify_variants_wikitext_smoke.sh
#
# 环境变量:
#   VERIFY_MODEL / MODEL          GGUF 路径（必填）
#   VERIFY_WIKITEXT / WIKITEXT_FILE / DATA  文本数据路径（必填）
#   VERIFY_ROOT                   llama.cpp 根目录（默认本脚本上级）
#   VERIFY_CHUNKS                 --chunks，默认 4（小批量）
#   VERIFY_CTX                    --ctx-size，默认 512
#   VERIFY_BATCH                  -b / --batch-size，默认 512
#   VERIFY_NGL_CPU                cpu_* 变体的 -ngl，默认 0
#   VERIFY_NGL_CUDA               cuda_* 变体的 -ngl，默认 99（大模型请按需改为 999）
#   VERIFY_THREADS                -t，默认 4（CPU 可调高；GPU 可 1~8）
#   VERIFY_TIMEOUT_SEC            每个变体 timeout，默认 7200（小模型可改 600）
#   VERIFY_PPL_EXTRA_ARGS         追加到 llama-perplexity 的额外参数（引号包裹）
#   VERIFY_VARIANTS               空格分隔的变体白名单，如 "cuda_only cuda_reex_fp16"；未设则扫描全部 build_verify_*
#   VERIFY_VARIANT_ORDER          变体执行顺序：gpu_first（默认，cuda_* 先于 cpu_*）| sorted（按名字字典序，与旧版一致）
#   CUDA_HOME / CUDA_PATH         同 verify_build_variants.sh
#
# Docker 示例（容器内挂载常为 /workspace，勿用宿主机 /home/... 路径）:
#   VERIFY_ROOT=/workspace/llama.cpp \\
#   VERIFY_MODEL=/workspace/llama.cpp/models/.../xxx.gguf \\
#   VERIFY_WIKITEXT=/workspace/llama.cpp/datasets/.../test.txt \\
#   ./scripts/verify_variants_wikitext_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ROOT="${VERIFY_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REEX_GATE_PROFILE="${REEX_GATE_PROFILE:-full}"
case "$REEX_GATE_PROFILE" in
  project|full) ;;
  *)
    echo "错误: 未知 REEX_GATE_PROFILE=$REEX_GATE_PROFILE（仅支持 project/full）" >&2
    exit 2
    ;;
esac
VERIFY_CHUNKS="${VERIFY_CHUNKS:-$([[ "$REEX_GATE_PROFILE" == "project" ]] && echo 64 || echo 128)}"
VERIFY_CHUNKS_CPU="${VERIFY_CHUNKS_CPU:-20}"
VERIFY_CHUNKS_CUDA="${VERIFY_CHUNKS_CUDA:-$VERIFY_CHUNKS}"
VERIFY_CHUNKS_CUDA_SLOW_REEX="${VERIFY_CHUNKS_CUDA_SLOW_REEX:-20}"
VERIFY_CTX="${VERIFY_CTX:-512}"
VERIFY_BATCH="${VERIFY_BATCH:-512}"
VERIFY_NGL_CPU="${VERIFY_NGL_CPU:-0}"
VERIFY_NGL_CUDA="${VERIFY_NGL_CUDA:-99}"
VERIFY_THREADS="${VERIFY_THREADS:-4}"
VERIFY_TIMEOUT_SEC="${VERIFY_TIMEOUT_SEC:-$([[ "$REEX_GATE_PROFILE" == "project" ]] && echo 1200 || echo 7200)}"
VERIFY_GPU_WARN_SEC="${VERIFY_GPU_WARN_SEC:-900}"
VERIFY_VARIANT_ORDER="${VERIFY_VARIANT_ORDER:-gpu_first}"
PROJECT_SMOKE_VARIANTS="${PROJECT_SMOKE_VARIANTS:-cuda_only cuda_reex_fp16 cuda_reex_gemm cuda_reex_full}"
FULL_SMOKE_VARIANTS="${FULL_SMOKE_VARIANTS:-cuda_only cuda_reex_fp16 cuda_reex_gemm cuda_reex_full cuda_reex_full_q8 cpu_default cpu_reex_fp16}"

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

# cuda_* 优先，其次 cpu_*，其余按名称排序（便于先观察 GPU / nvidia-smi）
reorder_variants_gpu_first() {
  local -a cuda_v=() cpu_v=() rest_v=() sorted=()
  local v
  for v in "$@"; do
    case "$v" in
      cuda_*) cuda_v+=("$v") ;;
      cpu_*) cpu_v+=("$v") ;;
      *) rest_v+=("$v") ;;
    esac
  done
  if ((${#cuda_v[@]})); then
    mapfile -t sorted < <(printf '%s\n' "${cuda_v[@]}" | sort -u)
    printf '%s\n' "${sorted[@]}"
  fi
  if ((${#cpu_v[@]})); then
    mapfile -t sorted < <(printf '%s\n' "${cpu_v[@]}" | sort -u)
    printf '%s\n' "${sorted[@]}"
  fi
  if ((${#rest_v[@]})); then
    mapfile -t sorted < <(printf '%s\n' "${rest_v[@]}" | sort -u)
    printf '%s\n' "${sorted[@]}"
  fi
}

# 若绝对路径不存在，再尝试 VERIFY_ROOT/<相对路径>（便于 MODEL=models/foo.gguf）
resolve_file() {
  local p="$1"
  local kind="$2"
  if [[ -z "$p" ]]; then
    echo "错误: ${kind} 为空。请设置 VERIFY_MODEL / VERIFY_WIKITEXT（或 MODEL / DATA）" >&2
    return 2
  fi
  if [[ -f "$p" ]]; then
    echo "$p"
    return 0
  fi
  if [[ -f "$VERIFY_ROOT/$p" ]]; then
    echo "$VERIFY_ROOT/$p"
    return 0
  fi
  echo "错误: ${kind} 文件不存在。" >&2
  echo "  已检查: $p" >&2
  echo "  已检查: $VERIFY_ROOT/$p" >&2
  echo "  当前 VERIFY_ROOT=$VERIFY_ROOT" >&2
  echo "  若在 Docker 内，请改用容器挂载路径（如 /workspace/llama.cpp/...），不要用宿主机 /home/... 路径。" >&2
  return 2
}

MODEL="$(resolve_file "${VERIFY_MODEL:-${MODEL:-}}" "VERIFY_MODEL")" || exit 2
WIKI="$(resolve_file "${VERIFY_WIKITEXT:-${WIKITEXT_FILE:-${DATA:-}}}" "VERIFY_WIKITEXT")" || exit 2

CUDA_LIB="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}/lib64"

collect_variants() {
  if [[ -n "${VERIFY_VARIANTS:-}" ]]; then
    read -r -a _COLLECTED <<< "${VERIFY_VARIANTS}"
    printf '%s\n' "${_COLLECTED[@]}"
    return
  fi
  if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
    local v lp
    read -r -a _PROJECT_VARIANTS <<< "${PROJECT_SMOKE_VARIANTS}"
    for v in "${_PROJECT_VARIANTS[@]}"; do
      lp="$VERIFY_ROOT/build_verify_${v}/bin/llama-perplexity"
      [[ -x "$lp" ]] && printf '%s\n' "$v"
    done
    return
  fi
  if [[ -n "${FULL_SMOKE_VARIANTS:-}" ]]; then
    local v lp
    read -r -a _FULL_VARIANTS <<< "${FULL_SMOKE_VARIANTS}"
    for v in "${_FULL_VARIANTS[@]}"; do
      lp="$VERIFY_ROOT/build_verify_${v}/bin/llama-perplexity"
      [[ -x "$lp" ]] && printf '%s\n' "$v"
    done
    return
  fi
  local d lp
  (
    shopt -s nullglob
    for d in "$VERIFY_ROOT"/build_verify_*; do
      [[ -d "$d" ]] || continue
      lp="$d/bin/llama-perplexity"
      [[ -x "$lp" ]] && basename "$d" | sed 's/^build_verify_//'
    done
  ) | sort -u
}

mapfile -t VARIANT_LIST < <(collect_variants)
if [[ "${VERIFY_VARIANT_ORDER:-gpu_first}" == "gpu_first" ]] && ((${#VARIANT_LIST[@]} > 0)); then
  mapfile -t VARIANT_LIST < <(reorder_variants_gpu_first "${VARIANT_LIST[@]}")
fi
if [[ "${#VARIANT_LIST[@]}" -eq 0 ]]; then
  echo "未找到 $VERIFY_ROOT/build_verify_*/bin/llama-perplexity。请先运行 ./scripts/verify_build_variants.sh" >&2
  exit 3
fi

echo "=== Wikitext 小批量 PPL 冒烟（各 build_verify_*）==="
echo "  REEX_GATE_PROFILE=$REEX_GATE_PROFILE"
echo "  VERIFY_ROOT=$VERIFY_ROOT"
echo "  MODEL=$MODEL"
echo "  WIKITEXT=$WIKI"
echo "  chunks: cpu_*=$VERIFY_CHUNKS_CPU  cuda_*=$VERIFY_CHUNKS_CUDA  cuda_reex_{gemm,full}=$VERIFY_CHUNKS_CUDA_SLOW_REEX"
echo "  ctx=$VERIFY_CTX batch=$VERIFY_BATCH threads=$VERIFY_THREADS"
echo "  ngl: cpu_*=$VERIFY_NGL_CPU  cuda_*=$VERIFY_NGL_CUDA"
echo "  GPU 慢路径告警阈值: ${VERIFY_GPU_WARN_SEC}s"
echo "  VARIANT_ORDER=$VERIFY_VARIANT_ORDER"
if [[ "$REEX_GATE_PROFILE" == "project" ]]; then
  echo "  PROJECT_SMOKE_VARIANTS=$PROJECT_SMOKE_VARIANTS"
else
  echo "  FULL_SMOKE_VARIANTS=$FULL_SMOKE_VARIANTS"
fi
echo "  变体数: ${#VARIANT_LIST[@]} → ${VARIANT_LIST[*]}"
echo ""

PASS=0
FAIL=0
declare -a PPL_LINES=()

ngl_for_variant() {
  local v="$1"
  if [[ "$v" == cuda_* ]]; then
    echo "$VERIFY_NGL_CUDA"
  else
    echo "$VERIFY_NGL_CPU"
  fi
}

chunks_for_variant() {
  local v="$1"
  if [[ "$v" == "cuda_reex_gemm" || "$v" == "cuda_reex_full" ]]; then
    echo "$VERIFY_CHUNKS_CUDA_SLOW_REEX"
    return
  fi
  if [[ "$v" == cuda_* ]]; then
    echo "$VERIFY_CHUNKS_CUDA"
  else
    echo "$VERIFY_CHUNKS_CPU"
  fi
}

batch_for_variant() {
  local v="$1"
  if [[ "$v" == "cuda_reex_gemm" || "$v" == "cuda_reex_full" ]]; then
    # The current CUDA W4xQ16 REEX mul_mat kernel only covers batch <= 8.
    # Keep smoke tests on the intended fast path instead of silently falling back.
    if (( VERIFY_BATCH > 8 )); then
      echo 8
      return
    fi
  fi
  echo "$VERIFY_BATCH"
}

gpu_log_has_markers() {
  local log="$1"
  grep -qE 'ggml_cuda_init: found|load_tensors: offloaded [0-9]+/[0-9]+ layers to GPU|CUDA0' "$log"
}

gpu_log_has_fallback() {
  local log="$1"
  grep -qE 'offloaded 0/[0-9]+ layers to GPU|device none|no CUDA-capable device is detected|failed to initialize CUDA|CUDA driver version is insufficient|falling back to CPU|using CPU' "$log"
}

run_one() {
  local v="$1"
  local bdir="$VERIFY_ROOT/build_verify_${v}"
  local lp="$bdir/bin/llama-perplexity"
  local ngl
  local chunks
  local batch
  local start_epoch
  ngl="$(ngl_for_variant "$v")"
  chunks="$(chunks_for_variant "$v")"
  batch="$(batch_for_variant "$v")"
  start_epoch="$(date +%s)"
  local ldpath="$bdir/bin"
  [[ -d "$CUDA_LIB" ]] && ldpath="$ldpath:$CUDA_LIB"
  [[ -n "${LD_LIBRARY_PATH:-}" ]] && ldpath="$ldpath:$LD_LIBRARY_PATH"

  if [[ ! -x "$lp" ]]; then
    echo "  [跳过/失败] 无可执行文件: $lp"
    FAIL=$((FAIL + 1))
    return
  fi

  echo "---------- ${v} (ngl=${ngl}, batch=${batch}, chunks=${chunks}) ----------"
  local log
  log="$(mktemp)"
  # shellcheck disable=SC2086
  set +e
  env LD_LIBRARY_PATH="$ldpath" timeout "$VERIFY_TIMEOUT_SEC" \
    "$lp" -m "$MODEL" -f "$WIKI" \
    --ctx-size "$VERIFY_CTX" -b "$batch" \
    --chunks "$chunks" \
    -ngl "$ngl" -t "$VERIFY_THREADS" \
    ${VERIFY_PPL_EXTRA_ARGS:-} \
    2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [[ "$rc" -ne 0 ]]; then
    echo "  [失败] llama-perplexity 退出码 $rc（$v）"
    FAIL=$((FAIL + 1))
    rm -f "$log"
    return
  fi
  if ! grep -qE 'Final estimate: PPL|Mean PPL\(Q\)' "$log"; then
    echo "  [失败] 输出中未找到 PPL 汇总行（$v），请检查数据格式与参数"
    FAIL=$((FAIL + 1))
    rm -f "$log"
    return
  fi
  local summary
  summary="$(grep -E 'Final estimate: PPL|Mean PPL\(Q\)' "$log" | tail -n 3)"
  PPL_LINES+=("$v | $(echo "$summary" | tr '\n' ' ' | sed 's/  */ /g')")
  local duration_sec=$(( $(date +%s) - start_epoch ))
  if [[ "$v" == cuda_* ]]; then
    if gpu_log_has_fallback "$log"; then
      echo "  [失败] 检测到 GPU 变体疑似退回 CPU / CUDA 初始化异常（$v）"
      FAIL=$((FAIL + 1))
      rm -f "$log"
      return
    fi
    if ! gpu_log_has_markers "$log"; then
      echo "  [失败] 未在日志中检测到明确 GPU 执行标记（$v），请检查是否真正使用 CUDA"
      FAIL=$((FAIL + 1))
      rm -f "$log"
      return
    fi
    local perf
    perf="$(grep -E 'tokens per second|tok/s|prompt eval time|second.*per pass|pp[0-9]+|t/s' "$log" | tail -n 5 | tr '\n' ' ' | sed 's/  */ /g')"
    [[ -n "$perf" ]] && echo "  [性能] $perf"
    if (( duration_sec > VERIFY_GPU_WARN_SEC )); then
      echo "  [告警] GPU 变体耗时偏长（$v: $(format_duration "$duration_sec")），请关注是否存在阻塞或效率异常"
    fi
  fi
  echo "  [通过] PPL 冒烟: $v（耗时: $(format_duration "$duration_sec")）"
  PASS=$((PASS + 1))
  rm -f "$log"
}

for v in "${VARIANT_LIST[@]}"; do
  run_one "$v"
  echo ""
done

echo "=== PPL 冒烟汇总 ==="
echo "  通过: $PASS   失败: $FAIL"
for line in "${PPL_LINES[@]:-}"; do
  [[ -n "$line" ]] && echo "  $line"
done

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
exit 0
