#!/usr/bin/env bash
# === REEX_SMOOTHQUANT BEGIN: KL-divergence 对比（FP16 基准 vs 各量化变体） ===
# 两步法（llama-perplexity 内建）:
#   1) 用未量化的 f16 跑一遍，把每个 token 位置的概率分布存成 .kld 基准文件
#   2) 待测模型跑同样的文本，逐 token 与基准比对，输出 KLD / top-token 一致率 等
#
# -c 与 --chunks 两步必须完全一致，否则比对的 token 位置会错位。
#
# 磁盘开销: 基准文件 ≈ 74 MB/chunk（n_vocab=152064, n_ctx=512），跑之前先确认空间。
#
# 用法（仓库根目录，容器内或宿主机）:
#   ./llm_quant/smoothquant/scripts/run_smoothquant_kld.sh \
#     <out_dir> [chunks] [--quant-type Q4_K_64] [--reuse-base]
set -euo pipefail

usage() {
  echo "用法: $0 <out_dir> [chunks] [--quant-type TYPE] [--reuse-base]" >&2
}

if (( $# < 1 )); then
  usage
  exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  usage
  exit 0
fi

OUT_DIR="$1"
shift
CHUNKS="256"
QUANT_TYPE="${REEX_SQ_QUANT_TYPE:-Q8_0_64}"
REUSE_BASE=0

if (( $# > 0 )) && [[ "$1" != --* ]]; then
  CHUNKS="$1"
  shift
fi

while (( $# > 0 )); do
  case "$1" in
    --quant-type)
      if (( $# < 2 )); then
        usage
        exit 2
      fi
      QUANT_TYPE="$2"
      shift 2
      ;;
    --reuse-base)
      REUSE_BASE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "错误: 未知参数 $1" >&2
      usage
      exit 2
      ;;
  esac
done

QUANT_TYPE="${QUANT_TYPE^^}"
if [[ ! "$QUANT_TYPE" =~ ^[A-Z0-9_]+$ ]]; then
  echo "错误: quant type 只能包含字母、数字和下划线: $QUANT_TYPE" >&2
  exit 2
fi
if [[ ! "$CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误: chunks 必须是正整数: $CHUNKS" >&2
  exit 2
fi
QUANT_TAG="${QUANT_TYPE,,}"

BIN="${REEX_SQ_BIN:-./build_cuda_q64/bin}"
F="${REEX_SQ_PPL_DATASET:-/home/zcx/sq_exp/data/wiki.test.raw}"
RUNS="${REEX_SQ_RUNS:-/mnt/data8t/zcx/sq_exp/runs}"
NGL="${REEX_SQ_NGL:-99}"
CTX="${REEX_SQ_CTX:-512}"

BASE_MODEL="$RUNS/pileval-alpha0.85/model-f16.gguf"
KLD_BASE="$OUT_DIR/f16.logits.kld"

mkdir -p "$OUT_DIR/logs"

if (( REUSE_BASE )); then
  if [[ ! -s "$KLD_BASE" ]]; then
    echo "错误: --reuse-base 指定的基准文件不存在或为空: $KLD_BASE" >&2
    exit 1
  fi
  echo "=== 复用 KLD 基准: $KLD_BASE (chunks=$CHUNKS, ctx=$CTX) ==="
else
  echo "=== KLD 基准: $BASE_MODEL (chunks=$CHUNKS, ctx=$CTX) ==="
  "$BIN/llama-perplexity" -m "$BASE_MODEL" -ngl "$NGL" -f "$F" -c "$CTX" --chunks "$CHUNKS" \
    --kl-divergence-base "$KLD_BASE" 2>&1 | tee "$OUT_DIR/logs/00_base_f16.log"
fi

# 待测模型: <标签>=<gguf 路径>
TARGETS=(
  "baseline_${QUANT_TAG}=$RUNS/pileval-alpha0.85/model-baseline-${QUANT_TYPE}.gguf"
  "smooth_${QUANT_TAG}_a0.85=$RUNS/pileval-alpha0.85/model-smooth-${QUANT_TYPE}.gguf"
  "smooth_${QUANT_TAG}_a0.80=$RUNS/pileval-alpha0.8/model-smooth-${QUANT_TYPE}.gguf"
)

for entry in "${TARGETS[@]}"; do
  model="${entry#*=}"
  if [[ ! -f "$model" ]]; then
    echo "错误: 找不到待测模型 $model" >&2
    exit 1
  fi
done

i=0
for entry in "${TARGETS[@]}"; do
  i=$((i + 1))
  label="${entry%%=*}"
  model="${entry#*=}"
  echo ""
  echo "=== KLD 比对 [$i/${#TARGETS[@]}]: $label ==="
  "$BIN/llama-perplexity" -m "$model" -ngl "$NGL" -f "$F" -c "$CTX" --chunks "$CHUNKS" \
    --kl-divergence-base "$KLD_BASE" --kl-divergence 2>&1 \
    | tee "$OUT_DIR/logs/$(printf '%02d' "$i")_${label}.log"
done

echo ""
echo "=== 完成，日志在 $OUT_DIR/logs/ ==="
echo "基准文件 $KLD_BASE 可在比对结束后删除（$(du -h "$KLD_BASE" | cut -f1)）"
# === REEX_SMOOTHQUANT END ===
