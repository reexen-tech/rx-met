#!/usr/bin/env bash
# Dump ONE 4096-token, top-4 MoE expert-reorder case in the same layout as
# output/case_00 (router / expert-major reorder / bounds / token view).
#
# top-4 is a RUNTIME override of the model's expert_used_count (native = 8) via
# --override-kv; the reorder capture (REEX_DUMP_MMID_DIR) is taken live from the
# real GPU tensors of this same prefill, so K=4 is reflected automatically.
#
# Overridable env: MODEL, OUT, NTOK, TOPK, ARCH_KEY.
set -euo pipefail
cd /mnt/data8t/zcx/aimet_rx/llama.cpp
export LD_LIBRARY_PATH="$PWD/build_cuda_q64/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export DUMP_MOE_CONTAINER="${DUMP_MOE_CONTAINER:-quant-gru-cuda128}"
export DUMP_MOE_GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

MODEL="${MODEL:-/mnt/data8t/share/models/Qwen/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-Q4_0_64.gguf}"
OUT="${OUT:-/mnt/data8t/zcx/aimet_rx/llama.cpp/output_tok4_4k}"
NTOK="${NTOK:-4096}"
TOPK="${TOPK:-4}"
ARCH_KEY="${ARCH_KEY:-qwen35moe.expert_used_count}"
CASE="$OUT/case_00"

mkdir -p "$CASE/mmid_raw"
export REEX_DUMP_MMID_DIR="$CASE/mmid_raw"

echo "=== DUMP start $(date)  (n_tokens=$NTOK, top-$TOPK) ==="
./build_cuda_q64/bin/dump_moe_prefill \
    -m "$MODEL" -o "$OUT" --cases 1 --n-tokens "$NTOK" \
    --override-kv "${ARCH_KEY}=int:${TOPK}"

echo "=== POST (mmid expert-reorder) start $(date) ==="
python3 tests/test-reex/mmid_postprocess.py "$CASE"

echo "=== POST (token-centric view) start $(date) ==="
python3 tests/test-reex/make_token_expert_slots.py "$CASE"

# The unconditional MoE value dump (case_00/raw/*.bin, several GB) is not part of
# the case_00-style reorder product; drop it. Keep mmid_raw for re-verification.
rm -rf "$CASE/raw"

echo "=== PIPELINE done $(date) ==="
echo "case dir: $CASE"
ls "$CASE"
