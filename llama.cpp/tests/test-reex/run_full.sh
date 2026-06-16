#!/usr/bin/env bash
# Full 6-case MoE prefill capture pipeline (GPU per-64 + LUT build).
set -u
cd /mnt/data8t/zcx/aimet_rx/llama.cpp
export LD_LIBRARY_PATH="$PWD/build_cuda_q64/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export DUMP_MOE_CONTAINER=quant-gru-cuda128
export DUMP_MOE_GIT_COMMIT="$(git rev-parse HEAD)"
OUT=/mnt/data8t/zcx/aimet_rx/llama.cpp/output

echo "=== DUMP start $(date) ==="
./build_cuda_q64/bin/dump_moe_prefill -o "$OUT" \
  && echo "=== POST start $(date) ===" \
  && python3 tests/test-reex/moe_postprocess.py --root "$OUT" \
  && echo "=== VALIDATE start $(date) ===" \
  && python3 tests/test-reex/moe_validate.py --root "$OUT"
echo "=== PIPELINE exit=$? $(date) ==="
