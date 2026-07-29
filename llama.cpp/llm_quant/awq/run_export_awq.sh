#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/data2/models/Qwen3.5-35B-A3B"
OUTPUT_DIR="/data2/yan.huang.srv/Qwen3.5-35B-A3B-q64/lut/AWQ"
export AWQ_PILEVAL_PATH="/home/yan.huang.srv/llm_eval/datasets/val.jsonl"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

python "$REPO_ROOT/export_awq_hf.py" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --w_bit 4 \
  --q_group_size 64 \
  --n_samples 128 \
  --seqlen 512 \
  --calib_data pileval
