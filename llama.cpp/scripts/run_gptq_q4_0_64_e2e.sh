#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 MODEL_PATH WIKITEXT_TEST OUTPUT_DIR" >&2
    exit 2
fi

MODEL_PATH=$1
WIKITEXT_TEST=$2
OUTPUT_DIR=$3
CUDA_DEVICE=${CUDA_DEVICE:-5}
BUILD_DIR=${BUILD_DIR:-build_cuda}

HF_GPTQ_DIR="${OUTPUT_DIR}/qwen2.5-0.5b-gptq-hf"
F16_GGUF="${OUTPUT_DIR}/qwen2.5-0.5b-f16.gguf"
GPTQ_GGUF="${OUTPUT_DIR}/qwen2.5-0.5b-gptq-Q4_0_64.gguf"
RTN_GGUF="${OUTPUT_DIR}/qwen2.5-0.5b-rtn-Q4_0_64.gguf"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python export_gptq_hf.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${HF_GPTQ_DIR}"

python convert_hf_to_gguf.py \
    "${HF_GPTQ_DIR}" \
    --outtype f16 \
    --outfile "${GPTQ_GGUF}"

python validate_gptq_gguf.py \
    --sidecar "${HF_GPTQ_DIR}/gptq_q4_0_64.pt" \
    --gguf "${GPTQ_GGUF}"

python convert_hf_to_gguf.py \
    "${MODEL_PATH}" \
    --outtype f16 \
    --outfile "${F16_GGUF}"

"${BUILD_DIR}/bin/llama-quantize" --pure \
    --token-embedding-type f16 \
    --output-tensor-type f16 \
    --leave-output-tensor \
    "${F16_GGUF}" "${RTN_GGUF}" Q4_0_64

for variant in f16 rtn gptq; do
    case "${variant}" in
        f16)  model="${F16_GGUF}" ;;
        rtn)  model="${RTN_GGUF}" ;;
        gptq) model="${GPTQ_GGUF}" ;;
    esac
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        "${BUILD_DIR}/bin/llama-perplexity" \
        -m "${model}" \
        -f "${WIKITEXT_TEST}" \
        -ngl 99 -c 512 --chunks 4 \
        2>&1 | tee "${OUTPUT_DIR}/ppl-${variant}.log"
done

PROMPT="The capital of France is"
for backend in cpu cuda; do
    if [[ "${backend}" == "cpu" ]]; then
        ngl=0
        visible_devices=
    else
        ngl=99
        visible_devices=${CUDA_DEVICE}
    fi
    CUDA_VISIBLE_DEVICES="${visible_devices}" \
        "${BUILD_DIR}/bin/llama-completion" \
        -m "${GPTQ_GGUF}" \
        -p "${PROMPT}" \
        -n 64 --temp 0 --seed 42 -ngl "${ngl}" \
        --no-conversation --simple-io --log-disable --no-display-prompt \
        > "${OUTPUT_DIR}/generation-${backend}.txt"
done

cmp "${OUTPUT_DIR}/generation-cpu.txt" "${OUTPUT_DIR}/generation-cuda.txt"
echo "GPTQ Q4_0_64 end-to-end validation completed."
