#!/usr/bin/env bash
# Smoke-test a built rx-met Docker image (§3.6 in Release_packaging.md).
set -euo pipefail

VERSION="${RX_MET_VERSION:-1.0.0}"
IMAGE="${RX_MET_IMAGE:-rx-met:${VERSION}}"

log() { printf '[verify_image] %s\n' "$*"; }

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    log "ERROR: image not found: ${IMAGE}"
    exit 1
fi

log "rx-met --help"
docker run --rm "${IMAGE}" rx-met --help

log "torch + aimet imports"
docker run --rm "${IMAGE}" python3 -c "
import torch
import torchvision
import aimet_torch
import onnxscript
import quant_gru
import rx_met_llm
import gguf
import transformers
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())
print('onnxscript', onnxscript.__version__)
print('quant_gru', quant_gru.__file__)
print('rx_met_llm', rx_met_llm.__version__)
"

log "Python dependency consistency"
docker run --rm "${IMAGE}" python3 -m pip check

log "native binaries under RX_MET_HOME"
docker run --rm "${IMAGE}" bash -c '
set -e
test -x "${RX_MET_HOME}/bin/llama-quantize"
test -x "${RX_MET_HOME}/bin/llama-imatrix"
test -x "${RX_MET_HOME}/bin/llama-perplexity"
test -x "${RX_MET_HOME}/bin/reex-hw-convert"
test -f "${RX_MET_HOME}/bin/convert_hf_to_gguf.py"
echo "binaries ok"
'

if [[ "${SKIP_GPU_TEST:-0}" == "1" ]]; then
    log "SKIP_GPU_TEST=1, skipping required GPU smoke test"
else
    log "GPU torch.cuda.is_available()"
    docker run --rm --gpus all "${IMAGE}" \
        python3 -c "import torch; assert torch.cuda.is_available()"
fi

log "all checks passed for ${IMAGE}"
