#!/usr/bin/env bash
# Smoke-test a CPU-only rx-met Docker image (no CUDA / no quant_gru).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCT_VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
VERSION="${RX_MET_VERSION:-${PRODUCT_VERSION}}"
IMAGE="${RX_MET_IMAGE:-rx-met:${VERSION}-cpu}"

log() { printf '[verify_image_cpu] %s\n' "$*"; }

if [[ "${VERSION}" != "${PRODUCT_VERSION}" ]]; then
    log "ERROR: RX_MET_VERSION=${VERSION} differs from ${ROOT}/VERSION (${PRODUCT_VERSION})"
    exit 1
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    log "ERROR: image not found: ${IMAGE}"
    exit 1
fi

log "rx-met --help"
docker run --rm "${IMAGE}" rx-met --help

log "torch + aimet imports (no quant_gru)"
docker run --rm "${IMAGE}" python3 -c "
import torch
import torchvision
import aimet_torch
import onnxscript
import onnxsim
import rx_met_llm
import gguf
import transformers
assert rx_met_llm.__version__ == '${VERSION}', (rx_met_llm.__version__, '${VERSION}')
assert torch.cuda.is_available() is False, 'CPU image must not report CUDA'
print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.cuda.is_available())
print('onnxscript', onnxscript.__version__)
print('rx_met_llm', rx_met_llm.__version__)
try:
    import quant_gru
except ImportError:
    print('quant_gru absent (expected for CPU release)')
else:
    raise SystemExit('quant_gru must not be installed in CPU image')
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

log "all checks passed for ${IMAGE}"
