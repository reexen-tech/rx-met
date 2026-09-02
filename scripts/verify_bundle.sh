#!/usr/bin/env bash
# Verify an ada200-rx-met software bundle against ada200_docker:latest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${1:-}"
IMAGE="${RX_MET_ADA200_IMAGE:-ada200_docker:latest}"

log() { printf '[verify_bundle] %s\n' "$*"; }
die() { printf '[verify_bundle] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ -z "${BUNDLE}" ]]; then
    latest="$(ls -1t "${ROOT}/.release/export"/ada200-rx-met-v*-linux_x86_64.tar.gz 2>/dev/null | head -1 || true)"
    [[ -n "${latest}" ]] || die "usage: ./scripts/verify_bundle.sh /path/to/ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz"
    BUNDLE="${latest}"
fi
[[ -f "${BUNDLE}" ]] || die "bundle not found: ${BUNDLE}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    die "image not found: ${IMAGE}"
fi

WORK="$(mktemp -d -t rx-met-verify.XXXXXX)"
cleanup() { rm -rf "${WORK}"; }
trap cleanup EXIT

log "extract $(basename "${BUNDLE}")"
tar -C "${WORK}" -xzf "${BUNDLE}"
shopt -s nullglob
dirs=("${WORK}"/ada200-rx-met-v*-linux_x86_64)
shopt -u nullglob
[[ ${#dirs[@]} -eq 1 ]] || die "expected one top-level ada200-rx-met-* directory"
PKG="${dirs[0]}"

log "layout"
[[ -x "${PKG}/install.sh" ]] || die "install.sh missing or not executable"
[[ -f "${PKG}/README.md" ]] || die "README.md missing"
[[ -f "${PKG}/ChangeLog.md" ]] || die "ChangeLog.md missing"
[[ -f "${PKG}/examples/quick_start_kws.py" ]] || die "quick_start_kws.py missing"
[[ -f "${PKG}/examples/onnx_ptq_quick_start.py" ]] || die "onnx_ptq_quick_start.py missing"
[[ -f "${PKG}/examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json" ]] \
    || die "QuantSim config JSON missing"
[[ -f "${PKG}/examples/config/quick_start_full_quant.json" ]] \
    || die "bitwidth config JSON missing"
[[ ! -e "${PKG}/examples/quick_start.py" ]] || die "legacy MRNN quick_start.py must not be packaged"
[[ ! -e "${PKG}/examples/data/to_band_matrix_erb_240_256.pt" ]] \
    || die "legacy MRNN band matrix must not be packaged"
[[ ! -e "${PKG}/examples/llm_quick_start.py" ]] || die "LLM example must not be packaged"
[[ ! -e "${PKG}/examples/config/llm_quant.json" ]] || die "llm_quant.json must not be packaged"
[[ ! -e "${PKG}/scripts/setup.sh" ]] || die "legacy setup.sh must not be packaged"
if grep -R -n --include='*.py' --include='*.json' -E 'rx_met_llm|llm_quant|llama-quantize' "${PKG}/examples" >/dev/null; then
    die "examples still mention LLM artifacts"
fi
if unzip -l "${PKG}"/wheels/rx_met-*.whl | grep -q rx_met_llm; then
    die "rx_met wheel contains rx_met_llm"
fi

INNER="${WORK}/in_container.sh"
cat > "${INNER}" <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
cp -a /opt/rx-met /tmp/rx-met
cd /tmp/rx-met
./install.sh
python3 - <<'PY'
import os
import runpy
import sys

import torch
import aimet_torch
import aimet_onnx
import quant_gru
import torchaudio
import librosa
import soundfile

assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "GPU not visible inside ada200_docker"
try:
    import rx_met_llm
except ImportError:
    pass
else:
    raise SystemExit("rx_met_llm must not be importable")
print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0))
print("aimet_torch", aimet_torch.__file__)
print("quant_gru", quant_gru.__file__)

os.chdir("/tmp/rx-met/examples")
sys.path.insert(0, "/tmp/rx-met/examples")
ns = runpy.run_path("quick_start_kws.py", run_name="verify_rx_met")
assert ns["DATA_ROOT"] == "/datasets/speech_commands_v0.02"
print("quick_start_kws import OK")
ns = runpy.run_path("onnx_ptq_quick_start.py", run_name="verify_rx_met")
assert ns["USE_CPU"] is True
assert str(ns["_DATA"]) == "/datasets"
print("onnx_ptq defaults OK", ns["_DATA"])
PY
python3 -m pip check
INNER
chmod +x "${INNER}"

log "install + imports in ${IMAGE}"
docker run --rm --gpus all \
    -e RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02 \
    -v "${PKG}:/opt/rx-met:ro" \
    -v "${INNER}:/tmp/in_container.sh:ro" \
    -w /tmp \
    "${IMAGE}" \
    bash /tmp/in_container.sh

log "all checks passed for ${BUNDLE}"
