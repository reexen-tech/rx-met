#!/usr/bin/env bash
# GPU-only rx-met software bundle for ada200_docker (wheels + examples, no image).
# DATE_TAG=vYYMMDD, PEP 440 VERSION=YY.M.D
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATE_TAG_RAW="${RX_MET_RELEASE_DATE:-$(date +%y%m%d)}"
if [[ ! "${DATE_TAG_RAW}" =~ ^[0-9]{6}$ ]]; then
    printf 'ERROR: RX_MET_RELEASE_DATE must be YYMMDD, got %s\n' "${DATE_TAG_RAW}" >&2
    exit 1
fi
DATE_TAG="v${DATE_TAG_RAW}"
VERSION="${DATE_TAG_RAW:0:2}.$((10#${DATE_TAG_RAW:2:2})).$((10#${DATE_TAG_RAW:4:2}))"

EXPORT_DIR="${RX_MET_EXPORT_DIR:-${ROOT}/.release/export}"
STAGING_ROOT="${ROOT}/.release/staging"
WHEEL_EXPORT="${ROOT}/.release/rx-met-wheels"
RELEASE_NAME="ada200-rx-met-${DATE_TAG}-linux_x86_64"
RELEASE_BUNDLE="${EXPORT_DIR}/${RELEASE_NAME}.tar.gz"
RELEASE_SHA256="${RELEASE_BUNDLE}.sha256"
GZIP_LEVEL="${RX_MET_GZIP_LEVEL:-6}"
CUDA_ARCHITECTURES="${RX_MET_CUDA_ARCHITECTURES:-80;86;89;90;120}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}"
STAGE_PARENT=""

log() { printf '[release_build] %s\n' "$*"; }
cleanup() {
    [[ -z "${STAGE_PARENT}" ]] || rm -rf "${STAGE_PARENT}"
}
trap cleanup EXIT

rx_met_docker_proxy_args() {
    if [[ -z "${RX_MET_BUILD_PROXY:-}" ]]; then
        return 0
    fi
    printf '%s\n' \
        --network host \
        --build-arg "HTTP_PROXY=${RX_MET_BUILD_PROXY}" \
        --build-arg "HTTPS_PROXY=${RX_MET_BUILD_PROXY}" \
        --build-arg "http_proxy=${RX_MET_BUILD_PROXY}" \
        --build-arg "https_proxy=${RX_MET_BUILD_PROXY}" \
        --build-arg "NO_PROXY=localhost,127.0.0.1"
}

rx_met_compress_release_bundle() {
    local temp_bundle="${RELEASE_BUNDLE}.tmp.$$"
    mkdir -p "${EXPORT_DIR}"
    tar -C "${STAGE_PARENT}" -cf - "${RELEASE_NAME}" | gzip "-${GZIP_LEVEL}" > "${temp_bundle}"
    mv -f "${temp_bundle}" "${RELEASE_BUNDLE}"
    (
        cd "${EXPORT_DIR}"
        sha256sum "$(basename "${RELEASE_BUNDLE}")" > "$(basename "${RELEASE_SHA256}")"
    )
}

if [[ ! "${GZIP_LEVEL}" =~ ^[1-9]$ ]]; then
    log "ERROR: RX_MET_GZIP_LEVEL must be an integer from 1 to 9"
    exit 1
fi
for required in \
    "${ROOT}/docker/Dockerfile" \
    "${ROOT}/release/install.sh" \
    "${ROOT}/release/README.md" \
    "${ROOT}/release/ChangeLog.md"
do
    [[ -f "${required}" ]] || { log "ERROR: missing ${required}"; exit 1; }
done

export RX_MET_VERSION="${VERSION}"

docker_args=(
    --target export
    --build-arg "RX_MET_VERSION=${VERSION}"
    --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
    --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    -f "${ROOT}/docker/Dockerfile"
    -o "${WHEEL_EXPORT}"
)

if [[ -n "${RX_MET_BUILD_PROXY:-}" ]]; then
    log "build proxy enabled: ${RX_MET_BUILD_PROXY}"
    while IFS= read -r arg; do
        [[ -n "${arg}" ]] || continue
        docker_args+=("${arg}")
    done < <(rx_met_docker_proxy_args)
fi

rm -rf "${WHEEL_EXPORT}"
mkdir -p "${WHEEL_EXPORT}" "${EXPORT_DIR}" "${STAGING_ROOT}"

log "builder docker build -> ${WHEEL_EXPORT} (version=${VERSION}, tag=${DATE_TAG})"
DOCKER_BUILDKIT=1 docker build "${docker_args[@]}" "${ROOT}"

WHEELS_DIR="${WHEEL_EXPORT}/wheels"
[[ -d "${WHEELS_DIR}" ]] || { log "ERROR: export missing ${WHEELS_DIR}"; exit 1; }
shopt -s nullglob
rx_met_wheels=("${WHEELS_DIR}"/rx_met-*.whl)
quant_gru_wheels=("${WHEELS_DIR}"/quant_gru-*.whl)
shopt -u nullglob
[[ ${#rx_met_wheels[@]} -eq 1 ]] || { log "ERROR: expected one rx_met wheel"; exit 1; }
[[ ${#quant_gru_wheels[@]} -eq 1 ]] || { log "ERROR: expected one quant_gru wheel"; exit 1; }

STAGE_PARENT="$(mktemp -d "${STAGING_ROOT}/${RELEASE_NAME}.XXXXXX")"
RELEASE_DIR="${STAGE_PARENT}/${RELEASE_NAME}"
mkdir -p "${RELEASE_DIR}/wheels" \
    "${RELEASE_DIR}/examples/common" \
    "${RELEASE_DIR}/examples/config" \
    "${RELEASE_DIR}/examples/data"

log "assemble bundle -> ${RELEASE_NAME}"
cp -a "${WHEELS_DIR}"/. "${RELEASE_DIR}/wheels/"
cp "${ROOT}/examples/quick_start.py" "${RELEASE_DIR}/examples/"
cp "${ROOT}/examples/onnx_ptq_quick_start.py" "${RELEASE_DIR}/examples/"
cp "${ROOT}/examples/common/fft2band.py" "${RELEASE_DIR}/examples/common/"
cp "${ROOT}/examples/common/torch_stft.py" "${RELEASE_DIR}/examples/common/"
cp "${ROOT}/examples/data/to_band_matrix_erb_240_256.pt" \
    "${ROOT}/examples/data/inv_to_band_matrix_erb_240_256.pt" \
    "${RELEASE_DIR}/examples/data/"
cp "${ROOT}/examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json" \
    "${RELEASE_DIR}/examples/config/"
cp "${ROOT}/examples/config/quick_start_full_quant.json" \
    "${RELEASE_DIR}/examples/config/"
cp "${ROOT}/examples/config/README.md" \
    "${RELEASE_DIR}/examples/config/README.md"
cp "${ROOT}/release/install.sh" "${RELEASE_DIR}/install.sh"
cp "${ROOT}/release/README.md" "${RELEASE_DIR}/README.md"
cp "${ROOT}/release/ChangeLog.md" "${RELEASE_DIR}/ChangeLog.md"
chmod +x "${RELEASE_DIR}/install.sh"
sed -i \
    -e "s/@VERSION@/${VERSION}/g" \
    -e "s/@DATE_TAG@/${DATE_TAG}/g" \
    -e "s/@BUNDLE_NAME@/${RELEASE_NAME}/g" \
    "${RELEASE_DIR}/README.md" \
    "${RELEASE_DIR}/ChangeLog.md"

if [[ -e "${RELEASE_DIR}/examples/llm_quick_start.py" ]] \
    || [[ -e "${RELEASE_DIR}/examples/config/llm_quant.json" ]]; then
    log "ERROR: LLM examples leaked into the bundle"
    exit 1
fi

log "compress -> ${RELEASE_BUNDLE}"
rx_met_compress_release_bundle

log "done:"
log "  version: ${VERSION} (${DATE_TAG})"
log "  release: ${RELEASE_BUNDLE} ($(du -h "${RELEASE_BUNDLE}" | cut -f1))"
log "  sha256:  ${RELEASE_SHA256}"
