#!/usr/bin/env bash
# GPU-only rx-met software bundle for ada200_docker (wheels + examples, no image).
# Requires team image ada200_docker:rx-met-dev from scripts/build_dev_image.sh.
# Component / wheel version: VERSION file (MAJOR.MINOR.PATCH).
# Tarball name: ada200-rx-met-vYYMMDD-linux_x86_64.tar.gz (SDK package tag).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_IMAGE="${RX_MET_DEV_IMAGE:-ada200_docker:rx-met-dev}"

VERSION_FILE="${ROOT}/VERSION"
[[ -f "${VERSION_FILE}" ]] || { printf 'ERROR: missing %s\n' "${VERSION_FILE}" >&2; exit 1; }
VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}")"
if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    printf 'ERROR: VERSION must be MAJOR.MINOR.PATCH, got %s\n' "${VERSION}" >&2
    exit 1
fi
SEMVER="v${VERSION}"

DATE_TAG_RAW="${RX_MET_RELEASE_DATE:-${RX_MET_SDK_DATE:-$(date +%y%m%d)}}"
if [[ ! "${DATE_TAG_RAW}" =~ ^[0-9]{6}$ ]]; then
    printf 'ERROR: RX_MET_RELEASE_DATE must be YYMMDD, got %s\n' "${DATE_TAG_RAW}" >&2
    exit 1
fi
DATE_TAG="v${DATE_TAG_RAW}"
SDK_TAG="${DATE_TAG}"

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

rx_met_require_image_name() {
    local name="$1"
    if [[ ! "${name}" =~ ^[A-Za-z0-9._/-]+(:[A-Za-z0-9._-]+)?$ ]]; then
        printf 'ERROR: invalid image name: %s\n' "${name}" >&2
        exit 1
    fi
}

rx_met_assemble_examples() {
    local dest="$1"
    [[ -d "${ROOT}/examples" ]] || { log "ERROR: missing ${ROOT}/examples"; exit 1; }
    mkdir -p "${dest}"
    cp -a "${ROOT}/examples/." "${dest}/"
    find "${dest}" -depth \( \
            \( -type d \( -name '__pycache__' -o -name 'output' -o -name '.ipynb_checkpoints' \) \) \
            -o \( -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) \) \
        \) -exec rm -rf {} +
    [[ -f "${dest}/quick_start_kws.py" ]] || { log "ERROR: examples/quick_start_kws.py missing"; exit 1; }
    [[ -f "${dest}/onnx_ptq_quick_start.py" ]] || { log "ERROR: examples/onnx_ptq_quick_start.py missing"; exit 1; }
    [[ -f "${dest}/config/mrnn_quantsim_config_custom_mixed_precision_v2.json" ]] \
        || { log "ERROR: examples/config QuantSim JSON missing"; exit 1; }
    [[ -f "${dest}/config/quick_start_full_quant.json" ]] \
        || { log "ERROR: examples/config bitwidth JSON missing"; exit 1; }
    if [[ -e "${dest}/llm_quick_start.py" ]] \
        || [[ -e "${dest}/config/llm_quant.json" ]]; then
        log "ERROR: LLM examples leaked into the bundle"
        exit 1
    fi
    if [[ -e "${dest}/quick_start.py" ]] \
        || [[ -e "${dest}/data/to_band_matrix_erb_240_256.pt" ]] \
        || [[ -e "${dest}/data/inv_to_band_matrix_erb_240_256.pt" ]]; then
        log "ERROR: leftover MRNN example files must not be packaged"
        exit 1
    fi
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
    "${ROOT}/scripts/build_wheels_in_container.sh" \
    "${ROOT}/release/install.sh" \
    "${ROOT}/release/README.md" \
    "${ROOT}/release/ChangeLog.md"
do
    [[ -f "${required}" ]] || { log "ERROR: missing ${required}"; exit 1; }
done

rx_met_require_image_name "${DEV_IMAGE}"
if ! docker image inspect "${DEV_IMAGE}" >/dev/null 2>&1; then
    log "ERROR: missing packaging image ${DEV_IMAGE}"
    log "先构建团队开发和打包环境："
    log "  ./scripts/build_dev_image.sh"
    log "有 ada200_docker:latest 则从它派生；没有则先用 docker/Dockerfile.ada200 造运行时底图。"
    exit 1
fi

export RX_MET_VERSION="${VERSION}"

rm -rf "${WHEEL_EXPORT}"
mkdir -p "${WHEEL_EXPORT}/wheels" "${EXPORT_DIR}" "${STAGING_ROOT}"

docker_run=(
    --rm
    -v "${ROOT}:/opt/rx-met-src:ro"
    -v "${WHEEL_EXPORT}:/artifacts"
    -e "RX_MET_VERSION=${VERSION}"
    -e "RX_MET_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
    -e "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    -e "RX_MET_WHEEL_OUT=/artifacts/wheels"
)
if [[ -n "${RX_MET_BUILD_PROXY:-}" ]]; then
    log "build proxy enabled: ${RX_MET_BUILD_PROXY}"
    docker_run+=(
        --network host
        -e "HTTP_PROXY=${RX_MET_BUILD_PROXY}"
        -e "HTTPS_PROXY=${RX_MET_BUILD_PROXY}"
        -e "http_proxy=${RX_MET_BUILD_PROXY}"
        -e "https_proxy=${RX_MET_BUILD_PROXY}"
        -e "NO_PROXY=localhost,127.0.0.1"
    )
fi

log "build wheels in ${DEV_IMAGE} -> ${WHEEL_EXPORT} (component=${SEMVER}, wheel=${VERSION}, package=${DATE_TAG})"
docker run "${docker_run[@]}" "${DEV_IMAGE}" \
    bash /opt/rx-met-src/scripts/build_wheels_in_container.sh

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
mkdir -p "${RELEASE_DIR}/wheels"

log "assemble bundle -> ${RELEASE_NAME}"
cp -a "${WHEELS_DIR}"/. "${RELEASE_DIR}/wheels/"
rx_met_assemble_examples "${RELEASE_DIR}/examples"
cp "${ROOT}/release/install.sh" "${RELEASE_DIR}/install.sh"
cp "${ROOT}/release/README.md" "${RELEASE_DIR}/README.md"
cp "${ROOT}/release/ChangeLog.md" "${RELEASE_DIR}/ChangeLog.md"
chmod +x "${RELEASE_DIR}/install.sh"
sed -i \
    -e "s/@VERSION@/${VERSION}/g" \
    -e "s/@SEMVER@/${SEMVER}/g" \
    -e "s/@SDK_TAG@/${SDK_TAG}/g" \
    -e "s/@DATE_TAG@/${SDK_TAG}/g" \
    -e "s/@BUNDLE_NAME@/${RELEASE_NAME}/g" \
    "${RELEASE_DIR}/README.md" \
    "${RELEASE_DIR}/ChangeLog.md"

log "compress -> ${RELEASE_BUNDLE}"
rx_met_compress_release_bundle

log "done:"
log "  version: ${SEMVER} (wheel ${VERSION}, package ${DATE_TAG})"
log "  release: ${RELEASE_BUNDLE} ($(du -h "${RELEASE_BUNDLE}" | cut -f1))"
log "  sha256:  ${RELEASE_SHA256}"
