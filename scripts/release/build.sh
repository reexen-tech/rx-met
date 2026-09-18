#!/usr/bin/env bash
# 复用稳定环境镜像，构建、验收并导出 rx-met 产品镜像。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BAKE_FILE="${ROOT}/docker/docker-bake.hcl"
VERSION_FILE="${ROOT}/VERSION"
source "${ROOT}/scripts/lib/environment_images.sh"

IMAGE_REPOSITORY="${RX_MET_IMAGE_REPOSITORY:-rx-met}"
EXPORT_DIR="${RX_MET_EXPORT_DIR:-${ROOT}/.release/export}"
EXPORT_IMAGES="${RX_MET_EXPORT_IMAGES:-1}"
ZSTD_THREADS="${RX_MET_ZSTD_THREADS:-2}"
BUILDER="${RX_MET_BUILDER:-}"
AUTO_ENVIRONMENT="${RX_MET_AUTO_BUILD_ENVIRONMENT:-1}"
REBUILD_ENVIRONMENT=0
ENV_ARCHIVE_DIR="${RX_MET_ENV_ARCHIVE_DIR:-}"
ENV_ARCHIVES=()
TARGETS=()
EXAMPLES_CONTAINER=""
EXAMPLES_STAGING=""

log() { printf '[release_build] %s\n' "$*"; }
die() { printf '[release_build] ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [[ -n "${EXAMPLES_CONTAINER}" ]]; then
        docker rm -f "${EXAMPLES_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${EXAMPLES_STAGING}" ]]; then
        rm -rf -- "${EXAMPLES_STAGING}"
    fi
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
用法: ./scripts/release/build.sh [选项] [cu118|cu126|cu130 ...]

环境镜像选项：
  --environment-dir DIR
                      从 DIR 校验并加载标准命名的环境镜像归档
  --environment-archive FILE
                      指定一个 build-env+runtime-env 归档；可重复使用
  --rebuild-environment
                      构建前强制成对重建所选环境镜像
  --no-auto-environment
                      环境镜像缺失且未提供归档时直接退出

产品选项：
  --no-export         构建并验收，但不导出产品镜像 tar.zst
  -h, --help          显示帮助

默认构建全部三个 CUDA 变体。环境镜像已有则复用；没有归档时自动构建。
EOF
}

while (($#)); do
    case "$1" in
        --environment-dir)
            (($# >= 2)) || die "--environment-dir 缺少参数"
            ENV_ARCHIVE_DIR="$2"
            shift
            ;;
        --environment-archive)
            (($# >= 2)) || die "--environment-archive 缺少参数"
            ENV_ARCHIVES+=("$2")
            shift
            ;;
        --rebuild-environment) REBUILD_ENVIRONMENT=1 ;;
        --no-auto-environment) AUTO_ENVIRONMENT=0 ;;
        --no-export) EXPORT_IMAGES=0 ;;
        -h|--help) usage; exit 0 ;;
        cu118|cu126|cu130) TARGETS+=("$1") ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done
if ((${#TARGETS[@]} == 0)); then
    TARGETS=(cu118 cu126 cu130)
fi

declare -A seen=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen[${target}]:-}" ]] || die "目标重复: ${target}"
    seen["${target}"]=1
done
rx_met_validate_environment_config || die "环境镜像名称或版本无效"
[[ "${IMAGE_REPOSITORY}" =~ ^[A-Za-z0-9._/-]+$ ]] \
    || die "无效的产品镜像仓库名: ${IMAGE_REPOSITORY}"
[[ "${EXPORT_IMAGES}" == "0" || "${EXPORT_IMAGES}" == "1" ]] \
    || die "RX_MET_EXPORT_IMAGES 必须是 0 或 1"
[[ "${AUTO_ENVIRONMENT}" == "0" || "${AUTO_ENVIRONMENT}" == "1" ]] \
    || die "RX_MET_AUTO_BUILD_ENVIRONMENT 必须是 0 或 1"
[[ "${ZSTD_THREADS}" =~ ^[1-8]$ ]] \
    || die "RX_MET_ZSTD_THREADS 必须是 1 到 8"
for command in docker git; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done
if ((EXPORT_IMAGES)); then
    command -v zstd >/dev/null 2>&1 || die "导出镜像需要 zstd"
fi
[[ -f "${VERSION_FILE}" ]] || die "缺少 ${VERSION_FILE}"
VERSION="$(tr -d '[:space:]' < "${VERSION_FILE}")"
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "VERSION 必须是 MAJOR.MINOR.PATCH，实际为 ${VERSION}"
docker info >/dev/null 2>&1 || die "无法访问 Docker daemon"
rx_met_require_docker_builder "${BUILDER}" || die "请切换到 default docker builder"

SOURCE_REVISION="$(git -C "${ROOT}" rev-parse HEAD)"
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    SOURCE_REVISION="${SOURCE_REVISION}-dirty"
fi
export VERSION IMAGE_REPOSITORY SOURCE_REVISION
export ENV_VERSION="${RX_MET_ENV_VERSION}"
export BUILD_ENV_REPOSITORY="${RX_MET_BUILD_ENV_REPOSITORY}"
export RUNTIME_ENV_REPOSITORY="${RX_MET_RUNTIME_ENV_REPOSITORY}"

log "产品版本: ${VERSION}，环境版本: ${RX_MET_ENV_VERSION}"
log "构建目标: ${TARGETS[*]}"

missing_targets=()
for target in "${TARGETS[@]}"; do
    build_image="$(rx_met_build_environment_image "${target}")"
    runtime_image="$(rx_met_runtime_environment_image "${target}")"
    build_exists=0
    runtime_exists=0
    docker image inspect "${build_image}" >/dev/null 2>&1 && build_exists=1
    docker image inspect "${runtime_image}" >/dev/null 2>&1 && runtime_exists=1
    if ((REBUILD_ENVIRONMENT)); then
        missing_targets+=("${target}")
    elif ((build_exists && runtime_exists)); then
        log "复用本机环境镜像: ${target}"
    elif ((build_exists != runtime_exists)); then
        die "${target} 只存在一半环境镜像；请使用 --rebuild-environment 成对重建"
    else
        missing_targets+=("${target}")
    fi
done

if ((REBUILD_ENVIRONMENT)); then
    environment_args=(--force --no-verify --no-export)
    [[ -z "${BUILDER}" ]] || environment_args+=(--builder "${BUILDER}")
    "${ROOT}/scripts/environment/build.sh" \
        "${environment_args[@]}" "${TARGETS[@]}"
elif ((${#missing_targets[@]})); then
    if [[ -n "${ENV_ARCHIVE_DIR}" || ${#ENV_ARCHIVES[@]} -gt 0 ]]; then
        load_args=(--no-verify)
        [[ -z "${ENV_ARCHIVE_DIR}" ]] || load_args+=(--archive-dir "${ENV_ARCHIVE_DIR}")
        for archive in "${ENV_ARCHIVES[@]}"; do
            load_args+=(--archive "${archive}")
        done
        log "从归档加载缺失环境: ${missing_targets[*]}"
        "${ROOT}/scripts/environment/load.sh" \
            "${load_args[@]}" "${missing_targets[@]}"
    elif ((AUTO_ENVIRONMENT)); then
        environment_args=(--no-verify --no-export)
        [[ -z "${BUILDER}" ]] || environment_args+=(--builder "${BUILDER}")
        log "没有可用归档，自动构建缺失环境: ${missing_targets[*]}"
        "${ROOT}/scripts/environment/build.sh" \
            "${environment_args[@]}" "${missing_targets[@]}"
    else
        die "环境镜像缺失: ${missing_targets[*]}；请提供归档或运行 scripts/environment/build.sh"
    fi
fi

"${ROOT}/scripts/environment/verify.sh" "${TARGETS[@]}"

buildx_args=()
if [[ -n "${BUILDER}" ]]; then
    buildx_args+=(--builder "${BUILDER}")
fi
proxy_count=0
for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
    http_proxy https_proxy all_proxy no_proxy; do
    proxy_value="${!proxy_var:-}"
    if [[ -n "${proxy_value}" ]]; then
        buildx_args+=(--set "*.args.${proxy_var}=${proxy_value}")
        ((proxy_count += 1))
    fi
done
((proxy_count == 0)) || log "向本次构建传递 ${proxy_count} 个代理变量"

for target in "${TARGETS[@]}"; do
    log "构建产品镜像: ${target}"
    (
        cd "${ROOT}"
        docker buildx bake \
            "${buildx_args[@]}" \
            -f "${BAKE_FILE}" \
            --load \
            --progress="${RX_MET_BUILD_PROGRESS:-plain}" \
            "${target}"
    )
done

log "执行产品镜像无 GPU 验收"
RX_MET_VERIFY_GPU=0 "${ROOT}/scripts/release/verify_bundle.sh" "${TARGETS[@]}"

if ((EXPORT_IMAGES == 0)); then
    log "已按要求跳过产品镜像归档"
    exit 0
fi

mkdir -p "${EXPORT_DIR}"
archives=()
images=()
for target in "${TARGETS[@]}"; do
    images+=("${IMAGE_REPOSITORY}:${VERSION}-${target}")
done

examples_output="${EXPORT_DIR}/examples"
EXAMPLES_STAGING="$(mktemp -d "${EXPORT_DIR}/.rx-met-examples.XXXXXX")"
EXAMPLES_CONTAINER="$(docker create "${images[0]}")"
docker cp "${EXAMPLES_CONTAINER}:/opt/rx-met/examples/." "${EXAMPLES_STAGING}"
docker rm "${EXAMPLES_CONTAINER}" >/dev/null
EXAMPLES_CONTAINER=""
for required_example in README.md quick_start_kws.py onnx_ptq_quick_start.py \
    prepare_onnx_ptq_data.py config/mrnn_quantsim_config_custom_mixed_precision_v2.json \
    config/quick_start_full_quant.json; do
    [[ -f "${EXAMPLES_STAGING}/${required_example}" ]] \
        || die "产品镜像缺少示例文件: ${required_example}"
done
rm -rf -- "${examples_output}"
mv -- "${EXAMPLES_STAGING}" "${examples_output}"
EXAMPLES_STAGING=""
log "导出示例源码: ${examples_output}"

for target in "${TARGETS[@]}"; do
    image="${IMAGE_REPOSITORY}:${VERSION}-${target}"
    archive="rx-met-v${VERSION}-${target}-linux-amd64.tar.zst"
    output="${EXPORT_DIR}/${archive}"
    log "导出 ${image} -> ${output}"
    docker save "${image}" \
        | zstd -T"${ZSTD_THREADS}" -10 -o "${output}.partial"
    mv -f -- "${output}.partial" "${output}"
    archives+=("${archive}")
done

manifest="${EXPORT_DIR}/image-manifest.json"
docker image inspect "${images[@]}" > "${manifest}.partial"
mv -f -- "${manifest}.partial" "${manifest}"
cp "${ROOT}/packaging/release-bundle/README.md.in" "${EXPORT_DIR}/README.md"
cp "${ROOT}/packaging/release-bundle/CHANGELOG.md.in" "${EXPORT_DIR}/ChangeLog.md"
sed -i "s/@VERSION@/${VERSION}/g" \
    "${EXPORT_DIR}/README.md" "${EXPORT_DIR}/ChangeLog.md"
(
    cd "${EXPORT_DIR}"
    checksum_args=("ChangeLog.md" "README.md" "image-manifest.json")
    checksum_args+=("${archives[@]}")
    mapfile -d '' -t example_files < <(find examples -type f -print0 | sort -z)
    ((${#example_files[@]} > 0)) || die "导出的 examples 目录为空"
    checksum_args+=("${example_files[@]}")
    sha256sum "${checksum_args[@]}" | sort -k2 > SHA256SUMS.partial
    mv -f SHA256SUMS.partial SHA256SUMS
)

log "构建完成，产品镜像归档位于 ${EXPORT_DIR}"
log "发布前必须在目标 GPU 执行: ./scripts/release/verify_bundle.sh --gpu ${TARGETS[*]}"
