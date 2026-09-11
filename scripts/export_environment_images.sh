#!/usr/bin/env bash
# 将环境镜像导出到本地目录；复制或移动到共享盘由维护人手工完成。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/lib/environment_images.sh"
OUTPUT_DIR="${ROOT}/.release/environment-images/${RX_MET_ENV_VERSION}"
ZSTD_THREADS="${RX_MET_CACHE_ZSTD_THREADS:-2}"
FORCE=0
TARGETS=()
STAGING_DIR=""

log() { printf '[export_environment_images] %s\n' "$*"; }
die() { printf '[export_environment_images] ERROR: %s\n' "$*" >&2; exit 1; }
cleanup() {
    [[ -z "${STAGING_DIR}" || ! -d "${STAGING_DIR}" ]] \
        || rm -rf -- "${STAGING_DIR}"
}
trap cleanup EXIT

usage() {
    cat <<'EOF'
用法: ./scripts/export_environment_images.sh [选项] [cu118|cu126|cu130 ...]

选项：
  --output-dir DIR    本地输出目录，默认 .release/environment-images/<环境版本>
  --force             全部新文件生成并校验后，替换本地同名文件
  -h, --help          显示帮助

脚本只生成可搬运文件，不写 /mnt/data2。共享盘复制或移动由维护人手工执行。
EOF
}

while (($#)); do
    case "$1" in
        --output-dir)
            (($# >= 2)) || die "--output-dir 缺少参数"
            OUTPUT_DIR="$2"
            shift
            ;;
        --force) FORCE=1 ;;
        -h|--help) usage; exit 0 ;;
        cu118|cu126|cu130) TARGETS+=("$1") ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done
if ((${#TARGETS[@]} == 0)); then
    TARGETS=(cu118 cu126 cu130)
fi
rx_met_validate_environment_config || die "环境镜像名称或版本无效"
[[ "${ZSTD_THREADS}" =~ ^[1-8]$ ]] \
    || die "RX_MET_CACHE_ZSTD_THREADS 必须是 1 到 8"
for command in docker realpath sha256sum zstd; do
    command -v "${command}" >/dev/null 2>&1 || die "缺少命令: ${command}"
done
OUTPUT_DIR="$(realpath -m -- "${OUTPUT_DIR}")"
case "${OUTPUT_DIR}" in
    /mnt/data2|/mnt/data2/*)
        die "导出脚本禁止写共享盘，请先输出到本地再人工复制: ${OUTPUT_DIR}"
        ;;
esac
mkdir -p "${OUTPUT_DIR}"
[[ -w "${OUTPUT_DIR}" ]] || die "输出目录不可写: ${OUTPUT_DIR}"

declare -A seen=()
archives=()
images=()
for target in "${TARGETS[@]}"; do
    [[ -z "${seen[${target}]:-}" ]] || die "目标重复: ${target}"
    seen["${target}"]=1
    build_image="$(rx_met_build_environment_image "${target}")"
    runtime_image="$(rx_met_runtime_environment_image "${target}")"
    for image in "${build_image}" "${runtime_image}"; do
        docker image inspect "${image}" >/dev/null 2>&1 \
            || die "环境镜像不存在: ${image}"
    done
    archive="$(rx_met_environment_archive_name "${target}")"
    if [[ -e "${OUTPUT_DIR}/${archive}" && "${FORCE}" == "0" ]]; then
        die "本地归档已存在，拒绝覆盖: ${OUTPUT_DIR}/${archive}"
    fi
    archives+=("${archive}")
    images+=("${build_image}" "${runtime_image}")
done
for metadata in README.md environment-manifest.json SHA256SUMS; do
    if [[ -e "${OUTPUT_DIR}/${metadata}" && "${FORCE}" == "0" ]]; then
        die "本地元数据已存在，拒绝覆盖: ${OUTPUT_DIR}/${metadata}"
    fi
done

STAGING_DIR="$(mktemp -d -p "${OUTPUT_DIR}" .rx-met-env-export.XXXXXX)"
for target in "${TARGETS[@]}"; do
    build_image="$(rx_met_build_environment_image "${target}")"
    runtime_image="$(rx_met_runtime_environment_image "${target}")"
    archive="$(rx_met_environment_archive_name "${target}")"
    log "导出 ${build_image} + ${runtime_image} -> ${archive}"
    docker save "${build_image}" "${runtime_image}" \
        | zstd -T"${ZSTD_THREADS}" -6 -o "${STAGING_DIR}/${archive}"
    zstd --test "${STAGING_DIR}/${archive}"
done

docker image inspect "${images[@]}" \
    > "${STAGING_DIR}/environment-manifest.json"
{
    printf '# rx-met 稳定环境镜像本地导出\n\n'
    printf -- '- 环境版本：`%s`\n' "${RX_MET_ENV_VERSION}"
    printf -- '- 平台：Linux x86_64\n'
    printf -- '- 内容：每个 CUDA 变体一个文件，包含 build-env 和 runtime-env 两个 tag\n'
    printf -- '- 生成时间：`%s`\n\n' "$(date --iso-8601=seconds)"
    printf '把本目录全部文件人工复制到目标位置；消费者使用：\n\n```bash\n'
    printf './scripts/load_environment_images.sh --archive-dir /path/to/%s cu118 cu126 cu130\n' \
        "${RX_MET_ENV_VERSION}"
    printf '```\n'
} > "${STAGING_DIR}/README.md"
(
    cd "${STAGING_DIR}"
    checksum_files=(README.md environment-manifest.json)
    checksum_files+=("${archives[@]}")
    sha256sum "${checksum_files[@]}" | sort -k2 > SHA256SUMS
    sha256sum --check --strict SHA256SUMS
)

for archive in "${archives[@]}"; do
    mv -f -- "${STAGING_DIR}/${archive}" "${OUTPUT_DIR}/${archive}"
done
for metadata in README.md environment-manifest.json SHA256SUMS; do
    mv -f -- "${STAGING_DIR}/${metadata}" "${OUTPUT_DIR}/${metadata}"
done
rmdir -- "${STAGING_DIR}"
STAGING_DIR=""
log "本地可搬运文件已生成: ${OUTPUT_DIR}"
