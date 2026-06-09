# shellcheck shell=bash
# REEX 验证脚本共用：在 PATH 中找不到 nvcc 时，尝试使用标准 CUDA 安装路径。
# 用法: source "$(dirname "$0")/reex_cuda_env.sh" && reex_prepend_cuda_path
# 或设 SKIP_CUDA=1 跳过本逻辑。

reex_prepend_cuda_path() {
  if [[ "${SKIP_CUDA:-0}" == "1" ]]; then
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi
  local root="${CUDA_HOME:-/usr/local/cuda}"
  if [[ ! -x "${root}/bin/nvcc" ]]; then
    # 常见多版本安装
    local alt
    alt="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
    if [[ -x "$alt" ]]; then
      root="$(cd "$(dirname "$alt")/.." && pwd)"
    fi
  fi
  if [[ -x "${root}/bin/nvcc" ]]; then
    export CUDA_HOME="$root"
    export PATH="${root}/bin:${PATH}"
    if [[ -d "${root}/lib64" ]]; then
      export LD_LIBRARY_PATH="${root}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    return 0
  fi
  return 1
}
