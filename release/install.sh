#!/usr/bin/env bash
# Install rx-met wheels into the current Python (ada200_docker system interpreter).
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS="${PKG_DIR}/wheels"
PYTHON="${RX_MET_PYTHON:-python3}"
SKIP_CHECK="${RX_MET_SKIP_ENV_CHECK:-0}"

log() { printf '[install] %s\n' "$*"; }
die() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "${WHEELS}" ]] || die "wheels/ not found: ${WHEELS}"

if [[ "${SKIP_CHECK}" != "1" ]]; then
    "${PYTHON}" - <<'PY'
import platform
import sys

if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"need CPython 3.10, found {sys.implementation.name} {sys.version_info.major}.{sys.version_info.minor}"
    )
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit(f"need Linux x86_64, found {platform.system()} {platform.machine()}")

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is not installed; run this inside ada200_docker") from exc

ver = torch.__version__.split("+", 1)[0]
if ver != "2.8.0":
    raise SystemExit(f"need torch==2.8.0, found {torch.__version__}")
if getattr(torch.version, "cuda", None) != "12.8":
    raise SystemExit(f"need torch.version.cuda == 12.8, found {torch.version.cuda}")
print("env check ok", "python", sys.version.split()[0], "torch", torch.__version__)
PY
else
    log "RX_MET_SKIP_ENV_CHECK=1, skipping environment check"
fi

shopt -s nullglob
rx_met_wheels=("${WHEELS}"/rx_met-*.whl)
quant_gru_wheels=("${WHEELS}"/quant_gru-*.whl)
shopt -u nullglob
[[ ${#rx_met_wheels[@]} -eq 1 ]] || die "expected exactly one rx_met-*.whl in ${WHEELS}"
[[ ${#quant_gru_wheels[@]} -eq 1 ]] || die "expected exactly one quant_gru-*.whl in ${WHEELS}"

log "uninstall official AIMET if present"
"${PYTHON}" -m pip uninstall -y aimet-torch aimet-onnx aimet-common >/dev/null 2>&1 || true

log "install rx-met + QuantGRU"
"${PYTHON}" -m pip install --no-index --no-deps \
    "${rx_met_wheels[0]}" \
    "${quant_gru_wheels[0]}"

extras=()
shopt -s nullglob
for wheel in "${WHEELS}"/*.whl; do
    base="$(basename "${wheel}")"
    case "${base}" in
        rx_met-*|quant_gru-*|torch-*|torchvision-*|nvidia-*|numpy-*|scipy-*|scikit_learn-*|scikit-learn-*|joblib-*|threadpoolctl-*|onnx-*|protobuf-*|ml_dtypes-*)
            continue
            ;;
    esac
    extras+=("${wheel}")
done
shopt -u nullglob

if ((${#extras[@]} > 0)); then
    log "install extra wheels (${#extras[@]} files)"
    "${PYTHON}" -m pip install --no-index --no-deps "${extras[@]}"
fi

log "register NVIDIA pip CUDA runtime (AIMET libpymo needs libcudart.so.12)"
CUDA_ENV="${PKG_DIR}/cuda_libs.env"
"${PYTHON}" - <<'PY' > "${CUDA_ENV}"
import glob
import os

dirs = sorted(glob.glob("/usr/local/lib/python3.10/dist-packages/nvidia/*/lib"))
torch_lib = "/usr/local/lib/python3.10/dist-packages/torch/lib"
if os.path.isdir(torch_lib):
    dirs.append(torch_lib)
print("export LD_LIBRARY_PATH=\"%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"" % ":".join(dirs))
PY
if [[ -w /etc/ld.so.conf.d ]]; then
    "${PYTHON}" - <<'PY' > /etc/ld.so.conf.d/rx-met-nvidia.conf
import glob
import os

dirs = sorted(glob.glob("/usr/local/lib/python3.10/dist-packages/nvidia/*/lib"))
torch_lib = "/usr/local/lib/python3.10/dist-packages/torch/lib"
if os.path.isdir(torch_lib):
    dirs.append(torch_lib)
print("\n".join(dirs))
PY
    ldconfig || log "ldconfig failed; source ${CUDA_ENV} before running examples"
else
    log "cannot write /etc/ld.so.conf.d; source ${CUDA_ENV} before running examples"
fi

log "verify imports"
"${PYTHON}" - <<'PY'
import importlib
import sys

for name in ("aimet_torch", "aimet_onnx", "quant_gru", "torchaudio", "librosa", "soundfile", "onnxsim"):
    importlib.import_module(name)
    print("import ok", name)

try:
    import rx_met_llm
except ImportError:
    print("rx_met_llm absent (expected)")
else:
    raise SystemExit("rx_met_llm must not be installed")

print("install ok")
PY

cat <<EOF

安装完成。

下一步：
  1. 将数据集挂到容器 /datasets
       小模型:  /datasets/speech_commands_v0.02
       ONNX:    /datasets/yolo-fastest  或  /datasets/watchhar
  2. cd ${PKG_DIR}/examples
  3. python3 quick_start_kws.py
     python3 onnx_ptq_quick_start.py

EOF
