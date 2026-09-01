# syntax=docker/dockerfile:1.7
# Builder-only image: AIMET native + quant_gru + extra wheels. No customer runtime.

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RX_MET_VERSION
ARG CUDA_ARCHITECTURES=80;86;89;90;120
ARG TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;12.0

RUN test -n "${RX_MET_VERSION}"

RUN sed -i \
        -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
        -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
        /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        libgomp1 \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN --mount=type=cache,id=rx-met-pip-cu128-torch28,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install \
        pip==25.2 \
        setuptools==70.2.0 \
        wheel==0.45.1

COPY requirements-build.txt /tmp/requirements-build.txt
RUN --mount=type=cache,id=rx-met-pip-cu128-torch28,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install -r /tmp/requirements-build.txt

RUN --mount=type=cache,id=rx-met-pip-cu128-torch28,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install \
        torch==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /src
COPY . .

RUN python3 scripts/prepare_packaging.py --root /src --version "${RX_MET_VERSION}"

ENV RX_MET_VERSION=${RX_MET_VERSION}
ENV RX_MET_ENABLE_CUDA=1
ENV RX_MET_CUDA_VERSION=12.8
ENV RX_MET_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV RX_MET_WHEEL_OUT=/artifacts/wheels
ENV RX_MET_QUANT_GRU_BUILD_DIR=/build/quant-gru
ENV RX_MET_AIMET_BUILD_DIR=/build/aimet
ENV RX_MET_PYTHON=python3
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

RUN --mount=type=cache,id=rx-met-pip-cu128-torch28,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,id=rx-met-aimet-cu128-torch28,target=/build/aimet,sharing=locked \
    ./scripts/build_wheel.sh
RUN --mount=type=cache,id=rx-met-pip-cu128-torch28,target=/root/.cache/pip,sharing=locked \
    ./scripts/build_quant_gru_wheel.sh
RUN --mount=type=cache,id=rx-met-pip-cu128-torch28,target=/root/.cache/pip,sharing=locked \
    ./scripts/download_extra_wheels.sh /artifacts/wheels

RUN python3 - <<'PY'
from pathlib import Path
import zipfile
wheels = list(Path("/artifacts/wheels").glob("rx_met-*.whl"))
assert len(wheels) == 1, wheels
names = zipfile.ZipFile(wheels[0]).namelist()
assert not any(name.startswith("rx_met_llm") for name in names), "rx_met_llm leaked into wheel"
print("wheel ok", wheels[0].name)
assert list(Path("/artifacts/wheels").glob("quant_gru-*.whl")), "missing quant_gru wheel"
assert list(Path("/artifacts/wheels").glob("torchaudio-*.whl")), "missing torchaudio wheel"
assert list(Path("/artifacts/wheels").glob("librosa-*.whl")), "missing librosa wheel"
assert list(Path("/artifacts/wheels").glob("onnxsim-*.whl")), "missing onnxsim wheel"
print("artifact wheels", sorted(p.name for p in Path("/artifacts/wheels").glob("*.whl")))
PY

FROM scratch AS export
COPY --from=builder /artifacts/wheels /wheels
