# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.0
ARG UBUNTU_VERSION=22.04

# ---------------------------------------------------------------------------
# Builder: CUDA Toolkit, compiler and source exist only in this stage.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG RX_MET_VERSION=1.0.0
ARG CUDA_ARCHITECTURES=80;86;89;120
ARG TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;12.0

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
    && rm -rf /var/lib/apt/lists/*

# The devel image ships libcuda.so but no SONAME symlink because the real
# libcuda.so.1 is normally injected by the host driver at runtime.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install \
        pip==25.2 \
        setuptools==70.2.0 \
        wheel==0.45.1

RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install \
        torch==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /src
COPY . .

ENV RX_MET_VERSION=${RX_MET_VERSION}
ENV RX_MET_CUDA_TAG=cuda12.8
ENV RX_MET_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV RX_MET_NATIVE_OUT=/artifacts/native
ENV RX_MET_WHEEL_OUT=/artifacts/wheels
ENV RX_MET_LLAMA_BUILD_DIR=/build/llama
ENV RX_MET_QUANT_GRU_BUILD_DIR=/build/quant-gru
ENV RX_MET_PYTHON=python3
# GPU-less image builds link CUDA driver symbols against NVIDIA's stub library.
# The stub is never copied into the runtime image; the host driver supplies libcuda.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    ./scripts/package_native.sh
RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    ./scripts/build_wheel.sh
RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    ./scripts/build_quant_gru_wheel.sh

# ---------------------------------------------------------------------------
# Runtime: no nvcc, compiler, CMake, source tree or build intermediates.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG RX_MET_VERSION=1.0.0
ARG RX_MET_REVISION=unknown

LABEL org.opencontainers.image.title="rx-met" \
      org.opencontainers.image.version="${RX_MET_VERSION}" \
      org.opencontainers.image.revision="${RX_MET_REVISION}"

RUN sed -i \
        -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
        -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
        /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements-release.txt /tmp/requirements-release.txt
COPY --from=builder /artifacts/wheels/ /tmp/wheels/
RUN --mount=type=cache,id=rx-met-pip-cu128,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install -r /tmp/requirements-release.txt \
    && python3 -m pip install /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /tmp/requirements-release.txt

COPY --from=builder /artifacts/native/rx-met-native-*.tar.gz /tmp/
RUN mkdir -p /opt/rx-met \
    && tar xf /tmp/rx-met-native-*.tar.gz -C /opt/rx-met \
    && rm /tmp/rx-met-native-*.tar.gz

ENV RX_MET_HOME=/opt/rx-met
ENV PATH="${RX_MET_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${RX_MET_HOME}/lib:${LD_LIBRARY_PATH}"

RUN groupadd --gid 10001 rxmet \
    && useradd --uid 10001 --gid rxmet --create-home rxmet \
    && mkdir -p /work \
    && chown rxmet:rxmet /work

USER 10001:10001
WORKDIR /work
