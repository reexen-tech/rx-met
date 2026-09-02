# 团队开发 / 打包图 ada200_docker:rx-met-dev。
# 只在运行时底图上加编译工具，不安装、不替换 torch。
ARG BASE_IMAGE=ada200_docker:latest
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=8

USER root
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
        python3-dev \
        python3-pip \
        libgomp1 \
        libeigen3-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN if command -v nvcc >/dev/null 2>&1; then \
        echo "nvcc already present: $(nvcc --version | tail -1)"; \
    else \
        wget -q -O /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        || wget -q -O /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb; \
        dpkg -i /tmp/cuda-keyring.deb; \
        rm -f /tmp/cuda-keyring.deb; \
        apt-get update && apt-get install -y --no-install-recommends \
            cuda-nvcc-${CUDA_MAJOR}-${CUDA_MINOR} \
            cuda-cudart-dev-${CUDA_MAJOR}-${CUDA_MINOR} \
            cuda-compiler-${CUDA_MAJOR}-${CUDA_MINOR} \
            cuda-cccl-${CUDA_MAJOR}-${CUDA_MINOR} \
            libcublas-dev-${CUDA_MAJOR}-${CUDA_MINOR} \
            libcusparse-dev-${CUDA_MAJOR}-${CUDA_MINOR} \
        && rm -rf /var/lib/apt/lists/*; \
        ln -sfn /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR} /usr/local/cuda; \
    fi

# Official runtime ships matching headers via torch nvidia.* wheels, but nvcc
# only searches the CUDA toolkit include tree. Expose curand/cudnn there.
RUN python3 - <<'PY'
from pathlib import Path

import nvidia

root = Path(nvidia.__path__[0])
cuda_incs = [
    Path("/usr/local/cuda/targets/x86_64-linux/include"),
    Path("/usr/local/cuda/include"),
]
targets = [path for path in cuda_incs if path.is_dir()]
if not targets:
    raise SystemExit("CUDA include directory not found")

for name in ("cudnn", "curand"):
    src = root / name / "include"
    if not src.is_dir():
        raise SystemExit(f"missing torch nvidia header dir: {src}")
    headers = [path for path in src.iterdir() if path.suffix == ".h"]
    if not headers:
        raise SystemExit(f"no .h files in {src}")
    for dest_dir in targets:
        for header in headers:
            dest = dest_dir / header.name
            if dest.exists() or dest.is_symlink():
                continue
            dest.symlink_to(header.resolve())
    print(f"exposed {len(headers)} {name} headers")
PY

RUN if [ -e /usr/local/cuda/lib64/stubs/libcuda.so ] && [ ! -e /usr/local/cuda/lib64/stubs/libcuda.so.1 ]; then \
        ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    fi

RUN python3 -m pip install --no-cache-dir \
        setuptools==70.2.0 \
        wheel==0.45.1 \
        pybind11==2.13.6

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV RX_MET_ENABLE_CUDA=1
ENV RX_MET_CUDA_VERSION=12.8
ENV RX_MET_CUDA_ARCHITECTURES=80;86;89;90;120
ENV TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;12.0
ENV RX_MET_PYTHON=python3

LABEL rx.met.role="dev-builder" \
      rx.met.runtime="ada200_docker"
