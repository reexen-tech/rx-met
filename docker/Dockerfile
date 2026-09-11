ARG BUILD_ENV_IMAGE=rx-met-build-env:deps-v1-cu126
ARG RUNTIME_ENV_IMAGE=rx-met-runtime-env:deps-v1-cu126

FROM ${BUILD_ENV_IMAGE} AS project-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CUDA_VARIANT=cu126
ARG RX_MET_CUDA_VERSION=12.6
ARG RX_MET_CUDA_ARCHITECTURES=80;86;89;90
ARG TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0
ARG RX_MET_VERSION=1.0.0
ARG ENV_VERSION=deps-v1

WORKDIR /opt/rx-met-src
COPY . /opt/rx-met-src

RUN grep -Fx "environment_role=build" /opt/rx-met-env/environment.env \
    && grep -Fx "environment_version=${ENV_VERSION}" /opt/rx-met-env/environment.env \
    && grep -Fx "cuda_variant=${CUDA_VARIANT}" /opt/rx-met-env/environment.env \
    && grep -Fx "cuda_version=${RX_MET_CUDA_VERSION}" /opt/rx-met-env/environment.env \
    && [[ "${RX_MET_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]

RUN CUDA_VARIANT="${CUDA_VARIANT}" \
    RX_MET_VERSION="${RX_MET_VERSION}" \
    RX_MET_CUDA_ARCHITECTURES="${RX_MET_CUDA_ARCHITECTURES}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    RX_MET_WHEEL_OUT=/opt/project-wheels \
    RX_MET_BUILD_COPY=/tmp/rx-met-build \
    ./scripts/internal/build_wheels_in_container.sh \
    && cd /opt/project-wheels \
    && sha256sum ./*.whl | sort -k2 > SHA256SUMS


FROM ${RUNTIME_ENV_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CUDA_VARIANT=cu126
ARG RX_MET_CUDA_VERSION=12.6
ARG RX_MET_VERSION=1.0.0
ARG ENV_VERSION=deps-v1
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG MIN_DRIVER_VERSION=560.35.05
ARG SOURCE_REVISION=unknown

ENV CUDA_VARIANT=${CUDA_VARIANT} \
    RX_MET_CUDA_VERSION=${RX_MET_CUDA_VERSION} \
    RX_MET_HOME=/opt/rx-met \
    RX_MET_VERSION=${RX_MET_VERSION} \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

RUN grep -Fx "environment_role=runtime" /opt/rx-met-env/environment.env \
    && grep -Fx "environment_version=${ENV_VERSION}" /opt/rx-met-env/environment.env \
    && grep -Fx "cuda_variant=${CUDA_VARIANT}" /opt/rx-met-env/environment.env \
    && grep -Fx "cuda_version=${RX_MET_CUDA_VERSION}" /opt/rx-met-env/environment.env

COPY --from=project-builder /opt/project-wheels /tmp/project-wheels
COPY --from=project-builder /opt/rx-met-src/docker/requirements /opt/rx-met/requirements
COPY --from=project-builder /opt/rx-met-src/examples /opt/rx-met/examples
COPY --from=project-builder /opt/rx-met-src/release/README.md /opt/rx-met/README.md
COPY --from=project-builder /opt/rx-met-src/release/ChangeLog.md /opt/rx-met/ChangeLog.md

RUN sed -i "s/@VERSION@/${RX_MET_VERSION}/g" \
        /opt/rx-met/README.md /opt/rx-met/ChangeLog.md \
    && cd /tmp/project-wheels \
    && sha256sum --check --strict SHA256SUMS \
    && mkdir -p /opt/rx-met/metadata \
    && cp SHA256SUMS /opt/rx-met/metadata/project-wheel-SHA256SUMS \
    && python3 -m pip install \
        --no-cache-dir \
        --no-index \
        --no-deps \
        /tmp/project-wheels/*.whl \
    && python3 -m pip check \
    && rm -rf /tmp/project-wheels

RUN EXPECTED_TORCH="${TORCH_VERSION}" \
    EXPECTED_TORCHVISION="${TORCHVISION_VERSION}" \
    EXPECTED_TORCHAUDIO="${TORCHAUDIO_VERSION}" \
    EXPECTED_CUDA="${RX_MET_CUDA_VERSION}" \
    python3 - <<'PY'
import importlib.metadata
import os
import runpy

import aimet_onnx
import aimet_torch
import quant_gru
import torch

for package, expected in {
    "torch": os.environ["EXPECTED_TORCH"],
    "torchvision": os.environ["EXPECTED_TORCHVISION"],
    "torchaudio": os.environ["EXPECTED_TORCHAUDIO"],
}.items():
    actual = importlib.metadata.version(package).split("+", 1)[0]
    assert actual == expected, (package, actual, expected)
assert torch.version.cuda == os.environ["EXPECTED_CUDA"], torch.version.cuda
runpy.run_path("/opt/rx-met/examples/quick_start_kws.py", run_name="verify_rx_met")
runpy.run_path("/opt/rx-met/examples/onnx_ptq_quick_start.py", run_name="verify_rx_met")
print("rx-met runtime import check passed", torch.__version__, torch.version.cuda)
PY

RUN printf '%s\n' \
        "rx_met_version=${RX_MET_VERSION}" \
        "environment_version=${ENV_VERSION}" \
        "cuda_variant=${CUDA_VARIANT}" \
        "cuda_version=${RX_MET_CUDA_VERSION}" \
        "torch_version=${TORCH_VERSION}" \
        "torchvision_version=${TORCHVISION_VERSION}" \
        "torchaudio_version=${TORCHAUDIO_VERSION}" \
        "minimum_driver_version=${MIN_DRIVER_VERSION}" \
        "source_revision=${SOURCE_REVISION}" \
        > /opt/rx-met/metadata/image.env \
    && python3 -m pip freeze --all > /opt/rx-met/metadata/pip-freeze.txt \
    && dpkg-query --show > /opt/rx-met/metadata/dpkg-packages.txt

LABEL org.opencontainers.image.vendor="Reexen" \
      org.opencontainers.image.description="rx-met 小模型量化运行环境" \
      org.opencontainers.image.version="${RX_MET_VERSION}" \
      com.reexen.rx-met.environment.version="${ENV_VERSION}" \
      com.reexen.rx-met.cuda.variant="${CUDA_VARIANT}" \
      com.reexen.rx-met.cuda.version="${RX_MET_CUDA_VERSION}" \
      com.reexen.rx-met.driver.minimum="${MIN_DRIVER_VERSION}" \
      com.reexen.rx-met.torch.version="${TORCH_VERSION}"

CMD ["bash"]
