ARG BUILD_ENV_IMAGE=scratch

FROM ${BUILD_ENV_IMAGE} AS project-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CUDA_VARIANT
ARG RX_MET_CUDA_VERSION
ARG RX_MET_CUDA_ARCHITECTURES
ARG TORCH_CUDA_ARCH_LIST
ARG RX_MET_VERSION
ARG ENV_VERSION
ARG JOBS=8
ENV RX_MET_ONNXRUNTIME_ROOT=/opt/rx-quant-env/onnxruntime
ENV JOBS=${JOBS} MAX_JOBS=${JOBS}

WORKDIR /opt/rx-met-src
COPY . /opt/rx-met-src

RUN grep -Fx "environment_role=build" /opt/rx-quant-env/environment.env \
    && grep -Fx "environment_version=${ENV_VERSION}" /opt/rx-quant-env/environment.env \
    && grep -Fx "cuda_variant=${CUDA_VARIANT}" /opt/rx-quant-env/environment.env \
    && grep -Fx "cuda_version=${RX_MET_CUDA_VERSION}" /opt/rx-quant-env/environment.env \
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


FROM scratch AS wheels
COPY --from=project-builder /opt/project-wheels /wheels
