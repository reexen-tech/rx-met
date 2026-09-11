variable "VERSION" {
  default = "1.0.0"
}

variable "IMAGE_REPOSITORY" {
  default = "rx-met"
}

variable "SOURCE_REVISION" {
  default = "unknown"
}

variable "ENV_VERSION" {
  default = "deps-v1"
}

variable "BUILD_ENV_REPOSITORY" {
  default = "rx-met-build-env"
}

variable "RUNTIME_ENV_REPOSITORY" {
  default = "rx-met-runtime-env"
}

group "default" {
  targets = ["cu118", "cu126", "cu130"]
}

group "environments" {
  targets = [
    "environment-build-cu118",
    "environment-runtime-cu118",
    "environment-build-cu126",
    "environment-runtime-cu126",
    "environment-build-cu130",
    "environment-runtime-cu130",
  ]
}

target "variant-cu118" {
  args = {
    CUDA_VARIANT              = "cu118"
    RX_MET_CUDA_VERSION       = "11.8"
    DEVEL_IMAGE               = "nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04"
    BASE_IMAGE                = "nvcr.io/nvidia/cuda:11.8.0-base-ubuntu22.04"
    RX_MET_CUDA_ARCHITECTURES = "80;86;89;90"
    TORCH_CUDA_ARCH_LIST      = "8.0;8.6;8.9;9.0"
    TORCH_VERSION             = "2.7.1"
    TORCHVISION_VERSION       = "0.22.1"
    TORCHAUDIO_VERSION        = "2.7.1"
    MIN_DRIVER_VERSION        = "520.61.05"
  }
}

target "variant-cu126" {
  args = {
    CUDA_VARIANT              = "cu126"
    RX_MET_CUDA_VERSION       = "12.6"
    DEVEL_IMAGE               = "nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04"
    BASE_IMAGE                = "nvcr.io/nvidia/cuda:12.6.3-base-ubuntu22.04"
    RX_MET_CUDA_ARCHITECTURES = "80;86;89;90"
    TORCH_CUDA_ARCH_LIST      = "8.0;8.6;8.9;9.0"
    TORCH_VERSION             = "2.8.0"
    TORCHVISION_VERSION       = "0.23.0"
    TORCHAUDIO_VERSION        = "2.8.0"
    MIN_DRIVER_VERSION        = "560.35.05"
  }
}

target "variant-cu130" {
  args = {
    CUDA_VARIANT              = "cu130"
    RX_MET_CUDA_VERSION       = "13.0"
    DEVEL_IMAGE               = "nvcr.io/nvidia/cuda:13.0.3-devel-ubuntu22.04"
    BASE_IMAGE                = "nvcr.io/nvidia/cuda:13.0.3-base-ubuntu22.04"
    RX_MET_CUDA_ARCHITECTURES = "80;86;89;90;120"
    TORCH_CUDA_ARCH_LIST      = "8.0;8.6;8.9;9.0;12.0"
    TORCH_VERSION             = "2.10.0"
    TORCHVISION_VERSION       = "0.25.0"
    TORCHAUDIO_VERSION        = "2.10.0"
    MIN_DRIVER_VERSION        = "580.126.20"
  }
}

target "environment-build" {
  context    = "."
  dockerfile = "docker/Dockerfile.environment"
  target     = "build-env"
  platforms  = ["linux/amd64"]
  pull       = false
  args = {
    ENV_VERSION = ENV_VERSION
  }
  labels = {
    "org.opencontainers.image.title"   = "rx-met build environment"
    "org.opencontainers.image.version" = ENV_VERSION
  }
}

target "environment-runtime" {
  context    = "."
  dockerfile = "docker/Dockerfile.environment"
  target     = "runtime-env"
  platforms  = ["linux/amd64"]
  pull       = false
  args = {
    ENV_VERSION = ENV_VERSION
  }
  labels = {
    "org.opencontainers.image.title"   = "rx-met runtime environment"
    "org.opencontainers.image.version" = ENV_VERSION
  }
}

target "environment-build-cu118" {
  inherits = ["environment-build", "variant-cu118"]
  tags     = ["${BUILD_ENV_REPOSITORY}:${ENV_VERSION}-cu118"]
}

target "environment-runtime-cu118" {
  inherits = ["environment-runtime", "variant-cu118"]
  tags     = ["${RUNTIME_ENV_REPOSITORY}:${ENV_VERSION}-cu118"]
}

target "environment-build-cu126" {
  inherits = ["environment-build", "variant-cu126"]
  tags     = ["${BUILD_ENV_REPOSITORY}:${ENV_VERSION}-cu126"]
}

target "environment-runtime-cu126" {
  inherits = ["environment-runtime", "variant-cu126"]
  tags     = ["${RUNTIME_ENV_REPOSITORY}:${ENV_VERSION}-cu126"]
}

target "environment-build-cu130" {
  inherits = ["environment-build", "variant-cu130"]
  tags     = ["${BUILD_ENV_REPOSITORY}:${ENV_VERSION}-cu130"]
}

target "environment-runtime-cu130" {
  inherits = ["environment-runtime", "variant-cu130"]
  tags     = ["${RUNTIME_ENV_REPOSITORY}:${ENV_VERSION}-cu130"]
}

target "release" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  target     = "runtime"
  platforms  = ["linux/amd64"]
  pull       = false
  args = {
    ENV_VERSION      = ENV_VERSION
    RX_MET_VERSION   = VERSION
    SOURCE_REVISION  = SOURCE_REVISION
  }
  labels = {
    "org.opencontainers.image.title"    = "rx-met"
    "org.opencontainers.image.version"  = VERSION
    "org.opencontainers.image.revision" = SOURCE_REVISION
  }
}

target "cu118" {
  inherits = ["release", "variant-cu118"]
  tags     = ["${IMAGE_REPOSITORY}:${VERSION}-cu118"]
  args = {
    BUILD_ENV_IMAGE   = "${BUILD_ENV_REPOSITORY}:${ENV_VERSION}-cu118"
    RUNTIME_ENV_IMAGE = "${RUNTIME_ENV_REPOSITORY}:${ENV_VERSION}-cu118"
  }
}

target "cu126" {
  inherits = ["release", "variant-cu126"]
  tags     = ["${IMAGE_REPOSITORY}:${VERSION}-cu126"]
  args = {
    BUILD_ENV_IMAGE   = "${BUILD_ENV_REPOSITORY}:${ENV_VERSION}-cu126"
    RUNTIME_ENV_IMAGE = "${RUNTIME_ENV_REPOSITORY}:${ENV_VERSION}-cu126"
  }
}

target "cu130" {
  inherits = ["release", "variant-cu130"]
  tags     = ["${IMAGE_REPOSITORY}:${VERSION}-cu130"]
  args = {
    BUILD_ENV_IMAGE   = "${BUILD_ENV_REPOSITORY}:${ENV_VERSION}-cu130"
    RUNTIME_ENV_IMAGE = "${RUNTIME_ENV_REPOSITORY}:${ENV_VERSION}-cu130"
  }
}
