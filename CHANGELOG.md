# Changelog

本文档记录 rx-met 的公开版本和重要兼容性变化。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，正式版本遵循
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 增加独立的环境镜像构建、验证、加载和导出命令。
- 增加 `CONTRIBUTING.md`，统一公开贡献、测试和文档更新要求。

### Changed

- 将 AIMET Python 包整理到 `src/`，将 AIMET 原生实现放在 `native/`，并将
  QuantGRU 放在 `operators/quant-gru/`。安装后继续沿用现有 Python import 名和
  wheel 名。
- 将发布制品模板从顶层 `release/` 迁移到 `packaging/release-bundle/`，使用
  `.md.in` 标识待渲染输入。
- 将公开命令按职责整理到 `scripts/release/` 和 `scripts/environment/`。
- 由 `pyproject.toml` 定义 Python 兼容范围，由 `docker/variants.json` 统一管理
  CUDA 变体和精确依赖；Bake 配置和 lock 文件由 `scripts/dependencies.py` 生成。
- 发布形式从 wheel 软件包改为包含完整运行环境的 rx-met Docker 镜像。
- Docker 构建拆分为可复用的 `build-env`、`runtime-env` 和项目发布镜像。
- 文档按用户指南、配置说明、维护者文档和架构决策重新组织。

### Fixed

- 修复 ONNX PTQ 快速示例读取错误 `rxmet_encodings` 字段的问题。

## [1.0.0] - 2026-09-02

rx-met 的首个公开版本。

### Added

- 提供 `aimet_common`、`aimet_torch` 和 `aimet_onnx` 量化工具包。
- 提供可替换 PyTorch GRU 的 QuantGRU 算子，支持 PTQ、QAT 和 ONNX 导出。
- 提供 Speech Commands KWS 的 PyTorch 量化示例。
- 提供已有 ONNX 模型的 PTQ 示例，输出 Clean ONNX、AIMET 原始 encodings、
  编译器 encodings 和运行元数据。
- 提供混合精度配置、Power-of-2 scale 和 `Sb = Sx * Sw` bias 对齐流程。

### Fixed

- 首次公开发行，修复项从后续版本开始记录。

### Compatibility

- 支持 Linux x86_64 和 NVIDIA GPU。
- 发布镜像提供 `cu118`、`cu126` 和 `cu130` 变体；精确依赖和驱动要求见
  `docker/variants.json` 及 `docs/User_guide.md`。
- Python 包支持 CPython 3.10 至 3.12，具体版本由镜像变体决定。

### Known limitations

- 模型、数据集和校准样本由用户单独准备。
- ONNX 直量化流程适用于 PTQ；QAT 使用 PyTorch 流程。
- 正式 GPU 制品在目标驱动环境验证；CPU 模式用于调试。

### Upgrade notes

- 该版本是公开安装和升级的起始版本。
