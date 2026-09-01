# Changelog

本文档记录本仓库所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

AIMET 相关改动见 [`aimet_CHANGELOG.md`](aimet_CHANGELOG.md)。

## [Unreleased]

### 变更

- 仓库改为独立小模型仓；大模型量化不在本仓库。客户交付改为 `ada200_docker` 上的 wheel 软件包（`vYYMMDD`）。

### ✨ 新增功能

- **计算类算子默认打开输入/输出量化**（见 `aimet_CHANGELOG.md`）
  - JSON 1 不再需要为 Conv/Add/Relu 等逐条写 `is_input_quantized`
  - 要禁用某个计算类，继续用 JSON 2 的 `disable_quantization`
