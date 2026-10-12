# Changelog

本文档记录 rx-met 的公开版本和重要兼容性变化。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，正式版本遵循
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- PyTorch 导出增加节点名/encoding key 双向对应及浮点计算输入覆盖检查；双输入计算
  必须覆盖两路浮点张量，标量常量与非浮点输入豁免。即使未启用量化器也不能放过缺失。
- PyTorch 导出末尾强制校验 ONNX 和 encodings；缺少已启用量化器的编码、通道或
  GRU/LSTM 内部参数，以及未校准、非法 scale、编码值不一致或最终图无效时直接报错。

- QuantLSTM 接入 FX/ConnectedGraph、AIMET v2 QuantSim、混合精度、共享校准、Po2、
  参数锁和 ONNX 导出回读；提供 KWS 用户示例与真实 CUDA 集成测试。
- 发布镜像包含 QuantLSTM wheel，增加构建依赖并将环境版本更新为 `deps-v2`。

- 在 `operators/quant-lstm/` 原样收录 QuantLSTM 源码、配置、构建脚本、测试和文档，
  与 QuantGRU 同级维护并支持独立构建。

- 增加独立的环境镜像构建、验证、加载和导出命令。
- 增加 `CONTRIBUTING.md`，统一公开贡献、测试和文档更新要求。

### Changed

- KWS 示例改用 `QuantizableBatchNorm2d`，经 FX 展开后校准 Sub/Div/Mul/Add 的两路
  输入；导出根据参数使用位置恢复被 reshape/常量折叠改名的 ONNX initializer。

- KWS 用户示例通过 `--rnn_type gru|lstm` 选择循环层，共用量化配置，输出按网络类型隔离。
  随机 LSTM 流程检查归入测试，客户说明直接随 examples 交付；发布验收覆盖两种模式。

- 同步 QuantLSTM 上游更新，默认构建 Release CUDA 库并关闭测试和示例，更新使用文档；
  算子目录不再收录上游 `.github/`、`docker/` 和 `.dockerignore`。

- 更新 QuantGRU 使用、配置和计算流程文档，合并 PWL 说明并清理历史算法草稿。
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

- 带偏置的高维 Linear 导出为 Reshape/Gemm/Reshape，复用已校准的融合层编码并恢复
  Gemm 的模块名称，避免 MatMul/Add 拆分造成中间编码缺失；训练侧 QuantSim 保持不变。

- 自定义 BN 使用无偏方差更新 running_var，支持累计平均并保持 buffer 对象不变；
  修复自定义及展开 BN 的 QAT 冻结识别。
- 原始模型和权重检查点移除展开模块残留的辅助参数量化器，支持 weights_only 重载。
- ONNX 简化保留乘一、加零及单输入 Concat，避免删除携带编码的量化边界。

- QuantLSTM signed symmetric 校准保留 `max_abs/qmax` 和零点 0，CPU/CUDA 与 QAT
  运行范围改为完整 INT8/INT16 区间，允许 `-128/-32768`；导出及回读范围与 GRU 一致。

- LSTM 的 rx-met 导出及回读统一使用与 GRU 相同的公共编码字段，移除原生参数备份
  和执行元数据；算子独立的原生检查点格式保持分离。外层输入、输出各保存一条
  `PER_TENSOR` 编码，h/c 状态按方向保留在内部量化点，不再展开为 `PER_CHANNEL` 数组。

- KWS 校准改用无随机增强的训练集；检查点记录网络类型并严格恢复，恢复验证失败时退出。

- 分阶段检查点同时保存和恢复普通层输入编码，避免回读后重复校准改变模型结果。

- 修复 QuantGRU CPU 参考实现引用缺失测试源码导致的 CMake 配置失败。
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
