# Changelog

本文档记录 rx-met 的公开版本和重要兼容性变化。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，正式版本遵循
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed

- 阶段编码重载恢复保存的整数网格、对称性及精确 scale/offset，防止 Po2 的 32 位 bias 被按新模型的 16 位网格恢复；拒绝无效形状、数值和不一致的编码。

- 修复 GRU `(1-z)` 重复减去更新门 zero point 的错误，覆盖主整数核、浮点参考和独立 CPU 参考；默认 zero point 为零的公式不变。

- QuantGRU 的 FP32/INT32 carrier 共用整数/LUT 前向与截断 mask；锁定前完成待处理的网格转换，公共编码重载恢复 Po2 执行模式，消除 QAT、整数评估和重载之间的数值分歧。

- PyTorch Po2 工作流遵守 `bias_bitwidth`，在 Conv/Linear bias 对齐到 `Sx*Sw` 前检查范围，并拒绝静默饱和。
- QuantGRU/QuantLSTM 在校准时为固定 bias 网格保留相对与最小绝对训练余量，防止零偏置的微小 QAT 更新立即饱和；保留导入编码。
- QuantGRU 输入线性截断 mask 在 bias 累加与跨 CUDA stream 同步之前应用，统一输入、权重和偏置梯度。
- QuantGRU QAT 将候选门输入/输出截断 mask 先传到 reset gate 上游，修正截断时泄漏的梯度。

## [1.1.0] - 2026-10-12

### Added

- QuantLSTM 接入 FX/ConnectedGraph、AIMET v2 QuantSim、混合精度、共享校准、Po2、
  参数锁和 ONNX 导出回读；提供 KWS 用户示例与真实 CUDA 集成测试。
- 在 `operators/quant-lstm/` 收录 QuantLSTM 源码、配置、构建脚本、测试和文档，
  与 QuantGRU 同级维护并支持独立构建；应用包增加 QuantLSTM wheel，环境使用公共 `rx-quant-env` v1。
- PyTorch 导出增加节点名/encoding key 双向对应及浮点计算输入覆盖检查；双输入计算
  必须覆盖两路浮点张量，标量常量与非浮点输入豁免。即使未启用量化器也不能放过缺失。
- PyTorch 导出末尾强制校验 ONNX 和 encodings；缺少已启用量化器的编码、通道或
  GRU/LSTM 内部参数，以及未校准、非法 scale、编码值不一致或最终图无效时直接报错。
- 增加独立的环境镜像构建、验证、加载和导出命令。
- 发布包提供仅含当前版本内容的 `CHANGELOG.md`，安装后保留，并纳入文件完整性校验。
- 增加 `CONTRIBUTING.md`，统一公开贡献、测试和文档更新要求。

### Changed

- 发布包启动命令使用对应 CUDA 变体的可读 runtime 标签；安装器按镜像 ID 校验并补齐标签，拒绝覆盖冲突标签。

- 安装器指定镜像文件的参数由 `--runtime-archive` 更名为 `--runtime-image`。

- GPU 环境改为公共 `rx-quant-env` v1；两个产品独立安装和更新，复用同一批固定 runtime 镜像。
- 发布构建支持 `--build-env-dir` 复用离线 build 镜像归档，`--runtime-dir` 更名为 `--runtime-env-dir`；本地缺少环境时校验并导入对应归档。
- 默认发布只生成应用包；镜像附件统一使用 `--with-images`，缺少归档时从本机固定 runtime 自动导出到交付目录 `images/`，移除 `--with-runtime`。
- 应用包只固定 runtime 镜像 ID，取消对归档文件 SHA256 的绑定；同一镜像重新导出的归档可跨应用包复用。归档独立提供 `.sha256` 文件校验。
- 安装器核对本次实际导入的镜像和归档内容，避免本机已有正确镜像时误接受错误归档；下载验证通过后才写入缓存，宿主机无需 Python。
- 独立环境导出、加载入口统一支持相同镜像的不同归档 SHA256，保留首次发布的环境身份记录。
- 安装器负责离线安装和自检，README 说明 Docker 挂载运行。

- 发布改为三个固定 runtime 镜像及三个独立应用包。用户通过 `install.sh` 离线安装，
  再按 README 使用原生 Docker 命令挂载运行；构建环境 `build-env` 继续供内部使用。
- runtime 支持独立 gzip 归档，应用包记录精确镜像身份及预留 HTTPS 下载地址。
- QuantGRU、QuantLSTM wheel 版本统一跟随 rx-met 产品版本，并保留 CUDA 变体标识。
- 验收改为纯 runtime、非 root 和只读应用目录，覆盖 ONNX、GRU、LSTM 的真实 GPU 流程。
- KWS 示例改用 `QuantizableBatchNorm2d`，经 FX 展开后校准 Sub/Div/Mul/Add 的两路
  输入；导出根据参数使用位置恢复被 reshape/常量折叠改名的 ONNX initializer。
- KWS 用户示例通过 `--rnn_type gru|lstm` 选择循环层，共用量化配置，输出按网络类型隔离。
  随机 LSTM 流程检查归入测试，客户说明随 examples 交付；发布验收覆盖两种模式。
- 同步 QuantLSTM 上游更新，默认构建 Release CUDA 库并关闭测试和示例，更新使用文档；
  算子目录不再收录上游 `.github/`、`docker/` 和 `.dockerignore`。
- 更新 QuantGRU 使用、配置和计算流程文档，合并 PWL 说明并清理历史算法草稿。
- 将 AIMET Python 包整理到 `src/`，将 AIMET 原生实现放在 `native/`，并将
  QuantGRU 放在 `operators/quant-gru/`。安装后继续沿用现有 Python import 名和 wheel 名。
- 将发布模板从顶层 `release/` 迁移到 `packaging/release-bundle/`，使用 `.in` 标识待渲染输入；
  CHANGELOG 统一按 `VERSION` 从仓库完整历史中提取，避免维护两份版本记录。
- 将公开命令按职责整理到 `scripts/release/` 和 `scripts/environment/`。
- 由 `pyproject.toml` 定义 Python 兼容范围，由 `docker/variants.json` 统一管理
  CUDA 变体和精确依赖；Bake 配置和 lock 文件由 `scripts/dependencies.py` 生成。
- 文档按用户指南、配置说明、维护者文档和架构决策重新组织。
- 用户 README 简化为安装、准备数据和工作目录、进入容器运行示例三个步骤；
  Docker 启动命令只保留一套，数据下载和预处理说明集中到 examples 文档。

### Fixed

- 为 CUDA 11.8/13.0 提供带来源与校验清单的 ONNX Runtime 离线头文件缓存，避免重复下载大型源码归档。

- 修复单向/双向 QuantLSTM 在 `bias=False` 时的 ONNX 导出：正确省略可选 bias 输入，
  消除缺少 `lstm.bias` 编码的校验错误；真实的全零 bias 仍保留参数和完整编码校验。
- prepare_model 在校准前自动展开 nn.GELU/F.gelu，保留精确或 tanh 近似语义，
  为所有中间计算提供独立编码；修复函数 GELU 的关键字输入准备及重复 prepare 命名。
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
- QuantLSTM 和 ONNX PTQ 默认配置随 wheel 打包，支持安装到外部目录后挂载使用。
- 补齐 ONNX 示例预处理后的数据根目录设置，使生成位置和读取位置一致。
- 修正 ONNX Runtime vendored README 更新后未同步的校验值，恢复 cu126 环境构建。

### Known limitations

- cu130 的 onnxsim/nanobind 在部分进程退出时输出引用泄漏诊断，已测 examples 的退出码为 0。
- 本版 GPU 验收使用 NVIDIA H200 NVL 和驱动 580.173.02，其他 GPU 架构和最低驱动未逐一实测。

### Upgrade notes

- 首次使用需要对应的公共 `rx-quant-env` v1 runtime。安装应用包后，按 README 挂载该版本的应用目录。
- 后续产品更新可复用兼容的 runtime；升级和回滚通过选择应用目录及其匹配的镜像完成。
- 工作目录、模型、数据和输出独立于应用安装目录保存。

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
