# Changelog

本文档记录 rx-met 的公开版本和重要兼容性变化。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，正式版本遵循
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [1.1.0] - 2026-10-12

### Added

- **QuantLSTM 集成**：支持 FX、AIMET v2 QuantSim、混合精度、校准、Po2、参数锁和 ONNX 导出恢复，CUDA 版支持浮点训练和 QAT；KWS 示例可选择 GRU 或 LSTM。
- **CPU 校准与 PTQ**：QuantLSTM 接通 CPU 浮点采样、18 个内部量化点校准和 FP32 q-carrier 量化前向；QuantGRU 复用主线 CPU 整数核心，补齐浮点校准、MinMax/SQNR/Percentile 和编码恢复。CPU 支持单向/双向、affine/Po2 及现有位宽和参数粒度，要求 `eval()` 与无梯度上下文，不支持浮点训练或 QAT。
- **ONNX 循环层 PTQ**：标准 GRU/LSTM 使用原生算子校准、量化、Po2 和编码导入导出，应用 `GRU_config` / `LSTM_config`，支持双向内部编码、逐通道参数、可选 h/c 状态及同图运行；可使用 CPU/CUDA，最终模型保留标准 ONNX GRU/LSTM 节点，clean ONNX 与编译器 JSON 格式保持兼容。
- **CPU 构建**：新增 `cpu` 依赖与发布变体，使用 CPU PyTorch、ONNX Runtime 和不链接 CUDA 的原生扩展；构建脚本通过 `RX_MET_ENABLE_CUDA=0` 生成三个项目 wheel。
- **PTQ 示例对照**：KWS 支持加载浮点检查点并跳过训练和 QAT，两个示例支持显式选择 CPU/CUDA 和复用相同校准样本；新增 ONNX 结构、initializer、JSON 及阶段输出的逐项比较和四次示例运行编排。
- **自动运行记录**：自动写入 `run.log` 和 `run.json`，记录详细日志、运行状态、版本、配置快照和产物 SHA-256，无需用户初始化 logger。`verbose` 只控制终端详细程度；失败保留日志和中间文件，成功清理临时工作目录。
- **ONNX PTQ 验证**：报告完整验证集上 FP32、PTQ、Po2 和恢复后的 Top-1 与输出误差；导出补全中间张量的 shape/dtype，并检查恢复后的量化输出和编译器编码。
- **公共运行环境**：提供可复用的 `rx-quant-env` v1，支持 `cpu`、`cu118`、`cu126`、`cu130`，以及独立的环境构建、验证、加载和离线导出命令。
- **Bias 设计差异说明**：记录普通层 INT32 乘积 scale 对齐与原生 GRU/LSTM 独立低位宽 bias 编码、训练余量及累加域重缩放的差异；部署约束与后续方案[待讨论](docs/QuantLSTM_integration.md#bias-编码差异)，仅记录现状，未调整 bias 策略。

### Changed

- **量化配置**：Torch、ONNX 和原生 GRU/LSTM 默认使用对称量化，仍支持显式非对称配置。普通 Torch/ONNX 量化器的整数符号与对称性独立，默认使用 INT，可显式选择 UINT。
- **编译器编码**：统一 Torch/ONNX 的编码转换、零点语义和循环层元数据，升级为 `schema_version=4`；兼容性要求见下方升级说明。
- **编码文件名**：Torch 与 ONNX PTQ 的最终编译器编码统一为 `<prefix>.json`，导出接口的参数和返回类型保持不变；编码结构、数值及 AIMET 原始编码命名不变。
- **输出目录**：RX 导出接口将指定目录作为输出根目录，默认创建 UTC 时间戳子目录；可用 `configure_output(timestamp_output=False)` 切换为固定目录并覆盖同名输出。最终产物放入 `exports/`，KWS 检查点放入 `checkpoints/`，默认根目录基于当前运行目录，原有输出目录覆盖参数继续有效。
- **ONNX 示例**：入口统一命名为 `quick_start_onnx_ptq.py`。MobileNetV2 + ImageNette 准备脚本只需模型和 parquet 路径，生成可复用的模型、校准数据和带标签的验证数据；准备与量化分两步执行，默认路径相互对应，原始输入可只读挂载。
- **公共环境**：按版本标签复用兼容镜像，源码摘要和镜像 ID 仅供追踪；移除仓库身份记录依赖。ORT 构建材料统一迁入 `docker/environment/third_party/`。
- **离线交付**：产品包和公共运行环境分别交付、独立校验。应用包按环境版本标签复用兼容 runtime，同一镜像重新导出的归档可继续使用；安装器核对归档内容并准备可读的镜像标签，宿主机无需 Python。
- **构建与安装**：构建前检查已有交付目录，默认在临时目录完成安装验收并清理。构建和安装分别支持 `--replace`，新内容验收通过后才替换目标，失败保留或恢复原内容。
- **默认打包目标**：应用打包、环境构建、runtime 导出和 Docker bake 默认仅选择 `cu118`、`cu126`、`cu130`；CPU 必须显式指定 `cpu`，支持与 CUDA 变体同时选择，`--with-env` 不会隐式加入 CPU。
- **项目组织**：源码、原生库、循环算子和发布脚本按职责分目录维护，保留现有 Python import 名称。算子 wheel 跟随产品版本并保留 `cpu` / `cuXXX` 变体标识；配置资源随 wheel 分发，支持安装目录迁移。
- **发布说明**：应用包及安装目录仅包含当前版本的 CHANGELOG，并纳入文件校验；使用流程、配置说明和贡献要求集中到对应文档。

### Fixed

#### 校准与整数量化

- ONNX Clip 的 min/max 按控制端口保留浮点值，修复 epsilon 被量化成 0 后导致 Log 非有限值的问题；共享数据分支继续量化，透明控制路径只在运行图中适配，配置、导出及编码重载共用端口角色校验。
- Torch/ONNX Percentile 共用动态直方图、样本计数和范围计算，修复后续大值漏统计、常量及极小范围处理、逐通道累计和重复校准问题，减少两端 scale 差异。
- CPU GRU host 收集器首次初始化时保留原始观测范围，修复 SQNR/Percentile 校准和 bias 范围丢失；CPU LSTM 按整批、逐量化点生成直方图后统一合并，与 CUDA 采样规则一致，修复按时间行动态重新分桶导致 affine 网格和 PTQ 输出额外偏移的问题。
- 混合精度正确应用默认位宽、逐通道设置及输入端口配置，保留显式层配置的优先级；ONNX 的输入精度可穿过 Reshape/Transpose 传递，常量 MatMul 与权重、bias 按各自角色处理，共享量化器配置冲突提前报错。
- 普通 Conv/Linear 等层的 bias 校准遵循配置位宽；Po2 工作流启用 bias 对齐时，按 `Sb = Sx * Sw` 对齐，并检查目标位宽和可表示范围，防止静默饱和。Torch 默认关闭此对齐，ONNX 默认开启；原生 GRU/LSTM 内部 bias 不在此范围。零值通道保留有效网格，非法 scale 在修改或导出前报错。
- CPU/CUDA QDQ 修复 INT32 小整数码丢失和非 Po2 半格舍入偏差，nearest 统一采用 ties-to-even，正确处理 NaN、Inf 和饱和边界。ONNX PTQ 默认关闭 TF32，显式 provider 选项仍可覆盖。

#### Torch 与循环算子

- 原生 GRU/LSTM 双向输出按正反向真实网格传播，不再将 forward 摘要套给整个张量；保形路径保留方向网格，独立量化边界明确建立新网格，并严格拒绝误导性标量声明。
- ONNX GRU/LSTM 共用循环算子调度，修复 LSTM 逐通道配置下初始化失败及 `LSTM_config` 被忽略；GPU 版 QuantGRU 的 ONNX 适配器明确拒绝 CPU 执行请求，提示安装 CPU wheel，避免 `CPUExecutionProvider` 下隐式调用 CUDA。
- FX 展开的 BN、自定义模块在参数读取处执行 QDQ，避免重复量化，并遵循所选校准方案、类型和位宽；保留原始浮点参数的梯度、重新校准和禁用能力。
- QuantGRU 统一 FP32/INT32 carrier 的整数与 LUT 前向，修复更新门 zero point 重复扣减和截断 mask 的梯度传播，确保 mask 在 bias 累加与跨 CUDA stream 同步前生效；CUDA 版在 eval 模式需要梯度时保留反向状态。
- QuantGRU/QuantLSTM 为固定 bias 网格保留训练余量，避免零偏置的微小 QAT 更新立即饱和；加载已有编码时保留其网格。
- QuantLSTM 正确导出单向、双向 `bias=False` 模型；有符号对称量化使用完整 INT8/INT16 运行范围，校准、QAT、导出和恢复保持一致。
- GELU 展开和高维 Linear 导出保留计算语义及编码对应关系；修复 BN 的 running variance 更新、累计平均和 QAT 冻结识别。

#### 导出与恢复

- 编译器新产物直接严格验收 schema v4 的字段、数组、整数网格端点、真实 ONNX 端口及循环方向映射，拒绝重复 JSON 键和非法端口；提供独立的 `scripts/verify_compiler_artifacts.py`。旧编码加载保留兼容，不用迁移或自动补写代替新产物验收。
- Torch/ONNX 导出共用 double 精度整数网格生成端点，修复 INT32 上界多一个整数码及验收容差掩盖损坏的问题；bias 累加器对齐保留显式粒度，单通道不再自动改成逐张量。
- 编译器、原始 AIMET 和阶段编码按保存的类型、整数范围、scale 和零点精确恢复，避免 INT32 精度丢失、符号误判或重载时改变网格；兼容旧 offset 约定，拒绝冲突、无效或有歧义的编码。
- 导出严格检查节点与编码对应关系、浮点输入覆盖、已启用量化器和循环层内部参数；恢复时拒绝缺失或篡改的编码、非有限输出及不一致的模型产物。
- ONNX 按实际 dtype 处理校准输入，保留 INT64 索引、BOOL mask、FLOAT16 和标量形状；整数、布尔及控制端口不再错误传播浮点量化编码，拒绝索引截断、溢出和错误的端口编码。
- ONNX GRU 后的计算输入在实际消费位置校准，避免重复量化或误用整段序列的网格；Concat 输入网格与恢复时的共享量化器检查保持一致。
- GRU/LSTM 显式关闭量化后可正常导出，通过 `excluded_layers` 记录禁用层，同时保持已启用层的严格校验。
- ONNX 简化保留携带量化边界的乘一、加零和单输入 Concat；LSTM 外层输入输出采用逐张量编码，h/c 状态保留在对应方向的内部编码中。

#### 示例与验收

- KWS 使用可量化 BN 和无随机增强的校准样本；检查点记录网络类型，保存原始模型时移除辅助量化器，支持 `weights_only` 恢复，验证失败时退出。
- `run.json` 按编码文档结构识别 `.json` 和 `.encodings` 中的格式版本，正确记录最终编译器编码的 schema 与 SHA-256，避免将普通 metadata 当作编码文件。
- KWS 验收定位本次运行的时间戳目录，从 `exports/` 读取最终 `.json` 并严格验收 schema v4，从 `checkpoints/` 检查 QAT 产物，修复旧目录和端口列表假设导致的验收失败。
- 往返分析使用比较报告记录的模型、最终编码、原始 AIMET 编码和 metadata 路径，支持容器外迁移及历史 `.encodings` 产物，避免误读同目录的原始编码或其他运行的文件。

#### 构建与安装

- 修复 ONNX Runtime 离线头文件缺失及校验值不同步、QuantGRU CPU 参考实现引用缺失测试源码等构建问题。
- 安装器验证显式提供或下载的 runtime 归档，避免本机已有正确镜像时误接受错误归档；下载内容通过验证后才进入缓存，产品和环境交付目录不再混入发布锁文件。

### Known limitations

- CPU 原生 GRU/LSTM 的前向要求 `eval()` 与 `no_grad()` / `inference_mode()`，训练模式或启用梯度时主动报错；该限制不适用于普通 PyTorch 层。
- CPU/CUDA 独立校准及浮点后端的 affine PTQ 结果不保证逐值相同；制品比较工具保留 FP32、affine、Po2 各阶段的原始差异，不通过放宽容差将其记为一致。
- KWS 前端 Log 前的小信号可能量化为零，其定义域及精度问题仍待处理；当前保留整数 QDQ，不使用 FP32 回退。
- 原生 C++/CUDA 直接写入终端的文本不经过 Python 日志捕获。
- cu130 的 onnxsim/nanobind 在部分进程退出时输出引用泄漏诊断，已测 examples 的退出码为 0。
- 本版 GPU 验收使用 NVIDIA H200 NVL 和驱动 580.173.02，其他 GPU 架构和最低驱动未逐一实测。

### Upgrade notes

- **编码格式**：schema v4 统一使用 `real = scale * (q - zero_point)`。外部消费者需按版本处理非对称零点符号；Torch 重载兼容无版本和 schema 1/2/3 的旧文件。格式转换不改变实际量化网格。
- **循环层字段**：整数记录补齐 `enc_type` 和 `n`；顶层 `input/output` 改为端口字典，如 `{"0": {...}}`，内部 `output` 保留单元素列表。加载接口兼容旧端口列表和缺少 `n` 的文件。
- **整数符号**：需要非对称或无符号量化时显式配置。JSON2 支持 `*_signed`、`input_signeds` 和 `default_config.signed`，更改对称性或位宽不会自动切换 INT/UINT；UINT 对称模式将负值截为零。
- **编码路径**：读取最终编码时使用 `export_onnx_json()` 返回的路径；原先自行拼接 `<prefix>.encodings` 的调用需更新。历史编译器 `.encodings` 仍可验收和分析，新产物旁的 AIMET 原始 `.encodings` / `.aimet.encodings` 不能替代最终 `.json`。
- **产物路径**：示例输出根目录为当前运行目录下的 `output/quick_start_kws/<gru|lstm>/` 或 `output/onnx_ptq_quick_start/`；默认产物位于时间戳目录的 `exports/`，关闭时间戳后位于根目录的 `exports/`。
- **示例入口**：原 `onnx_ptq_quick_start.py` 更名为 `quick_start_onnx_ptq.py`，命令及引用需同步更新；下载与两步运行方式见[示例说明](examples/README.md)。
- **CPU 变体**：CPU 部署需安装对应 wheel 和 runtime；原生 GRU/LSTM 仅支持校准/PTQ，KWS 使用浮点检查点并启用 `--ptq-only --device cpu`，不在 CPU 上执行示例默认的训练/QAT 流程。
- **安装与更新**：首次离线使用需同时提供产品目录和对应的 `rx-quant-env` v1 环境目录，后续产品更新可复用兼容 runtime。`install.sh --replace` 仅替换同名安装；其他版本、模型、数据和用户工作目录独立保留。

发布相关命令使用以下参数，替代旧名称：

| 用途 | 当前参数 | 替代的旧参数 |
| --- | --- | --- |
| 安装时指定 runtime 归档 | `install.sh --runtime-image FILE` | `--runtime-archive` |
| 构建时指定 runtime 归档目录 | `build.sh --runtime-env-dir DIR` | `--runtime-dir` |
| 交付时附带公共运行环境 | `build.sh --with-env` | `--with-runtime`、`--with-images` |

构建环境归档可用 `--build-env-dir DIR` 复用。应用包不再内嵌镜像。

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
