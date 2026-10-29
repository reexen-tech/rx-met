# Changelog

本文档记录 rx-met 的公开版本和重要兼容性变化。格式遵循
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，正式版本遵循
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [1.1.1] - 2026-10-29

### Changed

- 更新 KWS 示例网络：MFCC 前处理保持浮点并移至量化网络之外，调整 Attention 的算子组合，并将线性层替换为等价的 1×1 卷积。
- 公共环境目录改用每个镜像归档的独立 `.sha256` 校验文件，取消外层聚合 `SHA256SUMS`；目录 README 补充镜像、校验文件和元数据的用途及加载方式。
- 产品交付目录增加通用 README，说明应用归档和校验文件的用途、CPU/CUDA 变体选择及首次安装步骤。

### Fixed

- 修复以非 root UID 运行容器时，部分模型处理流程因用户名查询失败而中断的问题。
- 修复安装时对公共环境变体的识别，兼容不同版本的环境元数据。
- 修复 Python 3.10 下发布和环境命令省略变体时无法运行的问题；默认处理三个 CUDA 变体，CPU 仍需显式选择。
- 修复仅含原生 QuantGRU / QuantLSTM 的模型执行 Po2 转换时失败或转换未生效的问题，并修正普通层与原生循环层混用时的转换和结果比较。
- 修复 Po2 流程对 FX 控制常量的误判，以及参数比较对有符号、非对称量化的误报；对未校准或非法 scale 给出明确错误。
- 修复同一进程反复准备模型、创建量化模型时的临时追踪对象和 FX 源码累积，减少重复量化的内存占用；覆盖 `cpu`、`cu118`、`cu126`、`cu130`。
- 修复并发、派生进程和异步任务中的运行日志归属及任务结束后的资源释放问题。
- 修复使用 onnxsim 简化模型后，进程退出时可能出现的 nanobind 引用泄漏告警。

### Upgrade notes

- KWS 分类网络及导出模型改为接收外部提取的 MFCC 特征；示例加载器支持迁移模型类型和参数形状匹配的旧检查点。输入规格及使用方式见[示例说明](examples/README.md)。

## [1.1.0] - 2026-10-26

### Added

- **QuantLSTM 集成**：支持 FX、AIMET v2 QuantSim、混合精度、校准、Po2、参数锁和 ONNX 导出恢复，CUDA 版支持浮点训练和 QAT；KWS 示例可选择 GRU 或 LSTM。
- **CPU 校准与 PTQ**：QuantGRU / QuantLSTM 支持浮点校准、MinMax / SQNR / Percentile、编码恢复和量化前向，覆盖单向/双向、affine/Po2 及现有位宽和参数粒度。CPU 前向要求 `eval()` 与无梯度上下文，不支持浮点训练或 QAT。
- **ONNX 循环层 PTQ**：标准 GRU/LSTM 使用原生算子校准、量化、Po2 和编码导入导出，应用 `GRU_config` / `LSTM_config`，支持双向内部编码、逐通道参数、可选 h/c 状态及同图运行；可使用 CPU/CUDA，最终模型保留标准 ONNX GRU/LSTM 节点，clean ONNX 与编译器 JSON 格式保持兼容。
- **CPU 部署**：新增 `cpu` 发布变体，提供 CPU PyTorch、ONNX Runtime 和不依赖 CUDA 的原生扩展。
- **PTQ 示例对照**：KWS 支持加载浮点检查点并跳过训练和 QAT；KWS 与 ONNX 示例支持显式选择 CPU/CUDA、复用校准样本，并比较导出模型、编码和各阶段输出。
- **自动运行记录**：自动写入 `run.log` 和 `run.json`，记录详细日志、运行状态、版本、配置快照和产物 SHA-256，无需用户初始化 logger。`verbose` 只控制终端详细程度；失败保留日志和中间文件，成功清理临时工作目录。
- **ONNX PTQ 验证**：报告完整验证集上 FP32、PTQ、Po2 和恢复后的 Top-1 与输出误差；导出补全中间张量的 shape/dtype，并检查恢复后的量化输出和编译器编码。
- **公共运行环境**：提供可复用的 `rx-quant-env` `202610`，支持 `cpu`、`cu118`、`cu126`、`cu130`，以及独立的环境构建、验证、加载和离线导出命令。
- **Bias 编码说明**：补充普通层 INT32 乘积 scale 对齐与原生 GRU/LSTM 低位宽 bias 编码的差异及部署约束，见[QuantLSTM 集成说明](docs/QuantLSTM_integration.md#bias-编码差异)。

### Changed

- **公共环境与归档**：公共环境采用独立版本 `202610`，兼容的产品更新可复用相同环境。Docker 镜像归档使用 `.docker.tar.gz`，交付目录为 `rx-quant-env-202610/`，附带 `.json` 元数据及 `.sha256` 校验文件。导出和加载支持 `--role build|runtime|all`，默认处理 runtime。
- **公共 CPU PyTorch**：升级为 PyTorch `2.9.1+cpu`、torchvision `0.24.1+cpu`、torchaudio `2.9.1+cpu`；使用旧 Torch 编译的原生扩展需要重新编译。
- **量化配置**：Torch、ONNX 和原生 GRU/LSTM 默认使用对称量化，仍支持显式非对称配置。普通 Torch/ONNX 量化器的整数符号与对称性独立，默认使用 INT，可显式选择 UINT。
- **编译器编码**：统一 Torch/ONNX 的编码转换、零点语义和循环层元数据，升级为 `schema_version=4`；兼容性要求见下方升级说明。
- **编码文件名**：Torch 与 ONNX PTQ 的最终编译器编码统一为 `<prefix>.json`，导出接口的参数和返回类型保持不变；编码结构、数值及 AIMET 原始编码命名不变。
- **输出目录**：RX 导出接口将指定目录作为输出根目录，默认创建北京时间（UTC+8）的 `YYYYMMDD_HHMMSS_ffffff` 时间戳子目录；`run.json` 保留带 `+08:00` 和微秒的完整开始、结束时间，日志同步使用北京时间并标注时区。可用 `configure_output(timestamp_output=False)` 切换为固定目录并覆盖同名输出。最终产物放入 `exports/`，KWS 检查点放入 `checkpoints/`，默认根目录基于当前运行目录，原有输出目录覆盖参数继续有效。
- **ONNX 示例**：入口统一命名为 `quick_start_onnx_ptq.py`。MobileNetV2 + ImageNette 准备脚本只需模型和 parquet 路径，生成可复用的模型、校准数据和带标签的验证数据；准备与量化分两步执行，默认路径相互对应，原始输入可只读挂载。
- **离线交付**：产品包和公共运行环境分别交付、独立校验。应用包按环境版本标签复用兼容 runtime，同一镜像重新导出的归档可继续使用；安装器核对归档内容并准备可读的镜像标签，宿主机无需 Python。
- **增量打包**：CPU/CUDA 应用归档可分批追加到同一产品目录，仅覆盖同名归档时需要 `--replace`。替换保留其他变体及已解压目录，失败时保留或恢复原有产物。安装器的 `--replace` 仅替换同名安装。
- **默认打包目标**：应用打包、环境构建、runtime 导出和 Docker bake 默认仅选择 `cu118`、`cu126`、`cu130`；CPU 必须显式指定 `cpu`，支持与 CUDA 变体同时选择，`--with-env` 不会隐式加入 CPU。
- **算子安装包**：QuantGRU / QuantLSTM wheel 跟随产品版本，保留 `cpu` / `cuXXX` 变体标识和现有 Python import 名称；配置资源随 wheel 分发，支持安装目录迁移。
- **交付文件**：每个应用归档配独立 `.tar.gz.sha256`，产品外层不再生成 README、CHANGELOG、`release.json` 或聚合 `SHA256SUMS`，向旧目录首次成功发布时清理这些旧文件。包内 README、manifest、`SHA256SUMS` 和当前版本 CHANGELOG 完整保留，安装内容继续校验；公共环境目录的说明、元数据及校验机制不变。

### Fixed

#### 校准与整数量化

- 修复 ONNX Clip 控制参数被量化为零后导致 Log 输出非有限值的问题，同时保留共享数据分支的量化。
- 修复 Torch/ONNX Percentile 对新增大值、常量、极小范围、逐通道数据及重复校准的处理，减少两端的 scale 差异。
- 修复 CPU GRU 的 SQNR / Percentile 校准及 bias 范围丢失问题；修正 CPU LSTM 的校准统计，减少 affine 编码和 PTQ 输出的额外偏移。
- 混合精度正确应用默认位宽、逐通道设置及输入端口配置，保留显式层配置的优先级；ONNX 的输入精度可穿过 Reshape/Transpose 传递，常量 MatMul 与权重、bias 按各自角色处理，共享量化器配置冲突提前报错。
- 普通 Conv/Linear 等层的 bias 校准遵循配置位宽；Po2 工作流启用 bias 对齐时，按 `Sb = Sx * Sw` 对齐，并检查目标位宽和可表示范围，防止静默饱和。Torch 默认关闭此对齐，ONNX 默认开启；原生 GRU/LSTM 内部 bias 不在此范围。零值通道保留有效网格，非法 scale 在修改或导出前报错。
- 修复 CPU/CUDA QDQ 的 INT32 精度丢失和非 Po2 舍入偏差，`nearest` 统一采用 ties-to-even，并正确处理 NaN、Inf 和饱和边界。ONNX PTQ 默认关闭 TF32，仍可通过 provider 选项覆盖。

#### Torch 与循环算子

- 修复双向 GRU/LSTM 输出的量化参数传播，保留正向和反向各自的编码，避免将单一方向的编码错误应用于整个张量。
- ONNX GRU/LSTM 共用循环算子调度，修复 LSTM 逐通道配置下初始化失败及 `LSTM_config` 被忽略；GPU 版 QuantGRU 的 ONNX 适配器明确拒绝 CPU 执行请求，提示安装 CPU wheel，避免 `CPUExecutionProvider` 下隐式调用 CUDA。
- FX 展开的 BN、自定义模块在参数读取处执行 QDQ，避免重复量化，并遵循所选校准方案、类型和位宽；保留原始浮点参数的梯度、重新校准和禁用能力。
- 修复 QuantGRU 更新门的 zero point 计算及截断区域的梯度传播，改善不同前向实现和 CUDA stream 下的结果一致性；CUDA 版在 eval 模式需要梯度时支持反向传播。
- QuantGRU/QuantLSTM 为固定 bias 网格保留训练余量，避免零偏置的微小 QAT 更新立即饱和；加载已有编码时保留其网格。
- QuantLSTM 在校准后切换 Po2 时，按新 scale 重算非对称 zero point 和实数范围；复制模型时保留原始校准下界，导入编码后按已有网格完成转换。
- 修复 QuantLSTM CPU/CUDA 非对称量化在舍入边界处的偏差，以及 QAT 对尚未发生饱和的边界值错误截断梯度的问题。
- QuantLSTM 正确导出单向、双向 `bias=False` 模型；有符号对称量化使用完整 INT8/INT16 运行范围，校准、QAT、导出和恢复保持一致。
- GELU 展开和高维 Linear 导出保留计算语义及编码对应关系；修复 BN 的 running variance 更新、累计平均和 QAT 冻结识别。

#### 导出与恢复

- 修复编译器编码中的字段、整数范围、ONNX 端口和循环方向映射检查，拒绝重复 JSON 键、非法端口及损坏的编码；可使用 `scripts/verify_compiler_artifacts.py` 独立检查导出结果。
- Torch/ONNX 导出共用 double 精度整数网格生成端点，修复 INT32 上界多一个整数码及验收容差掩盖损坏的问题；bias 累加器对齐保留显式粒度，单通道不再自动改成逐张量。
- 编译器、原始 AIMET 和阶段编码按保存的类型、整数范围、scale 和零点精确恢复，避免 INT32 精度丢失、符号误判或重载时改变网格；兼容旧 offset 约定，拒绝冲突、无效或有歧义的编码。
- 导出严格检查节点与编码对应关系、浮点输入覆盖、已启用量化器和循环层内部参数；恢复时拒绝缺失或篡改的编码、非有限输出及不一致的模型产物。
- ONNX 按实际 dtype 处理校准输入，保留 INT64 索引、BOOL mask、FLOAT16 和标量形状；整数、布尔及控制端口不再错误传播浮点量化编码，拒绝索引截断、溢出和错误的端口编码。
- ONNX GRU 后的计算输入在实际消费位置校准，避免重复量化或误用整段序列的网格；Concat 输入网格与恢复时的共享量化器检查保持一致。
- GRU/LSTM 显式关闭量化后可正常导出，通过 `excluded_layers` 记录禁用层，同时保持已启用层的严格校验。
- ONNX 简化保留携带量化边界的乘一、加零和单输入 Concat；LSTM 外层输入输出采用逐张量编码，h/c 状态保留在对应方向的内部编码中。

#### 示例与工具

- 修复 GRU 示例工具从其他目录启动时的路径问题，运行失败时正确返回非零退出码并恢复位宽配置。
- KWS 使用可量化 BN 和无随机增强的校准样本；检查点记录网络类型，保存原始模型时移除辅助量化器，支持 `weights_only` 恢复，验证失败时退出。
- `run.json` 按编码文档结构识别 `.json` 和 `.encodings` 中的格式版本，正确记录最终编译器编码的 schema 与 SHA-256，避免将普通 metadata 当作编码文件。
- KWS 和 ONNX PTQ 示例不再主动保存配置快照；Torch/ONNX QuantSim 负责保存解析后的基础配置，创建、验证和整流程包装不再重复抓取配置，其他操作保留自动记录。
- KWS 的 QAT/PTQ 和 ONNX PTQ 示例明确打印最终配套使用的 `.onnx` 模型与 `.json` 量化参数路径，原始 AIMET 编码标为调试参考。
- 修复 KWS 示例工具对时间戳目录及最终模型、编码和检查点路径的识别，避免读取错误运行的产物。
- 往返分析使用比较报告记录的模型、最终编码、原始 AIMET 编码和 metadata 路径，支持容器外迁移及历史 `.encodings` 产物，避免误读同目录的原始编码或其他运行的文件。

#### 构建与安装

- 统一三个项目的 wheel 发行包名与文件名，修正 QuantGRU 包名为 `quant-gru`，保持 `quant_gru` 导入方式不变。
- 修复离线构建时 ONNX Runtime 头文件缺失和 QuantGRU CPU 构建依赖不完整的问题。
- 修复安装器在本机已有 runtime 时误接受错误镜像归档的问题；显式提供或下载的归档均需校验，损坏的下载不会写入镜像缓存。

### Known limitations

- CPU 原生 GRU/LSTM 的前向要求 `eval()` 与 `no_grad()` / `inference_mode()`，训练模式或启用梯度时主动报错；该限制不适用于普通 PyTorch 层。
- CPU/CUDA 独立校准及不同浮点后端的 affine PTQ 结果不保证逐值相同；可通过制品比较工具查看 FP32、affine、Po2 各阶段的差异。
- KWS 前端 Log 前的小信号可能量化为零，其定义域及精度问题仍待处理；当前保留整数 QDQ，不使用 FP32 回退。
- 原生 C++/CUDA 直接写入终端的文本不经过 Python 日志捕获。
- cu130 环境在部分模型简化流程结束时可能输出 onnxsim/nanobind 引用泄漏告警。
- GPU 兼容性尚未覆盖所有架构和最低驱动组合；部署要求见 `docker/variants.json`。

### Upgrade notes

- **编码格式**：schema v4 统一使用 `real = scale * (q - zero_point)`。外部消费者需按版本处理非对称零点符号；Torch 重载兼容无版本和 schema 1/2/3 的旧文件。格式转换不改变实际量化网格。
- **循环层字段**：整数记录补齐 `enc_type` 和 `n`；顶层 `input/output` 改为端口字典，如 `{"0": {...}}`，内部 `output` 保留单元素列表。加载接口兼容旧端口列表和缺少 `n` 的文件。
- **整数符号**：需要非对称或无符号量化时显式配置。JSON2 支持 `*_signed`、`input_signeds` 和 `default_config.signed`，更改对称性或位宽不会自动切换 INT/UINT；UINT 对称模式将负值截为零。
- **编码路径**：读取最终编码时使用 `export_onnx_json()` 返回的路径；原先自行拼接 `<prefix>.encodings` 的调用需更新。历史编译器 `.encodings` 仍可验收和分析，新产物旁的 AIMET 原始 `.encodings` / `.aimet.encodings` 不能替代最终 `.json`。
- **产物路径**：示例输出根目录为当前运行目录下的 `output/quick_start_kws/<gru|lstm>/` 或 `output/onnx_ptq_quick_start/`；默认产物位于时间戳目录的 `exports/`，关闭时间戳后位于根目录的 `exports/`。
- **示例入口**：原 `onnx_ptq_quick_start.py` 更名为 `quick_start_onnx_ptq.py`，命令及引用需同步更新；下载与两步运行方式见[示例说明](examples/README.md)。
- **CPU 变体**：CPU 部署需安装对应 wheel 和 runtime；原生 GRU/LSTM 仅支持校准/PTQ，KWS 使用浮点检查点并启用 `--ptq-only --device cpu`，不在 CPU 上执行示例默认的训练/QAT 流程。
- **安装与更新**：首次离线使用需同时提供产品目录和对应的 `rx-quant-env` `202610` 环境目录，后续产品更新可复用兼容 runtime。`install.sh --replace` 仅替换同名安装；其他版本、模型、数据和用户工作目录独立保留。

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

### Compatibility

- 支持 Linux x86_64 和 NVIDIA GPU。
- 发布镜像提供 `cu118`、`cu126` 和 `cu130` 变体；精确依赖和驱动要求见
  `docker/variants.json` 及 `docs/User_guide.md`。
- Python 包支持 CPython 3.10 至 3.12，具体版本由镜像变体决定。

### Known limitations

- 模型、数据集和校准样本由用户单独准备。
- ONNX 直量化流程适用于 PTQ；QAT 使用 PyTorch 流程。
- 本版 CPU 模式仅用于调试。
