# Changelog

本文档记录本仓库的重要变更。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

`1.0.0` 起的对外版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。
`1.0.0` 之前的记录按日期排列，不属于 rx-met 版本号。

## [Unreleased]

### 变更

- 将 AIMET Python 包整理到 `src/`，AIMET 原生实现保留在
  `native/`，QuantGRU 移至 `operators/quant-gru/`；安装后的 Python
  import 名和发布 wheel 名保持不变。
- 将产品交付文档模板从顶层 `release/` 迁至
  `packaging/release-bundle/`，并使用 `.md.in` 标识待渲染输入。
- 将公开维护脚本按职责整理到 `scripts/release/` 和
  `scripts/environment/`；跨流程的原生构建、依赖管理及内部实现位置不变。
- 将 Python 兼容范围收敛到 `pyproject.toml`，以
  `docker/variants.json` 统一管理 CUDA 变体和精确依赖；Bake 配置及每个变体的
  自包含 lock 由 `scripts/dependencies.py` 生成并校验。
- 交付方式从旧通用镜像上的 wheel 软件包改为 rx-met 自有 Docker 镜像。
- 将 Docker 流程拆分为稳定环境与产品发布：每个 CUDA 变体分别维护可复用的
  `build-env` 和 `runtime-env`，源码更新只重编项目 wheel 和最终产品镜像。
- 新增独立的环境构建、验证、加载和导出脚本；脚本只生成可搬运文件，不自动上传
  到共享存储，复制或移动由维护人手工执行。
- `release_build.sh` 优先复用本机环境镜像，也可读取指定环境归档；两者都没有时
  自动调用环境构建脚本，显式提供归档但校验失败时不会回退公网。
- 最终镜像内置运行依赖、AIMET native、QuantGRU、examples 和版本元数据。
- 修复 ONNX PTQ quick start 使用错误 `rxmet_encodings` 字段的问题。

## [1.0.0] - 2026-09-02

首个对外版本。

## 2026-09-02

### 🔄 变更

- 仓库改为独立小模型仓；大模型量化不在本仓库。客户交付改为旧通用镜像上的
  wheel 软件包（`vYYMMDD`）。
- 示例入口改为 `quick_start_kws.py`（PyTorch）和 `onnx_ptq_quick_start.py`（ONNX）。ONNX 示例使用公开 MobileNetV2，由 `prepare_onnx_ptq_data.py` 生成静态 batch 与校准 npy。
- **重组 ONNX 导出与 PTQ 模块路径**
  - PyTorch sim → ONNX + encodings（含 GRU 导出/后处理）迁至 `src/aimet_torch/rx_export/`
  - 直接 ONNX PTQ 迁至 `src/aimet_onnx/rx_ptq/`
  - 删除原顶层包 `export_onnx_and_encodings/`（不再提供旧 import 路径）

- **Torch JSON 2 对不上的 `layer_type_config` 改为忽略，拼写错误仍报错**
  （`src/aimet_torch/utils_rx.py`）
  - 已注册的 `Quantized*` 当前模型没有 → 忽略（与 ONNX、通用清单约定对齐）
  - 未注册且模型里也没有 → `ValueError`（如 `QuantizedConv2D`）
  - verbose 时打印忽略的类型

- **ONNX PTQ 对齐现有 Torch RX 契约**
  （`src/aimet_onnx/quantsim_config/quantsim_config.py`、`src/aimet_torch/power_of_2_quantization.py`、`src/aimet_torch/utils_rx.py`）
  - `cover_range`（容差 2%）、激活/权重对称、Conv/Gemm/MatMul 的 bias encodings 不走统计校准，校准后再写 `Sb = Sx * Sw`（默认 INT32，不走官方 INT32 concretize）
  - `aimet_torch.power_of_2_quantization` 中上述纯数值函数改为从 `aimet_common.power_of_2` 再导出，原有 import 路径保持兼容
  - `apply_mixed_precision_bitwidth` 对带 `qc_quantize_op_dict` 的 ONNX sim 复用 JSON 2
  - QuantSim 配置器：同一 tensor 上 Reshape 输出与 Add 输入冲突时打开 quantizer（对齐 Torch 消费端再 fake-quant），不再断言失败

- **compiler encodings 为计算节点补齐 I/O**
  （`src/aimet_onnx/rx_ptq/compiler_encodings.py`、`mixed_precision.py`、`po2.py`）
  - 导出时沿 Pad/Reshape/MaxPool 等 grid-preserving 边补齐 I/O，后级 Conv/Concat 不再缺 input
  - 复用 JSON 2 时不映射 `QuantizedPad`（Torch 整数 Pad disable），避免关掉空间 ZeroPad
  - Po2 遇到 scale<=0 的 quantizer 时关闭它，不再中断导出
  - 校准前关闭 Conv/Gemm/MatMul bias fake-quant：近零 per-channel `scale=0` 会向后续图注入 Inf

- **`apply_power_of_2_workflow()` 的量化器信息打印受 `verbose` 控制**
  （`src/aimet_torch/utils_rx.py`）
  - 「修改后的量化参数」调用 `print_quantizer_info` 时传入 `verbose=verbose`
  - `verbose=False` 时不再刷屏打印修改后的量化器详情

- **降低 `prepare_model` 过程中的日志噪音**
  （`src/aimet_torch/model_preparer.py`）
  - `_prepare_traced_model` 不再对每个 Functional / Reused/Duplicate 节点输出
    `Adding new module for node` 的 `logger.info`
  - 图改写逻辑不变；大模型 prepare 时日志量显著减少

### ✨ 新增功能

- **计算类算子默认打开输入/输出量化**
  （`src/aimet_common/quantsim_config/compute_ops.py`、`src/aimet_torch/quantsim_config/quantsim_config.py`、`src/aimet_onnx/quantsim_config/quantsim_config.py`）
  - 定点编译器要求每个计算节点都有 I/O encodings；官方 JSON 不允许 `defaults.is_input_quantized`
  - 配置器按黑名单判定计算类（非 grid-preserving / 非索引控制流），未写入 `op_type` 的 Relu/BN/自定义 leaf 也会打开输入量化
  - `QuantGRU` 不套用这套默认，仍走 `GRU_config`；配置结束后会关掉它身上被官方 defaults 重新打开的 I/O/param 量化器
  - JSON 1 的 `op_type` 只保留例外；搬数据/索引算子（Reshape/Pad/Floor）默认不开
  - Torch / ONNX / compiler encodings 共用同一套判定；`examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json` 已清空计算类清单
  - 单测：`tests/test_compute_ops.py`、`tests/test_compute_op_io_defaults.py`
  - JSON 1 不再需要为 Conv/Add/Relu 等逐条写 `is_input_quantized`；要禁用某个计算类，继续用 JSON 2 的 `disable_quantization`

- **ONNX 输入 PTQ**
  （`src/aimet_onnx/`、`src/aimet_common/power_of_2.py`、`src/aimet_onnx/rx_ptq/`）
  - 直接读取已有 ONNX，校准后输出 clean ONNX 与 compiler-native encodings（`schema_version: 3`），不经过 `aimet_torch` 导出链
  - 示例入口：`examples/onnx_ptq_quick_start.py`
  - 与 PyTorch 示例共用 JSON 1 / JSON 2；图里没有的 `Quantized*` 与 `GRU_config` 忽略
  - `aimet_onnx` 补 `set_percentile_value`：必须在 `compute_encodings` 前设置；native 默认 100 等于 min-max。默认 `percentile=99.99`
  - RX 量化原文抽到 `src/aimet_common/power_of_2.py`，供 Torch / ONNX 共用，不改写算法：
    `is_power_of_2` / `find_closest_power_of_2_scale` / `verify_power_of_2_scale` /
    `recompute_min_max_for_new_scale` / `compute_aligned_bias_range`

## 2026-07-17

### 🐛 Bug 修复

- **修复 QuantGRU reload 后仍执行浮点 Forward 的问题**
  （`src/aimet_torch/staged_quantization_utils.py`）
  - `load_quantizer_encodings` 此前仅调用 QuantGRU 接口恢复量化参数，未开启
    `use_quantization`；重载模块虽然 `is_calibrated()=True`，实际仍走浮点 GRU 路径，
    导致 QAT 导出前与 Reload 推理结果不一致。
  - 现由 AIMET 适配层在 QuantGRU 量化参数加载成功后显式设置
    `module.use_quantization=True`。

- **修复 reload 依赖原始混合精度配置才能恢复量化器的问题**
  （`src/aimet_torch/staged_quantization_utils.py`）
  - `load_quantizer_encodings` 现在将 encoding 作为量化配置的权威来源，加载范围前自动恢复
    `bitwidth`、`is_symmetric` 与对应的 `qmin/qmax`
  - 修复 INT32 bias encoding 遇到默认 INT8 quantizer 时被静默跳过、造成 QAT 与 Reload
    推理结果不一致的问题
  - encoding 应用失败现在会计入 `skipped_count`；`skip_if_not_found=False` 时会直接报错

- **修复 reload 时 index dict 格式 encodings 未被加载的问题**
  （`src/aimet_torch/staged_quantization_utils.py`）
  - 2026-07-15 起 postprocess 将 activation `input` / `output` 规范为 `{"0": {...}, "1": {...}}`
  - 加载器原先把 dict 当成旧版扁平格式读 `real_min` / `real_max`，对 index dict 恒为 `None`，
    静默跳过加载，随后 forward 报 `quantization parameters are not initialized`
  - 现统一支持 list、index dict、旧版扁平 dict 三种格式；output 侧 `None` 槽也改为跳过而非中断
  - 影响范围：导出后再 `load_quantizer_encodings` 重载验证 / QAT reload

## 2026-07-15

### 🔄 变更

- **`apply_power_of_2_workflow()` 默认策略改为 `cover_range`**
  （`src/aimet_torch/utils_rx.py`）
  - 原先 `method` 默认为 `"round"`（全部四舍五入到最近的 `2^n`）
  - 现默认改为 `"cover_range"`：当 `real_range` 接近 `2^n`（相对误差 < `tolerance`）时仍四舍五入，
    否则增大 scale 以覆盖原浮点范围，减少截断风险
  - 显式传入 `method="round"` 的行为不变；底层 `apply_power_of_2_quantization()` 默认仍为 `"round"`
  - 影响范围：未显式指定 `method` 的 `apply_power_of_2_workflow()` 调用

- **回退“标量输入不创建量化器”，改为支持双路量化**
  （`src/aimet_torch/_base/quantsim.py`）
  - 背景：编译器现已支持“双路量化”——二元 elementwise 算子（Add / Multiply / Subtract /
    Divide 等）即使某一路输入是标量（Python 数值，或 0 维 tensor `torch.Size([])`），该路
    也可以拥有独立的量化参数。
  - 变更：撤销此前在 `QuantizationSimModel` 构造期“探测并移除标量输入槽量化器”的逻辑
    （`record_metadata` 中的标量判定、`_remove_scalar_input_quantizers`、`_scalar_input_slots`
    标记），恢复为所有输入槽（含标量路）均创建量化器、导出完整量化参数。
  - 保留：以下“稀疏输入槽按位对应”的修复不受影响、继续保留（它们修复的是通用的按位
    错位老问题，与标量特性无关，双路量化下同样安全）——后处理 `flatten_activation_io_index_dict`
    保留 index dict 编号、`staged_quantization_utils.py` 的 loader 跳过 `None` 槽；
    `apply_mixed_precision_bitwidth` 中的标量 guard 因 `_scalar_input_slots` 不再产生而自动失效。

### 🐛 Bug 修复

- **修复 encodings 加载在“稀疏输入槽”下中途中断的问题**
  （`src/aimet_torch/staged_quantization_utils.py`）
  - 加载器 `load_quantizer_encodings` 遍历 `input` list 时遇到 `None` 槽会直接 `break`，
    连带漏加载其后非标量那一路的量化器；改为**跳过（`continue`）**该槽，保持后续按位加载。
  - 影响范围：存在“某输入槽无量化器”（如标量输入被移除）的多输入算子 encodings reload。

- **修复 `QuantizationSimModel` 配置中非法 `op_type` 静默失效的问题**
  （`src/aimet_torch/quantsim_config/quantsim_config.py`）
  - `config_file` 的 `op_type` 字段原先遇到大小写或拼写错误（如 `matmul`、`Linear`）时，
    只记录 INFO 日志并继续执行，导致对应算子量化配置未生效且用户难以察觉
  - 现对未知 `op_type` 直接报错，并输出所有合法 key；合法 key 包含 ONNX op name、
    AIMET backend op name 和 functional 映射名，且大小写敏感
  - 影响范围：`QuantizationSimModel(..., config_file=...)` 的基础量化配置解析

- **修复混合精度 `layer_type_config` 类型名错误时静默失效的问题**
  （`src/aimet_torch/utils_rx.py`）
  - `apply_mixed_precision_bitwidth` 原先按 `type(module).__name__` 匹配 `layer_type_config`，
    但配置里写错类型名或大小写时不会报错，只会跳过配置
  - 现会在应用配置前校验 `layer_type_config` 的 key；若不存在于当前 `sim.model` 的模块类型中，
    直接报错并输出所有可用的合法类型名
  - 影响范围：混合精度 bitwidth 配置中按模块类型批量设置输入/输出/参数位宽的流程

- **修复 input encoding 导出时 Torch / ONNX 路径耦合导致的编码缺失**
  （`src/aimet_torch/_base/quantsim.py`）
  - `_update_encoding_dict_for_input_activations` 原先用 `zip(input_tensors, input_encodings)`
    同时写 ONNX 与 Torch encoding；当 ONNX 对重复输入去重（如 `mul(x, x)`）时，
    Torch encoding 会漏掉未覆盖到的 `input_quantizer`
  - 现将两条路径拆开：ONNX sidecar 按 distinct input tensor 导出，
    Torch encoding 按全部 PyTorch `input_quantizer` 导出，保证 reload / QAT 契约完整
  - 影响范围：多输入算子、ONNX 输入 tensor 数与 PyTorch input quantizer 数不一致时的 encoding 导出

- **修复 postprocess 删除 activation input/output 编号的问题**
  （`export_onnx_and_encodings/postprocess_refactor/encodings_ops.py`）
  - `flatten_activation_io_index_dict` 原先会把 `{"0": {...}, "1": {...}}`
    展平成 list，导致后处理后的 `HA.encodings` 丢失 input slot 编号
  - 现将 activation `input` / `output` 规范为带编号的 index dict，保留 `"0"` / `"1"` 等编号；
    若中间流程产生 list，也会按顺序恢复为 `"0"`、`"1"` ...
  - 影响范围：多输入算子的 activation encodings 后处理、按输入下标区分量化参数的导出结果

- **修复 postprocess 重命名共享 initializer 时误影响其他节点的问题**
  （`export_onnx_and_encodings/postprocess_refactor/renaming.py`）
  - Conv 权重 / bias initializer 重命名时，原逻辑默认直接 rename 并更新全图引用；
    当多个 Conv 共享同一个 initializer 时，会把其他节点输入也一并改名，导致 ONNX 节点参数名与
    `param_encodings` 对应关系混乱
  - 现按 initializer 使用次数区分处理：只被一个节点使用时直接 rename；
    被多个 Conv 共享时 clone 一份给当前 Conv，仅替换当前节点输入，并复制对应 param encoding
  - 影响范围：ONNX 后处理阶段 Conv 权重 / bias 重命名、共享 initializer 模型的 encodings 对齐

- **修复 fake-quant 模块（如 `FakeQuantizedSum`）未被 Power-of-2 处理**
  （`src/aimet_torch/power_of_2_quantization.py`）
  - POT2 的 apply / verify / info 三处 `isinstance` 判断仅匹配 `QuantizationMixin`，
    导致 `FakeQuantizationMixin` 算子（如 `FakeQuantizedSum`）被跳过、scale 保持非 POT2
  - 将判断放宽到 `BaseQuantizationMixin`，使 fake-quant 的 input/output 量化器
    也参与 POT2 转换与校验
  - 影响范围：模型中以 fake-quant 实现的算子（如 `Sum`）的 POT2 导出

- **修复 INT32 最小 scale 地板不是 2 的幂**（`src/aimet_common/quantsim.py`）
  - `_get_minimum_scale` 返回 `0.01 / num_steps`，INT32 下为 `2.33e-12 = 2^-38.64`（非 POT2）
  - 近零 INT32 bias 通道触发该地板后落到非 POT2 scale，且 `get_scale()` 每次读取都会
    `clamp_min_` 到该地板，导致后续 Power-of-2 步骤无法纠正
  - 将地板向上取整到最近的 2 的幂（INT32 → `2^-38`），保留“可表示 -0.005..0.005”的语义；
    INT8/16 不受影响（本就是 `2^-23`）
  - 影响范围：近零 bias 的 INT32 per-channel 量化、POT2（仅移位）硬件导出

- **修复 Linear 层 bias scale 未对齐到 Sx·Sw**（`src/aimet_torch/utils_rx.py`）
  - `_align_conv_bias_scale` 原先仅覆盖卷积类型，`nn.Linear` 的 bias 仍走 AIMET 独立标定，
    scale 与累加器 `Sx·Sw` 脱节（可能小到 `2^-38`、移位超 32 位），近零通道还会撞上非 POT2 地板
  - 将 `nn.Linear` 纳入对齐范围，使所有卷积/全连接层 bias 统一采用 `Sb = Sx·Sw`
    （构造上即为 POT2，移位约 17~26 位）
  - 影响范围：含 Linear 层的 POT2 量化导出、INT32 bias 的整数域相加与移位预算

## 2026-07-09

### ✨ 新增功能

- **`apply_mixed_precision_bitwidth` 支持按输入下标配置混合精度位宽**
  （`src/aimet_torch/utils_rx.py`）
  - 新增 `input_bitwidths` 配置项，可按 input slot 精确设置输入量化器位宽，
    例如 `input_bitwidths: [8, 16]` 分别作用于 `input_quantizers[0]` 和 `input_quantizers[1]`
  - 未配置 `input_bitwidths` 时继续使用原有 `input_bitwidth` 行为，保持旧配置兼容
  - 缺失 `input_quantizer` 的自动补建逻辑保持不变：仍仅对 `idx == 0` 的主输入生效，
    避免默认量化 scalar、mask、dim 等非主数据输入
  - 影响范围：需要同一算子多个输入使用不同 bitwidth 的混合精度配置，
    如 `Div` / `Mul` 等双输入 tensor 算子的 consumer-side 输入量化

### 🐛 Bug 修复

- **修复量化器与输入 tensor 的 CPU/CUDA 设备不一致问题**
  （`src/aimet_torch/utils_rx.py`、`src/aimet_torch/v2/nn/true_quant.py`）
  - `apply_mixed_precision_bitwidth` 为 `QuantizedVar` 等模块补建 `input_quantizer` 时，
    自动将新 quantizer 迁移到模块所在设备，避免 quantizer 留在 CPU 而模块在 GPU
  - `_quantize_if_applicable` / `_quantize_dequantize_if_applicable` 量化前，
    将输入 tensor 对齐到 quantizer 所在设备（常量/scalar 输入常在 CPU）
  - 影响范围：混合精度 bitwidth 配置、QAT 前向中涉及常量/标量输入的量化路径
    （如 `Div` 第二路输入量化、`QuantizedVar` 等无 ONNX 映射模块）

### 🔧 改进

- **`apply_power_of_2_workflow()` 支持控制 QuantGRU 的 POT2 模式**
  - 新增对 QuantGRU POT2 mode 的外部配置能力，调用工作流时可显式控制是否启用 power-of-two 量化约束
  - 影响范围：使用 `apply_power_of_2_workflow()` 配置 QuantGRU 量化策略的流程

## 2026-06-09

### ✨ 新增功能

- **`export_onnx_json` 支持按 GRU 模块类型自动分发导出逻辑**
  - 自动识别 `QuantGRU` / `OptimizedQuantizableGRU`，选择匹配的 ONNX 导出路径
  - 减少下游脚本对具体 GRU 实现的手动分支判断
  - 影响范围：使用 `export_onnx_and_encodings.export_onnx_json` 导出 GRU 模型的流程

### 🔧 改进

- 更新 quick start 示例与完整量化配置示例，保持示例参数和当前导出流程一致

## 2026-04-29

### 🐛 Bug 修复

- **修复 `export_onnx_and_encodings` 子模块的 import 路径错误（继承自 2026-04-21）**
  - 2026-04-21 的 wheel 中 `export_onnx_and_encodings/{export_onnx_json, custom_gru_onnx, custom_bn_onnx}.py`
    使用了错误的 import 路径 `from aimet_rx.aimet_torch.xxx import ...`，导致
    `from export_onnx_and_encodings.export_onnx_json import export_onnx_json` 抛出
    `ModuleNotFoundError: No module named 'aimet_rx'`
  - 仓库源码已在 commit `b2a5d06 "Update Import"`（2026-04-28）改回正确的
    `from aimet_torch.xxx import ...`；本记录为重新打 wheel 发布该修复
  - 影响范围：所有依赖 `export_onnx_json` / `custom_gru_onnx` / `custom_bn_onnx` 的下游代码

### 🔧 改进

- **打包配置统一到 `pyproject.toml`（PEP 621 单一真相源）**
  - 删除 `setup.py`（此前与 `pyproject.toml` 双源、版本号一度脱节）
  - 把 `packages` / `package_data` / `include_package_data` / `zip_safe` 等
    setuptools 配置迁移到 `[tool.setuptools]` 子表
  - 运行时依赖改为 `dynamic`，由 `pyproject.toml` 直接读 `requirements.txt`，
    避免 `pyproject` `dependencies` 字段与 `requirements.txt` 双重维护

## 2026-04-21

### ✨ 新增功能

- **添加 QuantGRU 的源码级集成支持**
  （`src/aimet_torch/v2/nn/modules/custom.py`、`src/aimet_torch/model_preparer.py`）
  - 在 AIMET v2 自定义量化模块注册表中增加 `QuantGRU` 的透传包装器
  - `QuantGRU` 继续使用其自身的内部量化实现，不额外叠加 AIMET 输入/输出量化器

### 🔧 改进

- 增强 `prepare_model` 对 `QuantGRU` 的处理逻辑：
  - 自动将 `QuantGRU` 识别为 leaf module，避免 FX 展开其内部实现
  - 使用 `QuantGRU` 时，无需在业务脚本中额外手动配置 `module_classes_to_exclude=[QuantGRU]`
  - 降低 `quant-gru-pytorch` 接入 AIMET 的样板代码和使用门槛

### ⚠️ 已知问题

- 本日期对应的 wheel 包含错误的 import 路径，导致 `export_onnx_and_encodings` 子模块在使用时报
  `ModuleNotFoundError: No module named 'aimet_rx'`，详见 2026-04-29 的修复说明。

## 2024-12-24

### 🐛 Bug 修复

- **修复对称量化配置问题**（`src/aimet_torch/utils_rx.py`）
  - 修复 `apply_mixed_precision_bitwidth` 在设置对称量化时，`qmin` / `qmax` 没有正确更新的问题
  - 修复前：设置 `symmetric=True` 后，`qmin` / `qmax` 仍然保持非对称值（如 `0, 255`）
  - 修复后：正确更新为对称范围（如 8-bit: `-128, 127`；2-bit: `-2, 1`）
  - 影响范围：所有通过 JSON 配置文件设置 `input_symmetric` / `output_symmetric` 的量化器

### 🔧 改进

- 增强量化参数设置逻辑：
  - 位宽变更时自动根据对称性更新 `qmin` / `qmax`
  - 对称性变更时自动根据当前位宽更新 `qmin` / `qmax`
  - 确保量化器状态始终一致

## 2024-12-23

### ✨ 新增功能

- 添加混合精度位宽配置工具函数
- 支持通过 JSON 配置文件灵活控制量化参数

## 2024-12-20

### ✨ 初始版本

- 基于 AIMET 官方版本构建
- 提供 PyTorch 和 ONNX 量化支持
