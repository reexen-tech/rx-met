# Changelog

本文档记录本仓库所有重要变更。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本 SemVer](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 🐛 Bug 修复

- **修复 `QuantizationSimModel` 配置中非法 `op_type` 静默失效的问题**
  （`aimet_torch/quantsim_config/quantsim_config.py`）
  - `config_file` 的 `op_type` 字段原先遇到大小写或拼写错误（如 `matmul`、`Linear`）时，
    只记录 INFO 日志并继续执行，导致对应算子量化配置未生效且用户难以察觉
  - 现对未知 `op_type` 直接报错，并输出所有合法 key；合法 key 包含 ONNX op name、
    AIMET backend op name 和 functional 映射名，且大小写敏感
  - 影响范围：`QuantizationSimModel(..., config_file=...)` 的基础量化配置解析

- **修复混合精度 `layer_type_config` 类型名错误时静默失效的问题**
  （`aimet_torch/utils_rx.py`）
  - `apply_mixed_precision_bitwidth` 原先按 `type(module).__name__` 匹配 `layer_type_config`，
    但配置里写错类型名或大小写时不会报错，只会跳过配置
  - 现会在应用配置前校验 `layer_type_config` 的 key；若不存在于当前 `sim.model` 的模块类型中，
    直接报错并输出所有可用的合法类型名
  - 影响范围：混合精度 bitwidth 配置中按模块类型批量设置输入/输出/参数位宽的流程

- **修复 input encoding 导出时 Torch / ONNX 路径耦合导致的编码缺失**
  （`aimet_torch/_base/quantsim.py`）
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
  （`aimet_torch/power_of_2_quantization.py`）
  - POT2 的 apply / verify / info 三处 `isinstance` 判断仅匹配 `QuantizationMixin`，
    导致 `FakeQuantizationMixin` 算子（如 `FakeQuantizedSum`）被跳过、scale 保持非 POT2
  - 将判断放宽到 `BaseQuantizationMixin`，使 fake-quant 的 input/output 量化器
    也参与 POT2 转换与校验
  - 影响范围：模型中以 fake-quant 实现的算子（如 `Sum`）的 POT2 导出

- **修复 INT32 最小 scale 地板不是 2 的幂**（`aimet_common/quantsim.py`）
  - `_get_minimum_scale` 返回 `0.01 / num_steps`，INT32 下为 `2.33e-12 = 2^-38.64`（非 POT2）
  - 近零 INT32 bias 通道触发该地板后落到非 POT2 scale，且 `get_scale()` 每次读取都会
    `clamp_min_` 到该地板，导致后续 Power-of-2 步骤无法纠正
  - 将地板向上取整到最近的 2 的幂（INT32 → `2^-38`），保留“可表示 -0.005..0.005”的语义；
    INT8/16 不受影响（本就是 `2^-23`）
  - 影响范围：近零 bias 的 INT32 per-channel 量化、POT2（仅移位）硬件导出

- **修复 Linear 层 bias scale 未对齐到 Sx·Sw**（`aimet_torch/utils_rx.py`）
  - `_align_conv_bias_scale` 原先仅覆盖卷积类型，`nn.Linear` 的 bias 仍走 AIMET 独立标定，
    scale 与累加器 `Sx·Sw` 脱节（可能小到 `2^-38`、移位超 32 位），近零通道还会撞上非 POT2 地板
  - 将 `nn.Linear` 纳入对齐范围，使所有卷积/全连接层 bias 统一采用 `Sb = Sx·Sw`
    （构造上即为 POT2，移位约 17~26 位）
  - 影响范围：含 Linear 层的 POT2 量化导出、INT32 bias 的整数域相加与移位预算

## [1.3.9] - 2026-07-09

### ✨ 新增功能

- **`apply_mixed_precision_bitwidth` 支持按输入下标配置混合精度位宽**
  （`aimet_torch/utils_rx.py`）
  - 新增 `input_bitwidths` 配置项，可按 input slot 精确设置输入量化器位宽，
    例如 `input_bitwidths: [8, 16]` 分别作用于 `input_quantizers[0]` 和 `input_quantizers[1]`
  - 未配置 `input_bitwidths` 时继续使用原有 `input_bitwidth` 行为，保持旧配置兼容
  - 缺失 `input_quantizer` 的自动补建逻辑保持不变：仍仅对 `idx == 0` 的主输入生效，
    避免默认量化 scalar、mask、dim 等非主数据输入
  - 影响范围：需要同一算子多个输入使用不同 bitwidth 的混合精度配置，
    如 `Div` / `Mul` 等双输入 tensor 算子的 consumer-side 输入量化

### 🐛 Bug 修复

- **修复量化器与输入 tensor 的 CPU/CUDA 设备不一致问题**
  （`aimet_torch/utils_rx.py`、`aimet_torch/v2/nn/true_quant.py`）
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

## [1.3.8] - 2026-06-09

### ✨ 新增功能

- **`export_onnx_json` 支持按 GRU 模块类型自动分发导出逻辑**
  - 自动识别 `QuantGRU` / `OptimizedQuantizableGRU`，选择匹配的 ONNX 导出路径
  - 减少下游脚本对具体 GRU 实现的手动分支判断
  - 影响范围：使用 `export_onnx_and_encodings.export_onnx_json` 导出 GRU 模型的流程

### 🔧 改进

- 更新 quick start 示例与完整量化配置示例，保持示例参数和当前导出流程一致

### 📦 发布与升级

```bash
pip install --upgrade --force-reinstall aimet_rx-1.3.8-py3-none-any.whl

# 安装后自检
python -c "from export_onnx_and_encodings.export_onnx_json import export_onnx_json; print('OK')"
```

## [1.3.7] - 2026-04-29

### 🐛 Bug 修复

- **修复 `export_onnx_and_encodings` 子模块的 import 路径错误（继承自 1.3.6）**
  - 1.3.6 wheel 中的 `export_onnx_and_encodings/{export_onnx_json, custom_gru_onnx, custom_bn_onnx}.py`
    使用了错误的 import 路径 `from aimet_rx.aimet_torch.xxx import ...`，导致用户侧
    `from export_onnx_and_encodings.export_onnx_json import export_onnx_json` 抛出
    `ModuleNotFoundError: No module named 'aimet_rx'`
  - 仓库源码已在 commit `b2a5d06 "Update Import"`（2026-04-28）改回正确的
    `from aimet_torch.xxx import ...`；本版本随之重新打 wheel 发布该修复
  - 影响范围：所有依赖 `export_onnx_json` / `custom_gru_onnx` / `custom_bn_onnx` 的下游代码

### 🔧 改进

- **打包配置统一到 `pyproject.toml`（PEP 621 单一真相源）**
  - 删除 `setup.py`（此前与 `pyproject.toml` 双源、版本号一度脱节为 `1.0.2`/`1.3.6`）
  - 把 `packages` / `package_data` / `include_package_data` / `zip_safe` 等
    setuptools 配置迁移到 `[tool.setuptools]` 子表
  - 运行时依赖改为 `dynamic`，由 `pyproject.toml` 直接读 `requirements.txt`，
    避免 `pyproject` `dependencies` 字段与 `requirements.txt` 双重维护

### 📦 发布与升级

```bash
pip install --upgrade --force-reinstall aimet_rx-1.3.7-py3-none-any.whl

# 安装后自检
python -c "from export_onnx_and_encodings.export_onnx_json import export_onnx_json; print('OK')"
```

## [1.3.6] - 2026-04-21

### ✨ 新增功能

- **添加 QuantGRU 的源码级集成支持**
  （`aimet_torch/v2/nn/modules/custom.py`、`aimet_torch/model_preparer.py`）
  - 在 AIMET v2 自定义量化模块注册表中增加 `QuantGRU` 的透传包装器
  - `QuantGRU` 继续使用其自身的内部量化实现，不额外叠加 AIMET 输入/输出量化器

### 🔧 改进

- 增强 `prepare_model` 对 `QuantGRU` 的处理逻辑：
  - 自动将 `QuantGRU` 识别为 leaf module，避免 FX 展开其内部实现
  - 使用 `QuantGRU` 时，无需在业务脚本中额外手动配置 `module_classes_to_exclude=[QuantGRU]`
  - 降低 `quant-gru-pytorch` 接入 AIMET 的样板代码和使用门槛

### ⚠️ 已知问题

- 本版本 wheel 包含错误的 import 路径，导致 `export_onnx_and_encodings` 子模块在使用时报
  `ModuleNotFoundError: No module named 'aimet_rx'`，详见 [1.3.7] 的修复说明。
  **请直接升级到 1.3.7。**

## [1.2.2] - 2024-12-24

### 🐛 Bug 修复

- **修复对称量化配置问题**（`aimet_torch/utils_rx.py`）
  - 修复 `apply_mixed_precision_bitwidth` 在设置对称量化时，`qmin` / `qmax` 没有正确更新的问题
  - 修复前：设置 `symmetric=True` 后，`qmin` / `qmax` 仍然保持非对称值（如 `0, 255`）
  - 修复后：正确更新为对称范围（如 8-bit: `-128, 127`；2-bit: `-2, 1`）
  - 影响范围：所有通过 JSON 配置文件设置 `input_symmetric` / `output_symmetric` 的量化器

### 🔧 改进

- 增强量化参数设置逻辑：
  - 位宽变更时自动根据对称性更新 `qmin` / `qmax`
  - 对称性变更时自动根据当前位宽更新 `qmin` / `qmax`
  - 确保量化器状态始终一致

## [1.2.1] - 2024-12-23

### ✨ 新增功能

- 添加混合精度位宽配置工具函数
- 支持通过 JSON 配置文件灵活控制量化参数

## [1.2.0] - 2024-12-20

### ✨ 初始版本

- 基于 AIMET 官方版本构建
- 提供 PyTorch 和 ONNX 量化支持

<!--
版本对比链接（在使用 GitHub/GitLab 时填入对应仓库 URL，例如：

[Unreleased]: https://github.com/<org>/aimet-rx/compare/v1.3.8...HEAD
[1.3.8]: https://github.com/<org>/aimet-rx/compare/v1.3.7...v1.3.8
[1.3.7]: https://github.com/<org>/aimet-rx/compare/v1.3.6...v1.3.7
[1.3.6]: https://github.com/<org>/aimet-rx/compare/v1.2.2...v1.3.6
[1.2.2]:  https://github.com/<org>/aimet-rx/compare/v1.2.1...v1.2.2
[1.2.1]:  https://github.com/<org>/aimet-rx/compare/v1.2.0...v1.2.1
[1.2.0]:  https://github.com/<org>/aimet-rx/releases/tag/v1.2.0
-->
