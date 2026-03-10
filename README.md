# AIMET RX - AI Model Efficiency Toolkit (Redistributable Package)

这是一个包含 AIMET (AI Model Efficiency Toolkit) 核心组件的可重分发包。

## 包含的模块

- **aimet_common**: AIMET 的通用功能模块
- **aimet_onnx**: ONNX 模型的量化和优化工具
- **aimet_torch**: PyTorch 模型的量化和优化工具

## 更新日志

### v1.2.2 (2024-12-24)

#### 🐛 Bug 修复
- **修复对称量化配置问题** (`aimet_torch/utils_rx.py`)
  - 修复了 `apply_mixed_precision_bitwidth` 函数在设置对称量化时，`qmin`/`qmax` 没有正确更新的问题
  - 修复前：设置 `symmetric=True` 后，qmin/qmax 仍然保持非对称值（如 0, 255）
  - 修复后：正确更新为对称范围（如 8-bit: -128, 127；2-bit: -2, 1）
  - 影响范围：所有通过 JSON 配置文件设置 `input_symmetric` 和 `output_symmetric` 的量化器

#### ✨ 改进
- 增强了量化参数设置逻辑：
  - 位宽变更时自动根据对称性更新 qmin/qmax
  - 对称性变更时自动根据当前位宽更新 qmin/qmax
  - 确保量化器状态始终一致

### v1.2.1 (2024-12-23)

#### ✨ 新增功能
- 添加混合精度位宽配置工具函数
- 支持通过 JSON 配置文件灵活控制量化参数

### v1.2.0 (2024-12-20)

#### ✨ 初始版本
- 基于 AIMET 官方版本构建
- 提供 PyTorch 和 ONNX 量化支持

## 安装

使用 pip 安装生成的 whl 文件：

```bash
pip install aimet_rx-1.0.0-py3-none-any.whl
```

## 使用示例

### PyTorch 模型量化

```python
from aimet_torch.v2 import quantsim
from aimet_torch.utils_rx import apply_mixed_precision_bitwidth

# 1. 创建量化模拟器
sim = quantsim.QuantizationSimModel(
    model,
    dummy_input=sample_input,
    quant_scheme='percentile',
    config_file='quantsim_config.json',
    default_output_bw=8,
    default_param_bw=8
)

# 2. 应用混合精度位宽配置（可选）
stats = apply_mixed_precision_bitwidth(
    sim.model,
    config_file='bitwidth_config.json',
    verbose=True
)

# 3. 校准量化参数
import aimet_torch.v2 as aimet
with aimet.nn.compute_encodings(sim.model):
    for inputs, _ in calib_loader:
        sim.model(inputs)
```

### 混合精度配置示例 (JSON)

```json
{
  "layer_name_config": {
    "conv1": {
      "weight_bitwidth": 8,
      "input_bitwidth": 8,
      "output_bitwidth": 8,
      "input_symmetric": true,
      "output_symmetric": true
    }
  },
  "layer_type_config": {
    "QuantizedConv2d": {
      "weight_bitwidth": 4,
      "input_bitwidth": 4,
      "output_bitwidth": 4,
      "input_symmetric": true,
      "output_symmetric": true
    }
  }
}
```

### ONNX 模型量化

```python
from aimet_onnx import QuantizationSimModel

# 创建量化模拟器
sim = QuantizationSimModel(model, ...)
```

## 技术说明

### 对称量化 vs 非对称量化

| 特性 | 对称量化 | 非对称量化 |
|------|---------|-----------|
| **量化范围** | 8-bit: [-128, 127]<br>2-bit: [-2, 1] | 8-bit: [0, 255]<br>2-bit: [0, 3] |
| **零点偏移** | 固定为 0 | 需要计算 |
| **硬件友好性** | ✅ 更简单 | 需要额外支持 |
| **量化效率** | 可能浪费范围 | 充分利用范围 |

### 配置优先级

混合精度位宽配置的优先级（从高到低）：
1. `layer_name_config` - 精确匹配层名称
2. `layer_name_config` - 模式匹配（支持通配符 `*`）
3. `layer_type_config` - 按层类型匹配
4. `default_bitwidth` - 默认值

### 工具函数

- `apply_mixed_precision_bitwidth()`: 应用混合精度位宽配置
- `setup_percentile_calibration()`: 设置 Percentile 校准
- `apply_power_of_2_workflow()`: 应用 Power-of-2 量化
- `freeze_quantizer_parameters()`: 冻结量化器参数（QAT 准备）

## 构建和安装

### 构建 Wheel 包

```bash
cd aimet_rx
python -m build
```

### 安装

```bash
pip install dist/aimet_rx-1.2.2-py3-none-any.whl
```

## 许可证

BSD-3-Clause

## 版权

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.

---

**注意**: 本包基于 AIMET 官方版本进行定制和优化，增加了便捷的工具函数和 bug 修复。







