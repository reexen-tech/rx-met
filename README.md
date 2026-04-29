# AIMET RX - AI Model Efficiency Toolkit (Redistributable Package)

这是一个包含 AIMET (AI Model Efficiency Toolkit) 核心组件的可重分发包。

## 包含的模块

- **aimet_common**: AIMET 的通用功能模块
- **aimet_onnx**: ONNX 模型的量化和优化工具
- **aimet_torch**: PyTorch 模型的量化和优化工具

## 更新日志

完整版本历史见 [CHANGELOG.md](./CHANGELOG.md)。最新版本：**v1.3.7 (2026-04-29)** ——
修复 1.3.6 wheel 中 `export_onnx_and_encodings` 子模块的 import 路径错误。

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
pip install dist/aimet_rx-1.3.7-py3-none-any.whl
```

## 许可证

BSD-3-Clause

## 版权

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.

---

**注意**: 本包基于 AIMET 官方版本进行定制和优化，增加了便捷的工具函数和 bug 修复。







