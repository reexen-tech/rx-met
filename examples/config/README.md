# 示例量化配置

KWS 和 ONNX PTQ 示例默认使用本目录的两份 JSON。示例先按基础配置创建
`QuantizationSimModel`，再应用位宽配置，最后进行校准。运行方法见[示例说明](../README.md)，
下文的普通层匹配优先级适用于 KWS 使用的 PyTorch 路径。

| 文件 | 用途 |
| --- | --- |
| [`mrnn_quantsim_config_custom_mixed_precision_v2.json`](mrnn_quantsim_config_custom_mixed_precision_v2.json) | 基础量化规则：是否量化、对称性、per-channel |
| [`quick_start_full_quant.json`](quick_start_full_quant.json) | 普通层位宽，以及 `GRU_config`、`LSTM_config` |

KWS 的 `--rnn-type` 决定模型使用的循环算子。两个配置节可以同时放在完整 JSON 中，
模型中的循环算子只读取各自的配置。可用 `--quant-config` 指定另一份位宽配置；改动量化参数后
重新校准，KWS 的 QAT 循环沿用 PTQ 和 Po2 阶段得到的参数。

## 普通层

`layer_type_config` 控制卷积、全连接等普通层。类型名对应 QuantSim 模型中的类名，
例如 `QuantizedConv2d`、`QuantizedLinear`；它不控制 GRU / LSTM 的内部量化。

匹配优先级：`layer_name_config` 精确名 → `layer_type_config` → 名称通配符 → 默认配置。
`default_bitwidth.weight` 为未匹配普通层设置权重位宽，`default_bitwidth.output`
同时设置其输入和输出位宽；省略字段保留现有值。匹配规则按整条选择，不逐字段继承
低优先级规则。普通层位宽字段接受 2 到 32 的整数；`default_config.disable_quantization`
控制未匹配普通层，单条规则的 `disable_quantization` 控制该规则匹配的层。

## GRU 与 LSTM 的区别

| 内容 | `GRU_config` | `LSTM_config` |
| --- | --- | --- |
| 量化开关 | `default_config.disable_quantization` | `use_quantization` |
| 内部算子配置 | `operator_config` | `quant_config.operators` |
| 参数粒度字段 | `quantization_granularity` | `granularity` |
| 参数粒度值 | `PER_TENSOR` / `PER_GATE` / `PER_CHANNEL` | `per_tensor` / `per_gate` / `per_channel` |
| Po2 初始配置 | `default_config.use_pot2_scale` | `quant_config.scale_mode` 为 `affine` / `pot2` |
| 循环状态 | 隐藏状态 `output` | 隐藏状态 `output`、细胞状态 `cell_state` |

GRU 示例的各量化点和说明已列在完整 JSON 的 `GRU_config` 中。
LSTM 使用独立的原生配置格式，不能只把 GRU 配置节改名后复用。

两种算子的所有量化点默认均为对称量化，新建实例省略算子条目或 `is_symmetric`
不会启用非对称量化；激活量化点仍可显式配置 `is_symmetric=false`。
sigmoid 门输出的默认无符号设置与对称性独立。

## LSTM 配置示例

以下配置节放在完整 JSON 顶层，与 `layer_type_config`、`GRU_config` 同级：

```json
{
  "LSTM_config": {
    "use_quantization": true,
    "quant_config": {
      "schema_version": 1,
      "scale_mode": "affine",
      "operators": {
        "input": {"bitwidth": 8},
        "output": {"bitwidth": 8},
        "cell_state": {"bitwidth": 16},
        "weight_ih": {"bitwidth": 8, "granularity": "per_channel"},
        "weight_hh": {"bitwidth": 8, "granularity": "per_channel"}
      }
    }
  }
}
```

- `input` 是输入序列；`output` 是隐藏状态 h，同时作为序列输出和下一时间步输入。
- `cell_state` 是细胞状态 c。本示例设置为 16 位，用于演示混合位宽；是否需要调整应通过任务精度评估决定。
- `weight_ih` / `weight_hh` 分别为输入权重、循环权重；`bias_ih` / `bias_hh` 为对应偏置。
- 权重和偏置使用有符号对称量化，支持 2 到 32 位及三种参数粒度。激活和状态使用 per-tensor，可设置 `is_symmetric`、`is_unsigned`；有符号对称模式支持 2 到 32 位，有符号非对称模式支持 1 到 32 位，无符号模式支持 1 到 31 位。`UINT32` 和有符号对称 `INT1` 会报错。
- 未列出的量化点沿用算子默认配置，默认位宽为 8。门计算可以通过 `input_gate_input`、`forget_gate_input`、`cell_gate_input`、`output_gate_input` 及其 `_output` 配置；`cell_tanh_output` 控制细胞状态经过 tanh 后的量化点。
- `scale_mode` 控制首次 PTQ 的 scale。KWS 示例在校准后还会调用公共 Po2 转换接口，QAT 沿用转换后的固定参数。
- rx-met 的阶段保存和 ONNX 导出均将两组 bias 合并，因此 `bias_ih` / `bias_hh` 位宽应一致。

`comment` 字段用于配置说明。配置在校准前生效；已有量化参数时改变配置需先清除旧校准结果。
恢复模型时，使用同次保存的权重与 encodings，并保持网络类型和配置一致。

## MFCC 前处理与 Log 量化边界

当前 KWS 示例在分类网络外以浮点计算 MFCC；QuantSim 校准和导出的 ONNX 接收
`[1,49,20]` 特征，不包含 MFCC 中的 Clamp/Log。配置中的相应类型规则仅在模型内
存在这些模块时生效。恢复或部署时仍需使用检查点保存的前处理参数。

若将 MFCC 放回量化图，Log 前的低位宽 QDQ 可能把 epsilon 和弱信号舍入为零；
Log 输出量化又可能将负无穷截为有限值，因此最终输出有限不能证明内部定义域和
小信号保真正确。外置浮点前处理不证明整数 MFCC 满足这一约束，量化前处理需要
单独验证其数值行为和任务精度。
