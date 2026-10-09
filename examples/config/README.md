# 示例量化配置

KWS 的 GRU 和 LSTM 模式共用两份 JSON。先在 `QuantizationSimModel` 中使用基础配置，
再通过 `apply_mixed_precision_bitwidth` 应用位宽配置，最后进行校准。
运行方法见 [示例说明](../README.md)。

| 文件 | 用途 |
| --- | --- |
| `mrnn_quantsim_config_custom_mixed_precision_v2.json` | 基础量化规则：是否量化、对称性、per-channel |
| `quick_start_full_quant.json` | 普通层位宽，以及 `GRU_config`、`LSTM_config` |

`--rnn_type` 决定模型使用的循环算子。两个配置节可以同时放在完整 JSON 中，
只有模型中对应的算子会读取自己的配置。修改后重新运行 PTQ，QAT 循环中不重新校准。

## 普通层

`layer_type_config` 控制卷积、全连接等普通层。类型名对应 QuantSim 模型中的类名，
例如 `QuantizedConv2d`、`QuantizedLinear`；它不控制 GRU / LSTM 的内部量化。

匹配优先级：`layer_name_config` 精确名 → `layer_type_config` → 名称通配符 → 默认配置。
`default_bitwidth` 设置普通层的默认权重和输出位宽，`disable_quantization` 控制普通层开关。

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
- 位宽支持 8、16。权重和偏置固定对称量化，可选三种参数粒度；激活和状态使用 per-tensor，可设置 `is_symmetric`、`is_unsigned`。
- 未列出的量化点沿用算子默认配置，默认位宽为 8。门计算可以通过 `input_gate_input`、`forget_gate_input`、`cell_gate_input`、`output_gate_input` 及其 `_output` 配置；`cell_tanh_output` 控制细胞状态经过 tanh 后的量化点。
- `scale_mode` 控制首次 PTQ 的 scale。本 KWS 示例第 5 步还会调用公共 Po2 转换接口，QAT 沿用转换后的固定参数。
- ONNX 将两组 bias 合并导出，因此 `bias_ih` / `bias_hh` 位宽应一致。

`comment` 字段用于配置说明。配置在校准前生效；已有量化参数时改变配置需先清除旧校准结果。
恢复模型时，使用同次保存的权重与 encodings，并保持网络类型和配置一致。
