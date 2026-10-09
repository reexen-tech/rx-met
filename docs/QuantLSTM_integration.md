# QuantGRU 与 QuantLSTM 接入 rx-met

QuantLSTM 与 QuantGRU 一样，通过模型中的原生模块实例接入 AIMET v2。两个算子都
自行完成内部量化，AIMET 管理模型其余部分并提供统一校准、QAT 和导出入口。
`prepare_model()` 不会把任意 `nn.GRU` / `nn.LSTM` 自动转换为这些算子；先在模型中
显式替换模块并加载原权重，再调用模型准备和 QuantSim。

## GRU 接入分析与 LSTM 对应实现

| 模块 | 原有 GRU 接入 | LSTM 接入 |
| --- | --- | --- |
| `operators/` | 独立 C++/CUDA 核心、Python binding | `quant-lstm/`，保留自身配置及数值语义 |
| `model_preparer.py` | FX 把 QuantGRU 视为叶节点 | 同样保留 QuantLSTM，不展开原生执行 |
| `utils.py`、`onnx_utils.py` | 现有 GRU 图分析与映射 | 注册 LSTM 为 ConnectedGraph 叶节点及 ONNX LSTM 类型，保留三个输出间的关系 |
| `v2/nn/modules/custom.py` | `QuantizedQuantGRU` 透传包装 | `QuantizedQuantLSTM`，关闭通用输入、输出及权重量化器 |
| `aimet_common/quantsim_config/compute_ops.py` | 跳过 GRU 的通用量化默认值 | 同时跳过 LSTM，防止二次量化 |
| `utils_rx.py` | `GRU_config`、Po2、统计 | 同一入口分发配置/Po2，算子负责处理；保留独立计数 |
| `v2/quantsim/quantsim.py` | QuantScheme 映射及 Percentile | TF→MinMax、TF Enhanced→SQNR、Percentile 数值传入原生收集器 |
| `v2/nn.compute_encodings` | 同次前向收集 GRU 与普通层数据 | 同次收集 LSTM 两个方向，跳过已锁定参数，异常时恢复校准标志 |
| `staged_quantization_utils.py` | GRU AIMET encodings 加载 | LSTM 参数保存、加载、筛选、锁定；普通层输入编码也保存，保证阶段恢复完整 |
| `rx_export/export_onnx_json.py` | 标准 GRU 节点、编码回填 | 统一调用原生导出接口；算子负责 h/c、门与双向参数重排 |
| 构建与发布 | GRU wheel、镜像导入及 GPU 冒烟 | LSTM wheel、CUDA 变体版本、镜像导入及 QuantSim 校准/QAT 冒烟 |
| ONNX 直量化 | 忽略 PyTorch `GRU_config` | 同样忽略 `LSTM_config`；不在 ONNX PTQ 中重做原生 LSTM 内部校准 |

## 实现归属与统一接入

LSTM 修改先在独立 `quant-lstm` 仓库实现，再将源码同步至 `operators/quant-lstm/`。
专属逻辑集中于 `pytorch/lstm_aimet.py`，不依赖 AIMET：

- `load_bitwidth_config()` 解析 `LSTM_config`，与 GRU 使用同名配置入口。
- `calibration_context()` 管理收集、完成校准、锁定及异常恢复。
- `enable_pot2()` 负责本算子的 scale 转换与数值审计。
- `export_quant_params_to_aimet_format()` / `load_quant_params_from_aimet_format()`
  负责编码，工具包不解析 h/c 状态、内部算子和门顺序。
- `normalize_quant_lstm_onnx()` 负责节点及参数命名，共享 initializer 安全复制。

`src/aimet_torch/native_recurrent.py` 是 GRU/LSTM 的统一注册位置。模型准备、
ConnectedGraph、QuantSim 包装、配置/Po2、阶段参数、校准和导出调用该注册信息或
公共接口。新增同类算子无需在上述位置分别添加类型分支。

GRU 内部补充 `calibration_context()`、`enable_pot2()` 和 `owns_quantization`，
已有配置和编码接口保持兼容。其定点执行专用 `aimet_configure()` 不用于 PTQ 生命周期。
原生算子的 `owns_quantization=True` 使通用量化配置跳过这些模块，避免二次量化。

QuantLSTM 的 `load_quant_config()`、`percentile_value`、量化参数锁和安全复制行为
也在原仓库实现；复制模型保留已生成的参数 JSON，不复制 C++ 校准收集器，正在
校准时禁止复制。接口详情见
[算子集成文档](../operators/quant-lstm/docs/aimet_integration.md)。

## 使用与配置

支持单层、`dropout=0`、无 projection、CUDA float32，单向/双向及有/无 bias。
用户示例统一为 `examples/quick_start_kws.py`，使用真实 Speech Commands v0.02，
通过 `--rnn_type gru|lstm` 选择循环层，完成浮点训练、PTQ、Po2、QAT、导出和恢复。
默认每个训练阶段 1 epoch，用于学习流程；随机数据回归保留在 `tests/`。
运行条件、配置和制品说明随发布包放在 [examples/README.md](../examples/README.md)。

```bash
python3 examples/quick_start_kws.py --rnn_type lstm \
  --data-dir /datasets/speech_commands_v0.02 --output-dir /workspace/output/kws
```

两种模式共用 `examples/config/quick_start_full_quant.json`，LSTM 读取其中的独立节点：

```json
{
  "LSTM_config": {
    "use_quantization": true,
    "quant_config": {
      "schema_version": 1,
      "scale_mode": "affine",
      "operators": {
        "cell_state": {"bitwidth": 16},
        "weight_ih": {"bitwidth": 8, "granularity": "per_channel"}
      }
    }
  }
}
```

`quant_config` 使用 QuantLSTM 自身的 override 格式，未列出的字段使用算子默认值。
参数粒度是 `per_tensor`、`per_gate`、`per_channel`；量化点支持 8/16 位。
`LSTM_config` 缺省时保留模块设置，`use_quantization: false` 使用浮点路径。
配置需在校准前应用；相同配置可重复应用，已校准后修改配置会报错，必须先显式
`reset_calibration()`。量化参数按首次 PTQ 结果用于 QAT，不在训练循环中重新校准。

`apply_power_of_2_workflow()` 转换 LSTM 已校准 scale，更新对应实数范围，再通过
原生加载器重新派生执行参数并做数值审计。锁定参数不能直接修改 Po2；如需 Po2，
在保存并锁定参数之前执行转换，或在首次配置中指定 `scale_mode: "pot2"`。

## 导出和恢复约定

`export_onnx_json()` 输出无 Q/DQ 的标准 ONNX LSTM。ONNX 表达浮点运算，量化参数
写入配套 `.encodings`，顶层采用与 GRU 相同的 `schema_version: 3`：

- `activation_encodings[模块路径]` 标记 `is_LSTM: true`，包含输入、隐藏状态、cell
  state 及内部算子。输入列表顺序为 x/h0/c0，输出列表顺序为 sequence/h/c；编码包含
  `bitwidth` 和字符串 `is_symmetric`，与 GRU 相同。
- `internal_ops[量化点].output` 与 GRU 一样使用编码列表，反向量化点放在
  `internal_ops_reverse`；内部激活与参数编码分开。
- 双向 h/c 网格可以不同，以 `PER_CHANNEL` 一维数组表示，先正向 H 项再反向 H 项，
  不使用自定义 `forward` / `reverse` 字典。
- `param_encodings` 对齐实际 ONNX W/R/B initializer，四门由 PyTorch IFGO 重排到
  ONNX IOFC，原生 per-gate 参数已按行展开，仅 per-tensor 标量扩展至每行，按正向、反向顺序拼接为一维数组。
- B 合并 bias_ih/bias_hh；两者 dtype 与对称性必须相同，否则导出显式报错。
- `quant_lstm_encodings[模块路径]` 保存完整、未经重排的原生文档，作为参数回读依据。
  消费方需支持此 LSTM 编码结构；仅支持 GRU 的后端不能直接处理 LSTM。

`load_quantizer_encodings(..., allow_overwrite=False)` 恢复参数并启用 LSTM 量化，
后续共享校准跳过已锁定 LSTM。`reset_calibration()` 同时清除参数锁。
`save_quantizer_encodings()` 保存的阶段检查点同样包含原生 LSTM 文档。
阶段保存调用公共编码接口的 `for_onnx=False` 模式，不施加 ONNX 合并 bias 的限制。
导出测试用同一个校验器检查 GRU/LSTM 部署记录，并核对一维参数数组与实际 ONNX
W/R/B 的尺寸；原生检查点回读另行测试。
这些文件不保存模型权重，权重应与 encodings 配套保存和恢复。

## 验证

```bash
python3 -m unittest -v tests/test_dependencies.py
python3 -m unittest -v tests/test_quant_lstm_integration.py
bash operators/quant-lstm/tools/run_cpu_only_package_check.sh /tmp/quant-lstm-cpu
```

GPU 测试使用真实的 QuantSim、CUDA 算子和 ONNX Runtime，覆盖 FX 叶节点、通用
量化器隔离、混合精度、单/双向 PTQ/QAT、h/c 梯度、Percentile、Po2、失败清理、
参数锁、阶段恢复和 ONNX 回读，以及 GRU/LSTM 同模型校准与导出。
