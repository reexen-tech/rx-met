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
也在原仓库实现；复制模型保留已生成的参数 JSON 及 Po2 转换所需的校准范围，
不复制 C++ 校准收集器，正在校准时禁止复制。接口详情见
[算子集成文档](../operators/quant-lstm/docs/aimet_integration.md)。

## 使用与配置

支持单层、`dropout=0`、无 projection、CPU/CUDA float32，单向/双向及有/无 bias。
CPU 支持校准、PTQ、Po2、编码保存恢复及 ONNX 导出，要求 `eval()`，普通前向无需
额外关闭梯度；推荐使用 `torch.no_grad()` 或 `torch.inference_mode()` 减少推理开销。
实际跨 CPU 算子的反向传播会明确报错，浮点训练与 QAT 继续使用 CUDA。
CPU 算子 wheel 通过 `RX_MET_ENABLE_CUDA=0` 构建，不能用仅含 CUDA binding 的 wheel 代替。
用户示例统一为 `examples/quick_start_kws.py`，使用真实 Speech Commands v0.02，
通过 `--rnn_type gru|lstm` 选择循环层，完成浮点训练、PTQ、Po2、QAT、导出和恢复。
默认每个训练阶段 1 epoch，用于学习流程；随机数据回归保留在 `tests/`。
CPU 使用 `--device cpu --ptq-only --fp-checkpoint /path/to/model_fp_kws.pth`；
共享样本及 CPU/GPU 制品比较命令见 [示例说明](../examples/README.md)。
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
参数粒度是 `per_tensor`、`per_gate`、`per_channel`；有符号量化点支持 1–32 位，
无符号支持 1–31 位，有符号对称量化及权重、偏置从 2 位起。位宽不改变 FP32 载体
和 JSON 格式；高位宽仍遵循既有浮点精度报告与 exact 模式约束。
`LSTM_config` 缺省时保留模块设置，`use_quantization: false` 使用浮点路径。
配置需在校准前应用；相同配置可重复应用，已校准后修改配置会报错，必须先显式
`reset_calibration()`。量化参数按首次 PTQ 结果用于 QAT，不在训练循环中重新校准。

`apply_power_of_2_workflow()` 检查已启用循环层的校准状态，调用算子的 `enable_pot2()`，
再通过公共编码导出同步并验证实际 Po2 网格和内部编码覆盖。原生循环层独立于普通
AIMET 量化器处理；普通量化器为空时仍执行原生转换，比较报告在执行与验证之后生成。
LSTM 按新 scale 重算非对称 zero point，并更新 `real_min/real_max`，再通过原生
加载器重新派生执行参数并做数值审计。校准后优先使用内存中保留的校准下界；
从 encoding 文件加载后使用已有量化网格下界，文件中不增加校准诊断字段。
具体公式和示例见[算子 Po2 转换约定](../operators/quant-lstm/docs/aimet_integration.md#已校准参数的-po2-转换)。

锁定的 affine 参数不能直接转换为 Po2；如需 Po2，在保存并锁定参数之前执行转换，
或在首次配置中指定 `scale_mode: "pot2"`。更新后的校准接口需要配套重建 QuantLSTM
extension / wheel，并更新运行镜像，见[升级说明](../operators/quant-lstm/docs/installation.md#21-升级已有安装)。

普通 signed symmetric MinMax 校准采用 `scale=max_abs/qmax`、`zero_point=0`，其中 8 位的
`qmax=127`、16 位的 `qmax=32767`。运行时允许完整有符号范围：INT8 `[-128,127]`、
INT16 `[-32768,32767]`；CPU/CUDA、QAT 饱和掩码和导出范围均遵循此约定。
`real_min=(qmin-zero_point)*scale`、`real_max=(qmax-zero_point)*scale`，与 GRU 一致。
旧版 LSTM 编码若使用窄范围下界，需保留 scale/zero_point 并按完整范围重新导出，
同时更新算子库；仅替换 JSON 而沿用旧运行库会造成训练与部署语义不一致。

## Bias 编码差异

记录日期：2026-10-26。状态：待讨论。本节记录当前实现与常见整数部署方案的差异，
不作统一方案的决定；现有代码、位宽配置、校准及编码导入导出行为保持不变。

### 乘积 scale 与独立 bias scale

矩阵乘法的中心化整数累加器使用 `Sacc = Sx * Sw`；逐通道权重对应
`Sacc[c] = Sx * Sw[c]`。若 bias 直接量化到该域，则 `Sb[c] = Sacc[c]`，
通常使用 INT32，量化后的 bias 可直接与整数累加结果相加。

当前仓库存在两条不同路径：

| 项目 | 普通 Conv/Linear 等的 bias 对齐路径 | 原生 QuantGRU/QuantLSTM |
| --- | --- | --- |
| bias 编码 scale | 对齐后为 `Sx * Sw` | 根据 bias 自身范围独立计算 `Sb` |
| bias 量化位宽 | 对齐后默认有符号 32 位，可显式指定并检查范围 | 由算子配置控制的有符号 2–32 位；当前 KWS 示例两种 RNN 均配置为 8 位 |
| 加入累加器前 | 同一 scale，可直接相加 | 先将 bias 从 `Sb` 重缩放到 `Sacc` |
| Po2 工作流 | 启用 bias 对齐时执行乘积 scale 和 bias 位宽对齐 | 调用算子自身的 `enable_pot2()`，不自动改为 INT32 或乘积 bias scale |

普通层在统计 PTQ 阶段仍可按配置校准 bias；上表的乘积 scale 指启用
`align_bias_scale=True` 后的对齐结果，不能泛化为所有阶段始终使用乘积 scale。
PyTorch 工作流的 `align_bias_scale` 默认关闭，ONNX 工作流默认开启；启用对齐后
`bias_bitwidth` 默认值为 32，当前 KWS 示例显式开启对齐。
PyTorch 的该对齐覆盖 Conv/Linear，ONNX 的对应路径覆盖带 bias 的
Conv/ConvTranspose/Gemm/MatMul，不覆盖原生 GRU/LSTM 内部 bias。

原生循环层有输入和循环两条线性分支，概念关系为：

```text
Sacc_ih[c] = Sx * Sw_ih[c]
Sacc_hh[c] = Sh * Sw_hh[c]

q_bias_ih = Quantize(bias_ih, Sb_ih, bias_bitwidth)
bias_acc_ih[c] = Rescale(q_bias_ih[c], Sb_ih[c] -> Sacc_ih[c])
linear_ih = Rescale(acc_ih + bias_acc_ih, Sacc_ih -> Slinear_ih)
```

循环分支同样先把 `bias_hh` 转换到 `Sacc_hh`。实际重缩放使用算子定义的
M+shift 或 Po2 编码。bias 编码位宽、物理存储载体和累加器宽度是不同概念；
用 int32/FP32 容器保存低位宽整数码不会增加其量化精度。独立低位宽 bias
可能先丢失信息，随后重缩放到更宽的累加域不能恢复这些信息。

### 训练余量及待讨论事项

当前两种原生算子对新校准的 `bias_ih` / `bias_hh`，在各自量化组内采用：

```text
M = max(abs(bias_min), abs(bias_max))
Sb = max(2 * M, 0.01) / qmax
```

Po2 模式再将该 scale 严格向上取为 2 的幂。此策略扩大的是独立 bias 编码范围，
不会放大实际浮点 bias；目的是为固定网格下的 QAT 更新预留空间，代价是步长变粗。
新 CPU 校准/PTQ 也复用该策略，但 CPU 不因此支持训练。已导入或锁定的编码
不会被自动扩张，QAT 期间也不会持续扩大范围。

若采用 INT32 且 `Sb = Sx * Sw` 的直接累加方案，通常应检查该网格下的 bias
可表示范围，而不是另按 bias 观测值推导独立 scale。低位宽独立 bias 加训练余量
并不等价于该方案。现有 native 测试预期与余量策略的同步问题，也不能替代部署
规范是否允许独立 bias scale 的确认。

后续再讨论以下事项，当前不修改实现：

- 目标编译器/硬件是否要求原生 GRU/LSTM 的 bias 为 INT32 且使用对应分支的乘积 scale。
- CPU 纯 PTQ 是否应与 QAT 区分训练余量策略，并验证精度差异。
- 如统一 bias 方案，配置类型、校准、重缩放、参数导入导出和测试需要怎样同步。

相关实现见 [普通层 bias 对齐](../src/aimet_torch/utils_rx.py)、
[ONNX bias 对齐](../src/aimet_onnx/rx_ptq/po2.py)、
[GRU 整数线性计算](../operators/quant-gru/src/gru_forward_cpu_quant.cc)、
[LSTM 执行参数](../operators/quant-lstm/src/lstm/lstm_execution_params.cc) 和
[LSTM bias 校准](../operators/quant-lstm/src/lstm/quant_params.cc)。

## 导出和恢复约定

`export_onnx_json()` 输出无 Q/DQ 的标准 ONNX LSTM。ONNX 表达浮点运算，量化参数
写入配套 `.json`，顶层使用公共编译器格式 `schema_version: 4`，所有记录采用
`real = scale * (q - zero_point)`。独立算子阶段文件仍使用原有 schema 3；
`version: "1.0.0"` 来自 AIMET 原有编码版本设置。LSTM 不新增专用顶层格式：

- `activation_encodings[模块路径]` 标记 `is_LSTM: true`。与 GRU 一致，外层 `input`
  和 `output` 各保存一条正向 `PER_TENSOR` 标量编码，不枚举全部输入输出端口；
  编码包含 `bitwidth` 和字符串 `is_symmetric`。
- `internal_ops[量化点].output` 与 GRU 一样使用编码列表，反向量化点放在
  `internal_ops_reverse`；内部激活与参数编码分开。
- h0/h 共用各方向的 `output` 量化点，c0/c 共用 `cell_state` 量化点，其编码分别保存在
  `internal_ops` 和 `internal_ops_reverse`。每个方向内固定为 `PER_TENSOR` 标量，
  正反向可使用不同的 scale；状态和激活不支持按隐藏通道独立量化，不展开成通道数组。
- `param_encodings` 对齐实际 ONNX W/R/B initializer，四门由 PyTorch IFGO 重排到
  ONNX IOFC，原生 per-gate 参数已按行展开，仅 per-tensor 标量扩展至每行，按正向、反向顺序拼接为一维数组。
- 有 bias 时，B 合并 bias_ih/bias_hh；两者 dtype 与对称性必须相同，否则导出显式报错。
  `bias=False` 时 ONNX 省略可选 B 输入，`param_encodings` 也不包含 bias；真实 bias 即使全零仍需编码。
- 导出和回读只使用上述公共区段，不输出 `quant_lstm_encodings`、`model_info`、
  `execution_metadata` 或原生区段内的 `schema_version: 1`。消费方需支持 LSTM 本身的
  内部量化点；公共编码容器与 GRU 一致。

`load_quantizer_encodings(..., allow_overwrite=False)` 成功加载当前层的有效 encoding
后，启用该层量化并锁定参数；后续共享校准跳过已锁定层。
`reset_calibration()` 同时清除参数锁和旧校准范围。
`save_quantizer_encodings()` 也使用与 GRU 相同的公共参数布局，遵守相同的 bias 合并限制。
回读从公共编码拆分正反向参数、还原 IFGO 门顺序；展开的参数按 per-channel 恢复。
量化网格与数值输出保持一致，原始 per-tensor / per-gate 分组标签不作为部署元数据保存。
全部 scale 为 2 的幂时恢复 Po2 模式，cuBLAS 模式沿用目标模块设置。
算子独立的 `export_quant_params()` / `load_quant_params()` 仍保存原生检查点及执行信息，
该格式与 rx-met 面向编译器的导出明确分开。
导出测试比较实际 GRU/LSTM 的顶层字段，检查所有嵌套层级均无原生元数据，核对一维
参数数组与 ONNX W/R/B 尺寸，并直接从公共编码验证恢复输出。

GRU/LSTM 关闭量化时，即使已完成校准，阶段文件与部署 encodings 均不写入该层的
量化参数，ONNX 中仍保留对应浮点节点。恢复时先应用相同的阶段配置，再加载权重和 encodings；
关闭量化的层允许没有 encoding，严格加载仍会检查已启用层是否缺少参数。
显式加载文件中已有的有效 GRU/LSTM encoding 仍会启用该层量化。
这些文件不保存模型权重，权重应与 encodings 配套保存和恢复。
算子自身的显式 `export_quant_params_to_aimet_format()` 不负责开关筛选，
`load_quant_params_from_aimet_format()` 不改变量化开关；阶段开关策略统一由 rx-met 实现。

## 验证

```bash
python3 -m tests release -k dependencies -v
python3 -m tests recurrent examples -v
bash operators/quant-lstm/tools/run_cpu_only_package_check.sh /tmp/quant-lstm-cpu
```

GPU 测试使用真实的 QuantSim、CUDA 算子和 ONNX Runtime，覆盖 FX 叶节点、通用
量化器隔离、混合精度、单/双向 PTQ/QAT、h/c 梯度、Percentile、Po2、失败清理、
参数锁、阶段恢复和 ONNX 回读，以及 GRU/LSTM 同模型校准与导出。共享回归还覆盖
关闭量化、校准后关闭量化、两种算子混合启用、严格加载缺失参数，以及 GRU 非对称
Po2 与直接 Po2 校准的对照。
