# 循环算子接入 rx-met

QuantGRU 和 QuantLSTM 自行完成内部量化，AIMET 管理模型其余部分，并提供统一的
校准、QAT 和整模型导出入口。本文说明两种算子在 rx-met 中的调用顺序、框架协作和
部署约束；算子自身的配置字段、编码布局和方法语义由对应接口参考维护。

| 当前任务 | 阅读入口 |
| --- | --- |
| 在已安装的 rx-met 中接入自己的模型 | [PyTorch 模型量化](PyTorch_quantization.md)，循环层要求见下文 |
| 运行 KWS 或比较 CPU/GPU PTQ | [运行示例](../examples/README.md) |
| 单独使用或从源码构建循环算子 | [QuantGRU](../operators/quant-gru/README.md)、[QuantLSTM](../operators/quant-lstm/README.md) |
| 修改 rx-met 与算子的集成实现 | 本文的框架接入点、实现归属和验证章节 |

## 使用与配置

通过应用包安装的用户已经具备对应的算子 wheel，无需为整模型量化再次独立编译。
PyTorch 模型中的循环层需要显式替换：`prepare_model()` 保留已有 QuantGRU /
QuantLSTM，不自动转换任意 `nn.GRU` / `nn.LSTM`。

| 约束 | QuantGRU / QuantLSTM |
| --- | --- |
| 结构 | 单层、`dropout=0`、单向或双向、可选 bias；LSTM 不支持 projection |
| 数据 | 三维 CPU/CUDA float32 Tensor，布局与原模型一致；使用 `PackedSequence` 或无 batch 维的模型须先适配输入 |
| CPU | `eval()` 下的浮点前向、校准、PTQ、Po2、保存恢复和 ONNX 导出；实际反向传播报错 |
| CUDA | 浮点训练、校准、PTQ 和 QAT |
| 安装 | 使用与设备匹配的 CPU/CUDA wheel；CUDA binding 不隐式回退到 CPU |

上述可选 bias 指常规 PyTorch 前向。当前 QuantGRU 的 ONNX 导出要求 `bias=True`，
即使关闭量化也不能导出无 bias GRU；QuantLSTM 支持无 bias 导出。需要部署无 bias
GRU 时，可在创建 QuantSim 前改用 `bias=True` 并将两个 bias 参数置零，验证浮点等价性；
这仍会引入 bias 量化点，需重新校准并验证量化结果。

普通 CPU 前向无需额外关闭梯度；推荐 `torch.no_grad()` 或
`torch.inference_mode()` 减少推理开销。已有 ONNX 的原生循环层 PTQ 另有输入限制，
见 [ONNX GRU/LSTM 适配](ONNX_PTQ_COMPILER.md#gru--lstm-原生适配)。

以下片段将已经加载浮点权重的单个 `nn.GRU` / `nn.LSTM` 转为对应原生算子，
保留权重、方向、bias、布局、设备、参数冻结状态和训练模式。`model.encoder.rnn` 代表模型中实际的
循环层属性；如果有多个循环层，需要逐一替换。

```python
import torch
from torch import nn
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM

def replace_recurrent_layer(original):
    if type(original) not in (nn.GRU, nn.LSTM):
        raise TypeError("expected nn.GRU or nn.LSTM")
    if original.num_layers != 1 or original.dropout != 0:
        raise ValueError("requires num_layers=1 and dropout=0")
    if isinstance(original, nn.LSTM) and original.proj_size != 0:
        raise ValueError("LSTM projection is unsupported")
    weight = original.weight_ih_l0
    if weight.dtype != torch.float32:
        raise ValueError("requires float32 weights")
    module_type = QuantGRU if isinstance(original, nn.GRU) else QuantLSTM
    replacement = module_type(
        original.input_size, original.hidden_size,
        num_layers=1, bias=original.bias, batch_first=original.batch_first,
        dropout=0, bidirectional=original.bidirectional,
    ).to(device=weight.device, dtype=weight.dtype)
    replacement.load_state_dict(original.state_dict(), strict=True)
    for name, parameter in original.named_parameters():
        replacement.get_parameter(name).requires_grad_(parameter.requires_grad)
    replacement.train(original.training)
    return replacement

model.encoder.rnn = replace_recurrent_layer(model.encoder.rnn)
```

替换在模型准备、QuantSim 和 optimizer 创建之前完成。先在浮点模式下用固定输入
检查输出、状态和任务指标，再开始校准；CPU 前向前需切换到 `eval()`。不同实现的浮点累加
不保证逐位相同。

整模型按以下顺序接入：

1. 创建与原模型形状、方向、bias 和布局一致的 QuantGRU / QuantLSTM，严格加载浮点权重。
2. 调用 `prepare_model()`，再创建 `QuantizationSimModel`。原生循环层保持叶节点，
   通用量化器不会重复包裹其输入、输出和参数。
3. 校准前应用混合精度 JSON。普通层读取 `layer_type_config` 等规则，循环层分别读取
   `GRU_config` / `LSTM_config`；配置结构与默认值见[量化配置](Quant_config.md)和
   [示例配置](../examples/config/README.md)。
4. 使用同一次 `compute_encodings` 前向校准整模型。量化方案和 Percentile 值传入
   原生收集器，已锁定参数跳过收集，异常时恢复校准标志。
5. 如需在校准后统一转换到 Po2，调用 `apply_power_of_2_workflow()`，在冻结参数和 QAT 前完成转换。
   该入口分别处理普通量化器与原生循环层；模型只有循环层时仍执行并验证原生 Po2。
   原生算子也可由自身配置选择直接按 Po2 校准；整模型转换开关不会改变其初始校准模式。
6. CPU 路径直接评估和导出；CUDA 路径可冻结量化参数后执行 QAT，沿用已校准网格。
7. 配套保存主权重和编码，重建模型后比较相同输入的输出及任务指标。

GRU 与 LSTM 的配置接口并不相同。LSTM 使用原生 sparse override，已校准后改变
配置需要先显式 `reset_calibration()`；GRU 已有参数时会跳过位宽配置加载。
完整方法语义见 [GRU 配置](../operators/quant-gru/pytorch/config/README.md)和
[LSTM 集成接口](../operators/quant-lstm/docs/aimet_integration.md)。

## 导出和恢复约定

`export_onnx_json()` 生成标准 ONNX GRU/LSTM 与公共 schema 4 `.json`，二者必须
按同次导出结果配套使用。ONNX 表达浮点计算；整数部署还需要消费编码并支持相应
内部量化点。编码文件不保存主权重，也不等同于算子独立的原生参数检查点。

整模型保存、检查点类型和严格恢复步骤见[导出与恢复](Export_and_reload.md)。
循环层的端口编码、双向参数排列、h/c 状态网格、bias 合并和原生格式区别集中在
[循环算子编码接口](../operators/quant-lstm/docs/aimet_integration.md#编码与-onnx)；
开关筛选和 `allow_overwrite=False` 的锁定语义见
[量化开关与阶段恢复](../operators/quant-lstm/docs/aimet_integration.md#量化开关与阶段恢复)。

恢复时沿用相同的模型结构、准备方式和配置。只替换 JSON 而继续使用语义不同的旧版
算子库不能保证等价；源码升级及旧版 LSTM 窄范围编码的迁移条件见
[LSTM 升级说明](../operators/quant-lstm/docs/installation.md#21-升级已有安装)和
[编码兼容性](../operators/quant-lstm/docs/aimet_integration.md#编码与-onnx)。

## 框架接入点（维护者）

| 模块 | GRU 接入 | LSTM 接入 |
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
| ONNX 直量化 | 标准 GRU 通过原生适配器读取 `GRU_config` 并校准内部量化点 | 标准 LSTM 同样读取 `LSTM_config`，复用原生校准、Po2、编码和恢复接口 |

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

GRU 提供 `calibration_context()`、`enable_pot2()` 和 `owns_quantization`，
与现有配置和编码接口协作。其定点执行专用 `aimet_configure()` 不用于 PTQ 生命周期。
原生算子的 `owns_quantization=True` 使通用量化配置跳过这些模块，避免二次量化。

QuantLSTM 的 `load_quant_config()`、`percentile_value`、量化参数锁和安全复制行为
也在原仓库实现；复制模型保留已生成的参数 JSON 及 Po2 转换所需的校准范围，
不复制 C++ 校准收集器，正在校准时禁止复制。接口详情见
[算子集成文档](../operators/quant-lstm/docs/aimet_integration.md)。

## Bias 编码差异

普通层与原生循环层采用不同的 bias 编码方案。部署前应确认目标编译器和硬件是否支持
所用的 bias 网格及重缩放规则。

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

### 固定 bias 网格的训练余量

当前两种原生算子对新校准的 `bias_ih` / `bias_hh`，在各自量化组内采用：

```text
M = max(abs(bias_min), abs(bias_max))
Sb = max(2 * M, 0.01) / qmax
```

Po2 模式再将该 scale 严格向上取为 2 的幂。此策略扩大的是独立 bias 编码范围，
不会放大实际浮点 bias；目的是为固定网格下的 QAT 更新预留空间，代价是步长变粗。
新 CPU 校准/PTQ 也复用该策略，但 CPU 不因此支持训练。已导入或锁定的编码
不会被自动扩张，QAT 期间也不会持续扩大范围。

该规则适用于 MinMax、Percentile 和 SQNR；bias 不使用直方图裁剪或常数 bin 扩展范围。
观测范围至少扩大一倍，零 bias 获得至少 ±0.01 的可训练区间。8 位下最小 affine
步长约为 7.87e-5，Po2 步长约为 1.22e-4，可表示 1e-4 量级的更新。位宽、对称
零点和部署算术保持原有约定。余量不能保证任意学习率或训练时长下都不饱和，应同时
检查偏置饱和、梯度和任务指标；如需其他范围策略，可加载经过独立验证的显式编码。

若采用 INT32 且 `Sb = Sx * Sw` 的直接累加方案，通常应检查该网格下的 bias
可表示范围，而不是另按 bias 观测值推导独立 scale。低位宽独立 bias 加训练余量
并不等价于该方案。算子数值测试验证的是所配置网格下的执行结果，不能证明目标
部署规范接受独立 bias scale。

### 统一 bias 编码的待决问题

以下设计问题记录于 2026-10-26：

- 目标编译器/硬件是否要求原生 GRU/LSTM 的 bias 为 INT32 且使用对应分支的乘积 scale。
- CPU 纯 PTQ 是否应与 QAT 区分训练余量策略，并验证精度差异。
- 如统一 bias 方案，配置类型、校准、重缩放、参数导入导出和测试需要怎样同步。

相关实现见 [普通层 bias 对齐](../src/aimet_torch/utils_rx.py)、
[ONNX bias 对齐](../src/aimet_onnx/rx_ptq/po2.py)、
[GRU 整数线性计算](../operators/quant-gru/src/gru_forward_cpu_quant.cc)、
[LSTM 执行参数](../operators/quant-lstm/src/lstm/lstm_execution_params.cc) 和
[LSTM bias 校准](../operators/quant-lstm/src/lstm/quant_params.cc)。

## 验证

在源码仓库根目录、匹配的 rx-met Python 环境中运行：

```bash
python3 -m tests release -k dependencies -v
python3 -m tests recurrent examples -v
```

GPU 测试使用真实的 QuantSim、CUDA 算子和 ONNX Runtime，覆盖 FX 叶节点、通用
量化器隔离、混合精度、单/双向 PTQ/QAT、h/c 梯度、Percentile、Po2、失败清理、
参数锁、阶段恢复和 ONNX 回读，以及 GRU/LSTM 同模型校准与导出。共享回归还覆盖
关闭量化、校准后关闭量化、两种算子混合启用、严格加载缺失参数，以及 GRU 非对称
Po2 与直接 Po2 校准的对照。

独立验证 LSTM 的 CPU C++ 包时，按 [CPU-only reference package](../operators/quant-lstm/docs/installation.md#4-安装-cpu-only-reference-package)
准备 CMake 和 C++ 工具链，再从源码仓库根目录运行下列命令。它在指定目录中编译、测试、安装并运行一个外部消费者：

```bash
bash operators/quant-lstm/tools/run_cpu_only_package_check.sh /tmp/quant-lstm-cpu
```
