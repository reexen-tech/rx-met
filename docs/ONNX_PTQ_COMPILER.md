# ONNX PTQ 编译器产物

通过 `aimet_onnx` 可以直接校准已有 ONNX 模型、将量化 scale 对齐到 Power-of-2，
并导出供目标编译器配套使用的 clean ONNX 和 `<prefix>.json`。已有模型和预处理数据
可直接按下文量化；初次体验可从 [MobileNetV2 示例](#mobilenetv2) 开始。

## 术语

- **Clean ONNX**：移除 AIMET 仿真量化节点后的部署模型。模型保留原始输入输出
  契约，量化参数由外部 encodings 文件提供。
- **AIMET 原始 encodings**：由 `aimet_onnx` 导出的原生量化参数，用于调试和重载。
- **编译器 encodings**：目标编译器直接读取的节点级量化参数，包含节点输入、输出和
  参数映射。公共格式由 `src/aimet_common/compiler_encodings.py` 实现，各框架适配器
  负责源量化网格及节点映射。
- **数据清单**：示例准备脚本在 `manifest.json` 记录输入形状、抽样和前处理信息，
  并将验证文件名与标签配对。使用自备数据校准时不要求提供该文件。
- **Po2 scale**：通过 `cover_range` 规则选择并实际写入量化器的 `1/2^n` scale。
- **Bias scale**：Conv、Gemm 和 MatMul 的 bias 使用 `Sb = Sx * Sw`，默认采用
  INT32 位宽。
- **编译器覆盖率**：所有已启用的量化器都映射到编译器 encodings 的节点输入、
  节点输出或参数条目，每个 encoding 键都对应模型图中的对象。

## 运行前提

- Linux x86_64，CPython 3.10（cpu/cu118/cu126）或 3.12（cu130）；
- 按[用户使用指南](User_guide.md#3-加载并启动镜像)准备匹配的 rx-met 应用和 runtime，
  并进入容器。应用自带 AIMET ONNX 原生库，源码构建见[原生运行库说明](AIMET_ONNX_NATIVE.md)。
- `quick_start_onnx_ptq.py` 默认选择 CUDA。CPU runtime 上指定 `--device cpu` 或
  `RX_MET_ONNX_PTQ_DEVICE=cpu`。若模型包含 GRU/LSTM，另见[原生适配的运行时要求](#gru--lstm-原生适配)。
- 量化命令在可写目录执行。正式制品使用具有代表性的真实数据校准。

库接口 `create_onnx_ptq_sim()` / `run_onnx_ptq()` 的 `providers` 默认是
`("CPUExecutionProvider",)`。直接调用库进行 CUDA PTQ 时须显式传入 CUDA provider；
示例脚本的设备默认值不改变库接口的默认值。

## 量化自己的 ONNX 模型

通用入口接收已经完成前处理的模型输入，模型路径、校准目录和设备通过命令行明确指定。
先把模型和数据挂载到容器内（示例为 `/models` 和 `/datasets`），再在可写工作目录执行：

```bash
python3 /opt/rx-met/examples/quick_start_onnx_ptq.py \
  --model /models/model.onnx \
  --calib-dir /datasets/my_model/calib \
  --device cpu \
  --prefix my_model --output-dir output/my_model
```

CUDA runtime 将 `--device cpu` 改为 `--device cuda`。模型输入的每个维度必须为静态正整数，
每份样本包含一次前向所需的全部输入；如果模型声明了 batch 维度，shape 必须包含它。
模型专属的裁剪、归一化、
音频特征提取等由调用方完成。

单输入模型可每份样本保存一个 NPY，文件名不限；多输入模型推荐每份样本保存一个
NPZ，键必须与 ONNX 输入名完全一致，不得缺失或增加键。多输入 NPY 的配对依赖
加载器的文件名规则，例如输入 `input` / `audio_data` 对应同一前缀的
`00000_input_fp32.npy` / `00000_audio_data_fp32.npy`；任意输入名使用 NPZ 可避免
文件名匹配歧义。单文件样本按文件名排序，配对 NPY 按样本前缀排序；目录存在 NPZ 时只加载 NPZ。

加载器检查 shape 和有限数值，并按模型声明的 dtype 转换：只接受 NumPy
`same_kind` 转换，整数输入还检查取值范围。准备数据时应直接使用模型要求的 dtype，
避免转换导致精度变化或溢出；字符串、对象和非有限值不能作为校准输入。

需要自定义配置时，传入 `--config-file /workspace/config/base.json` 和
`--quant-config /workspace/config/bitwidth.json`。两份配置的职责与规则见
[量化配置参考](Quant_config.md)，默认配置见后文的配置说明。

脚本默认查找校准目录旁的 `val/`，可用 `--validation-dir /datasets/my_model/val`
或 `RX_MET_ONNX_PTQ_VAL` 指定其他目录。显式指定的目录必须存在且有效；默认目录不存在时，
脚本复用校准样本比较输出。验证目录使用与校准相同的文件格式。对分类模型，若在验证目录的
上一级提供 `manifest.json`，用 `val_files`（文件名）与 `val_labels`（每个文件中的样本标签，按
文件顺序排列）提供标签，脚本会用第一个 `[batch, classes]` 输出计算 Top-1；未提供标签时只做数值比较。
清单按文件名配对，每个文件对应一个标签，因此此方式适用于 batch-1 样本。
正式精度评估应使用与校准数据分离的验证集及适合模型的任务指标。

命令完成后使用终端返回的 ONNX、JSON 和 metadata 路径，按[输出](#输出)核对制品。
仅需复现公开演示时，按下面的 MobileNetV2 数据准备步骤运行。

## MobileNetV2

通用量化入口只接收用户准备好的 ONNX 和 NPY/NPZ 校准数据，不执行模型专属预处理。
公开模型、数据的下载与两步命令由[MobileNetV2 示例](../examples/README.md#onnx-ptqmobilenetv2)
维护。模型目录 `/models`、数据目录 `/datasets` 保持只读，准备和量化在同一可写工作目录执行。
准备脚本仅适用于这组公开模型和数据。它先验证输入，再将静态模型和 NPY 分别写入
当前运行目录的 `output/onnx_ptq_quick_start/models/`、`datasets/`，保留原始文件。
量化示例默认读取这些准备结果，两步在同一运行目录执行，无需设置输入路径变量。
可选 `--out-root DIR` 指定另一可写根目录，并打印第二步所需的模型和校准路径。
自有模型不需要此准备脚本；按[量化自己的 ONNX 模型](#量化自己的-onnx-模型)准备并指定输入即可。

输入名从 ONNX 图自动读取。多输入文件格式与命名规则见上文的输入要求。
示例与 `run_onnx_ptq` 的数据加载器要求模型输入具有静态的正整数维度；动态或缺失的
输入维度会在校准前报错。导出时可保留中间张量的符号维度。
默认输出根目录为当前运行目录下的 `output/onnx_ptq_quick_start/`，可用
`RX_MET_ONNX_PTQ_OUTPUT_DIR` 覆盖。每次量化在该根目录下创建北京时间运行目录，
格式为 `YYYYMMDD_HHMMSS_ffffff`，例如 `20261027_105308_820401`。
产物写入其中的 `exports/`，日志和运行信息写入 `run.log`、`run.json`。
准备好的 `models/`、`datasets/` 继续在根目录复用。

命令退出码为 0，日志输出“量化流程完成”及 metadata 路径，并生成下述配套文件，
即表示标准示例完成。`<prefix>.meta.json` 中的 `coverage`、`all_power_of_2` 和
`clean_onnx_vs_fp32` 用于进一步确认制品完整性和输出一致性。

## 输出

将同次导出的 `<prefix>.onnx` 与 `<prefix>.json` 一起交给目标编译器。调用代码应使用
导出接口返回的路径；目录中的 `.encodings` 和 `.aimet.encodings` 是 AIMET 原始编码。

示例的 `exports/` 包含：

- `<prefix>.onnx`：移除 AIMET `QcQuantizeOp` 后的 clean ONNX，保存中间张量的 shape 和 dtype；
- `<prefix>.json`：编译器的节点输入、输出和参数编码；
- `<prefix>.aimet.encodings`：用于重载的 AIMET 原始编码；
- `<prefix>.encodings`：底层 QuantSim 导出保留的 AIMET 原始编码；
- `<prefix>.meta.json`：输入 shape、校准数量、覆盖率、Po2 检查及各阶段的输出比较；
- `outputs.npz`：FP32、PTQ 和 Po2 阶段的评估输出，键名为 `<stage>_output<index>`。

导出时自动补全 `graph.value_info`。中间张量可以保留符号维度，标量使用零维 shape；
无法推导 dtype、rank 或维度时会报错。重载检查使用磁盘中已有的标注。

ONNX 的 `aimet_onnx.rx_ptq.export_onnx_json()` 返回 `onnx`、`encodings`、`aimet_encodings`
路径及 `coverage`、`onnx_shapes` 报告；原入口 `export_onnx_compiler_artifacts()` 保留原参数与返回值。
`run_onnx_ptq()` 则返回这三条路径和 `metadata` 路径，覆盖率与 shape 报告写入 metadata。
库函数未指定 `validation_dir` 时使用校准数据评估；它不会自动查找同级 `val/`。
PyTorch 同名导出入口及其元组返回值见[导出与重载](Export_and_reload.md)。

## 编码格式与验收

编译器 encodings 使用 `schema_version=4`，普通算子和原生 GRU/LSTM 均采用标准零点：
`real = scale * (q - zero_point)`，即实数 0 对应的整数码为 `zero_point`。
公共网格由 `aimet_common.quantization_grid.IntegerGrid` 表示，显式记录 dtype、
实际 qmin/qmax、scale 与标准零点，不按对称性选择整数符号。
Torch 原生量化器内部 offset 转为 `-offset`；原始 AIMET 文件和 ONNX 的
`TfEncoding.offset` 属于无符号索引域，转为 `qmin - offset`。
例如对称 INT8 的原始文件 `offset=-128` 与 `qmin=-128`
得到 `zero_point=0`，INT32 bias 同理。原生 GRU/LSTM 已使用标准零点，保持原值。
非对称量化保留实际零点，不能统一置零；无符号编码使用 `UINT8` 等明确 dtype。
当前编译器只支持对称量化，输入编译器的配置应保持对称；此修正不扩大编译器的量化支持范围。

旧文件（缺少 schema_version 或版本为 1/2/3）的普通算子使用加性零点，原生循环算子
已采用减性零点。公共规范化及 Torch 编码重载按版本迁移普通记录，重复处理 schema 4
不会再次取反。Torch 新的原始 legacy 记录通过 `offset_domain=unsigned_index`、
`signed` 和 qmin/qmax 标注索引域；旧 RX 目标域 offset 仍由兼容路径读取。
原始 Torch 编码重载使用保存的 scale/offset 精确恢复网格；无域标记的旧记录依据
保存的 scale、offset 与真实边界验证唯一解释，不一致或存在不同有效解释时拒绝。
ONNX 原始文件保存 `signed`，严格重载检查其与量化器配置一致；旧文件没有该字段时
保留本次显式配置的整数符号。编译器文件沿用 schema 4，只保留标准 zero_point；
非完整整数范围另保存 qmin/qmax，避免重载时扩大截断范围。
参数编码的通道字段统一为数组，单通道也保留数组结构。
普通算子和原生 GRU/LSTM 的整数记录均包含 `enc_type` 与 `n`，n 的形状与 scale 一致。
Po2 scale 满足 `scale = 2^(-n)`；非 Po2 scale 的 n 是舍入指数，实际网格以 scale 为准。
顶层 `input/output` 使用端口编号字典，如 `{"0": {...}}`。原生 GRU/LSTM 的
`internal_ops` 和 `internal_ops_reverse` 仍采用 `算子名 -> output -> [encoding]`，
其网格及字段由算子库生成，不由通用层重写。循环层加载接口兼容旧端口列表及缺少 n 的文件。

新产物验收直接读取保存的 schema v4 JSON，拒绝缺失字段、重复键、非法端口和
不符合整数网格公式的端点，不先迁移或补写。导出端使用公共
`finalize_compiler_encodings` 按 `IntegerGrid` 的 double 精度计算 real_min/max；
该函数不用于磁盘文件验收。旧文件加载仍使用兼容规范化接口。

在已配置 Python 依赖的源码仓库根目录运行：

```bash
python3 scripts/verify_compiler_artifacts.py \
  --onnx /path/to/model.onnx \
  --encodings /path/to/model.json
```

默认要求 Po2 和完整计算输入覆盖。已有 affine 产物使用 `--allow-affine`；
有意只验证已声明编码时使用 `--allow-partial`，其结果不表示完整量化覆盖。
独立脚本没有 live 模型来源信息，遇到无法从最终图证明端口的虚拟容器会明确拒绝。
它验证文件契约和图映射，不代替下游编译器执行验证。

普通参数的逐通道数量按 Conv、ConvTranspose、Gemm、MatMul 的已知通道轴或一维
参数长度验证；无法确定轴及尚未支持的普通分块/逐门声明会明确拒绝，不通过维度猜测放行。
原生循环参数采用自身的方向和 gate 打包规则单独验证。

双向 GRU/LSTM 的 `output[0]` 是 forward 兼容摘要，完整方向网格保存在各自的内部记录。
两个方向不同的网格可随未重新量化的保形路径保留，不能把摘要传播为整张张量的标量网格。
独立的输出 QDQ 或普通计算输入 QDQ 会建立新的消费端网格；严格图验收会拒绝错误的标量声明。

`onnx_shapes` 报告记录张量数、中间张量数及符号维度数。

## 默认配置与校准行为

ONNX 示例与 `run_onnx_ptq()` 默认使用以下两份配置，与 KWS 示例共用源文件：

- [基础量化配置](../examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json)。
- [混合精度配置](../examples/config/quick_start_full_quant.json)。模型含标准 GRU/LSTM 时，`GRU_config` / `LSTM_config` 由对应原生算子处理；已知但图中不存在的类型配置会忽略。

调用 `create_onnx_ptq_sim()` 时由调用方传入 `config_file`。该接口创建仿真模型，
应用混合精度、校准和 Po2 转换是后续步骤。基础配置也可选用
[`onnx_ptq_compiler_w8a8.json`](../src/aimet_common/quantsim_config/onnx_ptq_compiler_w8a8.json)。

示例的默认规则：

- 在 `compute_encodings` 前设置 `percentile=99.99`。Torch 和 ONNX 共用 `aimet_common.percentile` 的 2048 桶动态直方图、样本计数和百分位取点；后续批次超出范围时扩容并合并历史统计。设为 100 时保留完整观测范围。
- 激活和权重使用对称量化；Percentile PTQ 按 JSON2 的 `bias_bitwidth` 统计校准普通 bias，全零通道也保留有效网格；Po2 阶段再按 `Sb = Sx * Sw` 推导，默认位宽为 INT32。旧原生统计路径仍延后 bias 对齐，避免近零通道产生 `scale=0`。
- 默认整数类型固定为 INT；可通过 JSON2 `*_signed=false` 选择 UINT。整数符号与 `*_symmetric` 独立，UINT 对称模式将负值截为零，不自动退回 INT。
- Power-of-2 使用与 `src/aimet_torch/power_of_2_quantization.py` 相同的 `cover_range`（容差 2%）
- 权重默认 per-channel；Gemm 保持 per-channel；无 Conv+Relu supergroup

上述普通 bias 的乘积 scale 对齐不覆盖原生 GRU/LSTM。循环层内部 bias 当前按
独立配置的 bias 网格量化并重缩放到累加器域，不能将其默认视为 INT32 乘积 scale
编码。两种方案的差异和待确认的部署约束见
[Bias 编码差异（待讨论）](QuantLSTM_integration.md#bias-编码差异)。

ONNX 适配器仅在校准期间使用临时 session 采集中间张量，并将公共模块计算出的网格
写回原生量化器。公共模块使用 Torch 张量运算，统计与范围计算跟随 CPU/CUDA provider，
避免跨设备标量运算的末位差异。观测值经主机缓冲区传输，增加校准阶段的传输和内存开销，
每批处理后释放观测值，不保存全部样本。
推理和导出仍使用原 session 与原图。支持 `session.run` 的迭代器、回调和校准上下文入口；
校准期间暂不支持 IO binding、OrtValue 或异步执行。原生 GRU/LSTM 内部分别调用
QuantGRU/QuantLSTM 自身的校准接口。

原生 CPU/CUDA QDQ 共用居中整数网格计算，nearest 使用 ties-to-even。
INT32 不经过 float32 的 `2^31` 平移，避免小整数码丢失。该行为由原生库实现，
从旧版源码安装升级时须同步重建原生库；匹配的应用包已经包含原生库。
该内核规则不改变 scale、offset/zero_point 或保存格式。非 Po2 scale 使用正确舍入的
倒数乘法；最小范围保护仅在公共 Percentile 模块按目标整数域应用一次。
CUDA provider 默认设置 `use_tf32=0`，使校准、评价和恢复使用完整 FP32 矩阵计算；
显式传入的 provider 选项保持优先。

输入经过 Transpose/Reshape 等数据搬运算子时，JSON2 输入设置作用于上游有效量化网格。
`QuantizedMatMul` 的 `input_bitwidth` / `input_bitwidths` 包括常量矩阵，默认使用逐张量
输入网格；常量仍存入 `param_encodings`。显式 `weight_*` 字段保留参数配置的优先级；
只使用 `QuantizedLinear` 映射 MatMul 时，权重仍与激活输入分开配置。共享网格收到
不兼容的显式请求时，在修改量化器前报错。

网格传播只覆盖浮点数据输入及浮点输出。Gather 的 indices、Reshape 的 shape、
TopK/MaxPool 的索引输出、Dropout mask 和 Pad 控制参数不继承数据网格；模型的整数、
布尔输入输出也不会被强制添加量化器。导出检查量化器对应张量和端口的真实 dtype，
发现非浮点编码时报错；恢复验证会拒绝被追加到索引端口的 sidecar 编码。

常量 MatMul 输入在统计校准期间透传，普通权重仍执行参数 QDQ；同一共享量化器不能
同时承担这两种校准角色。

Torch 的 FX 参数与普通参数复用同一套量化器创建规则。选择 Percentile 时，展开 BN
和自定义模块的 weight/bias 同样按所设百分位校准。ONNX PTQ 按本次调用指定的方案
校准，不通过 ONNX metadata 强制覆盖参数策略。

## GRU / LSTM 原生适配

`create_onnx_ptq_sim` 自动识别标准 GRU 和 LSTM，分别调用 `quant_gru` / `quant_lstm`
算子包中的 ONNX 适配器，同一模型可同时包含两种循环层。门内统计、量化前向、Po2、
编码打包和恢复均由对应的原生算子负责；普通层继续使用 ONNX QuantSim。
循环层后的普通计算层保留自己的输入网格，用于混合精度和 bias 对齐。
这些输入在 Gather/Reshape 等操作之后采样，不在整个循环层输出上追加通用 QDQ；
普通计算分支后的 Concat 单独共享输入网格，保留各分支的输出网格。
运行时输入别名仅用于仿真和原始编码重载，编译器编码映射回原节点输入端口，clean ONNX 不增加节点。

GRU 运行时需要包含 `quant_gru_onnx` 和 `quant_gru_onnx_runtime` 的 QuantGRU wheel；
LSTM 需要包含 `quant_lstm_onnx` 和 `quant_lstm_onnx_runtime` 的 QuantLSTM wheel。
含这些循环层的模型还要求设备与算子 wheel 匹配：CPU PTQ 使用 CPU wheel，CUDA PTQ
使用 CUDA wheel；CUDA binding 不提供 CPU 回退。仅切换 `--device` 或 provider
不能替代安装匹配的算子。
桥接扩展还需使用运行时支持的 ORT API；当前 cu118 默认 LSTM 构建的头文件不匹配，
需按 [LSTM 安装说明](../operators/quant-lstm/docs/installation.md#22-onnx-runtime-ptq-bridge)
显式选择匹配头文件并重建。
桥接节点仅存在于仿真图；导出仍保存原标准 ONNX GRU/LSTM、权重及张量名称，并合并
`is_GRU` / `is_LSTM`、双向 `internal_ops` 和原生参数编码。原始 AIMET encodings 附加
`native_recurrent_encodings`，恢复时缺少门内信息、参数或方向信息会报错。

两种适配器均支持 CPU/CUDA float32、常量权重、单向前向或双向、layout 0/1，
以及可选初始状态。GRU 要求常量 W/R/B、`linear_before_reset=1` 和默认 Sigmoid/Tanh，
不支持无 bias。LSTM 要求常量 W/R、可选常量 B 和默认 Sigmoid/Tanh/Tanh，支持无 bias，
`initial_h` / `initial_c` 可以独立省略，cell state 使用自己的量化网格。
变长 `sequence_lens`、自定义激活及其参数、clip、反向单向、可覆盖或共享的权重均不支持；
LSTM 还不支持 peephole 和耦合输入遗忘门（`input_forget=1`）。这些情况在校准前报错。
桥接使用主机缓冲区与对应设备的原生算子交换数据，CUDA 路径会产生额外设备拷贝。

## 限制

- 该流程适用于 PTQ；QAT 使用 PyTorch 流程。
- 校准数据必须与模型输入名称、shape 和 dtype 一致。
- 编译器 encodings 是 rx-met 的部署格式，必须与同次导出的 Clean ONNX 配套使用。
- CPU 支持普通 ONNX 算子和原生 GRU/LSTM 的 PTQ；CPU 算子不提供训练或 QAT。
