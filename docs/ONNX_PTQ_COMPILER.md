# ONNX PTQ 编译器产物

这条流程通过 `aimet_onnx` 直接读取已有 ONNX。两种框架都使用
`export_onnx_json` 入口名，分别位于 `aimet_onnx.rx_ptq` 和
`aimet_torch.rx_export`，共用 `aimet_common.compiler_encodings` 中的格式和校验规则。
PyTorch 保留原参数、默认值及两个字符串路径的元组返回值；ONNX PTQ 返回包含
`onnx`、`encodings`、`aimet_encodings` 路径、`coverage` 和 `onnx_shapes` 的字典。
旧 ONNX 入口 `export_onnx_compiler_artifacts` 保留原参数及字典返回值。

## 术语

- **Clean ONNX**：移除 AIMET 仿真量化节点后的部署模型。模型保留原始输入输出
  契约，量化参数由外部 encodings 文件提供。
- **AIMET 原始 encodings**：由 `aimet_onnx` 导出的原生量化参数，用于调试和重载。
- **编译器 encodings**：目标编译器直接读取的节点级量化参数，包含节点输入、输出和
  参数映射。公共格式由 `src/aimet_common/compiler_encodings.py` 实现，各框架适配器
  负责源量化网格及节点映射。
- **校准清单**：记录校准样本、模型输入名、样本配对关系和文件校验值的可复现输入
  清单。
- **Po2 scale**：通过 `cover_range` 规则选择并实际写入量化器的 `1/2^n` scale。
- **Bias scale**：Conv、Gemm 和 MatMul 的 bias 使用 `Sb = Sx * Sw`，默认采用
  INT32 位宽。
- **编译器覆盖率**：所有已启用的量化器都映射到编译器 encodings 的节点输入、
  节点输出或参数条目，每个 encoding 键都对应模型图中的对象。

标准入口与 `examples/quick_start_kws.py` 的使用方式对应：

```bash
python3 /examples/quick_start_onnx_ptq.py
```

## 运行前提

- Linux x86_64，CPython 3.10（cu118/cu126）或 3.12（cu130）；
- 已安装与 CUDA 变体匹配的 ONNX Runtime GPU；
- 默认使用 `CUDAExecutionProvider`，需要 CPU 调试时显式设置
  `RX_MET_ONNX_PTQ_DEVICE=cpu`；
- 已构建 rx-met 的 ONNX native runtime；
- 正式制品使用真实的代表性数据完成校准。

## MobileNetV2

通用量化入口只接收用户准备好的 ONNX 和 NPY/NPZ 校准数据，不执行模型专属预处理。
复现 MobileNetV2 + ImageNette 公开演示时，模型目录 `/models`、数据目录 `/datasets`
保持只读，在可写工作目录中执行以下两步：

```bash
python3 /examples/prepare_onnx_ptq_data.py \
  --onnx /models/mobilenetv2-12.onnx \
  --parquet /datasets/imagenette2-320.parquet
python3 /examples/quick_start_onnx_ptq.py
```

准备脚本仅适用于这组公开模型和数据。它先验证输入，再将静态模型和 NPY 分别写入
当前运行目录的 `output/onnx_ptq_quick_start/models/`、`datasets/`，保留原始文件。
量化示例默认读取这些准备结果，两步在同一运行目录执行，无需设置输入路径变量。
可选 `--out-root DIR` 指定另一可写根目录，并打印第二步所需的模型和校准路径。
用户自己的模型和数据由用户按相应协议处理后，直接通过 `RX_MET_ONNX_PTQ_MODEL`、
`RX_MET_ONNX_PTQ_CALIB` 指定输入，无需调用此准备脚本。

输入名从 ONNX 图自动读取。多输入模型可用同一 sample 前缀的 npy，或改用 npz。
产物默认写入当前运行目录下的 `output/onnx_ptq_quick_start/`，可用
`RX_MET_ONNX_PTQ_OUTPUT_DIR` 覆盖。

命令退出码为 0，日志输出“量化流程完成”及 metadata 路径，并生成下述四类文件，
即表示标准示例完成。`<prefix>.meta.json` 中的 `coverage`、`all_power_of_2` 和
`clean_onnx_vs_fp32` 用于进一步确认制品完整性和输出一致性。

## 输出

为兼容已有 PyTorch 编译器格式，普通算子记录的 `zero_point` 是加性偏移：
`real = scale * (q + zero_point)`，与标准 ONNX Q/DQ 零点的符号相反。
ONNX 适配器将 `TfEncoding.offset` 转为对应整数域的加性偏移；例如有符号 INT8
的 `-128` 转为 `0`，INT32 bias 的 `-2147483648` 转为 `0`。
非对称量化保留实际偏移，不能统一置零。无符号编码使用 `UINT8` 等明确 dtype。
参数编码的通道字段统一为数组，单通道也保留数组结构。
原生 GRU/LSTM 的专用编码保留算子库拥有的字段、列表结构和网格约定，不由通用层重写。

每次运行生成：

- `<prefix>.onnx`：移除 AIMET `QcQuantizeOp` 并显式保存中间张量 shape/dtype 的 clean ONNX；
- `<prefix>.encodings`：compiler-native schema，包含 node-level `input/output`
  和 parameter encodings；
- `<prefix>.aimet.encodings`：AIMET 原始 encodings；
- `<prefix>.meta.json`：输入 shape、校准数量、Po2 变化、覆盖率和 FP32 对比。

导出自动补全 `graph.value_info`，不改写计算节点或权重。动态维度保留符号，
标量使用合法的零维 shape；不能推导的 dtype、rank 或维度直接报错。
产物回读先检查磁盘文件中已保存的标注，再执行量化恢复；检查过程不会补写缺失信息。
`onnx_shapes` 记录张量数、中间张量数及符号维度数，示例和通用入口会将其写入 metadata。

默认量化规则与本项目的 PyTorch 导出流程保持一致：

- 校准：`percentile=99.99`，在 `compute_encodings` 前设置。Torch/ONNX 共用 `aimet_common.percentile` 的 2048 桶动态直方图、样本计数和百分位取点，后续批次超出范围时扩容并合并历史统计；默认 100 保留完整观测范围。
- 激活和权重使用对称量化；Percentile PTQ 按 JSON2 的 `bias_bitwidth` 统计校准普通 bias，全零通道也保留有效网格；Po2 阶段再按 `Sb = Sx * Sw` 推导，默认位宽为 INT32。旧原生统计路径仍延后 bias 对齐，避免近零通道产生 `scale=0`。
- Power-of-2 使用与 `src/aimet_torch/power_of_2_quantization.py` 相同的 `cover_range`（容差 2%）
- 权重默认 per-channel；Gemm 保持 per-channel；无 Conv+Relu supergroup

ONNX 适配器仅在校准期间使用临时 session 采集中间张量，并将公共模块计算出的网格
写回原生量化器。公共模块使用 Torch 张量运算，统计与范围计算跟随 CPU/CUDA provider，
避免跨设备标量运算的末位差异。观测值经主机缓冲区传输，增加校准阶段的传输和内存开销，
每批处理后释放观测值，不保存全部样本。
推理和导出仍使用原 session 与原图。支持 `session.run` 的迭代器、回调和校准上下文入口；
校准期间暂不支持 IO binding、OrtValue 或异步执行。原生 GRU 内部仍调用 QuantGRU 自身的校准接口。

原生 CPU/CUDA QDQ 共用居中整数网格计算，nearest 使用 ties-to-even。
INT32 不经过 float32 的 `2^31` 平移，避免小整数码丢失；这需要重新构建原生库，
encoding 的 scale、offset/zero_point 和保存格式保持不变。非 Po2 scale 使用正确舍入的
倒数乘法；最小范围保护仅在公共 Percentile 模块按目标整数域应用一次。
CUDA provider 默认设置 `use_tf32=0`，使校准、评价和恢复使用完整 FP32 矩阵计算；
显式传入的 provider 选项保持优先。

配置与 `examples/quick_start_kws.py` 相同：

- JSON 1：`examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json`
- JSON 2：`examples/config/quick_start_full_quant.json`；图中存在标准 GRU 时，`GRU_config` 交由 QuantGRU 原生接口处理，其余已知但图中不存在的类型配置忽略。

`src/aimet_common/quantsim_config/onnx_ptq_compiler_w8a8.json` 仍可作为 `--config` 覆盖项。

输入经过 Transpose/Reshape 等数据搬运算子时，JSON2 输入设置作用于上游有效量化网格。
`QuantizedMatMul` 的 `input_bitwidth` / `input_bitwidths` 包括常量矩阵，默认使用逐张量
输入网格；常量仍存入 `param_encodings`。显式 `weight_*` 字段保留参数配置的优先级；
只使用 `QuantizedLinear` 映射 MatMul 时，权重仍与激活输入分开配置。共享网格收到
不兼容的显式请求时，在修改量化器前报错。
常量 MatMul 输入在统计校准期间透传，普通权重仍执行参数 QDQ；同一共享量化器不能
同时承担这两种校准角色。

## GRU 原生适配

`create_onnx_ptq_sim` 自动识别标准 GRU，调用 `quant_gru` 算子包中的 ONNX 适配器。
门内统计、量化前向、Po2、编码打包和恢复均由现有 QuantGRU 负责；普通层继续使用
ONNX QuantSim。GRU 后的普通计算层保留自己的输入网格，用于混合精度和 bias 对齐。
这些输入在 Gather/Reshape 等操作之后采样，不在整个 GRU 输出上追加通用 QDQ；
普通计算分支后的 Concat 单独共享输入网格，保留各分支的输出网格。
运行时输入别名仅用于仿真和原始编码重载，编译器编码映射回原节点输入端口，clean ONNX 不增加节点。

运行时需要包含 `quant_gru_onnx` 和 `quant_gru_onnx_runtime` 的更新版 QuantGRU wheel。
桥接节点仅存在于仿真图；导出仍保存原标准 ONNX GRU、权重及张量名称，并合并
`is_GRU`、双向 `internal_ops` 和原生参数编码。原始 AIMET encodings 附加
`native_recurrent_encodings`，恢复时缺少门内信息、参数或方向信息会报错。

当前支持 CUDA float32、常量 W/R/B、`linear_before_reset=1`、默认 Sigmoid/Tanh、
单向前向或双向、layout 0/1，以及可选的初始状态。无 bias、变长 `sequence_lens`、
自定义激活、clip、反向单向和可覆盖或共享的权重暂不支持，进入校准前报错。
桥接使用主机缓冲区与原生 CUDA 算子交换数据，会产生额外拷贝，不是零拷贝执行。

## 限制

- 该流程适用于 PTQ；QAT 使用 PyTorch 流程。
- 校准数据必须与模型输入名称、shape 和 dtype 一致。
- 编译器 encodings 是 rx-met 的部署格式，必须与同次导出的 Clean ONNX 配套使用。
- CPU 模式用于普通 ONNX 模型调试；原生 GRU 校准要求 CUDA，发布验证在目标 CUDA 环境执行。
