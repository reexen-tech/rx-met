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

- 校准：`percentile=99.99`（必须在 `compute_encodings` 前设置，native 默认 100 等于 min-max）
- 激活和权重使用对称量化；Conv/Gemm/MatMul 的 bias scale 在激活和权重校准后按 `Sb = Sx * Sw` 推导，默认位宽为 INT32。该流程绕过 AIMET ONNX 对近零 bias 通道生成的 `scale=0`，防止校准期 fake-quant 向后续激活传播 Inf 或全 0。
- Power-of-2 使用与 `src/aimet_torch/power_of_2_quantization.py` 相同的 `cover_range`（容差 2%）
- 权重默认 per-channel；Gemm 保持 per-channel；无 Conv+Relu supergroup

配置与 `examples/quick_start_kws.py` 相同：

- JSON 1：`examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json`
- JSON 2：`examples/config/quick_start_full_quant.json`（`Quantized*` / `GRU_config` 在当前 ONNX 图里没有则忽略）

`src/aimet_common/quantsim_config/onnx_ptq_compiler_w8a8.json` 仍可作为 `--config` 覆盖项。

## 限制

- 该流程适用于 PTQ；QAT 使用 PyTorch 流程。
- 校准数据必须与模型输入名称、shape 和 dtype 一致。
- 编译器 encodings 是 rx-met 的部署格式，必须与同次导出的 Clean ONNX 配套使用。
- CPU 模式用于调试，发布验证在目标 CUDA 环境执行。
