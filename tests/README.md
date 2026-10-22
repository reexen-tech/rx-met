# 测试

本目录保存跨源码目录和构建配置的项目级测试。各模块内部的专项测试继续放在对应模块
目录中。

## KWS PTQ 往返验证

`verify_kws_ptq_roundtrip.py` 从 KWS 浮点检查点开始，调用示例的模型、数据集和量化接口，
执行 PTQ、Po2 并导出，跳过 QAT；再将该 ONNX 作为输入重新 PTQ。
两轮使用相同 JSON 快照、相同顺序的真实校准样本和 batch size 1，并检查 PTQ 未改变浮点权重。
需要 CUDA 和包含 ONNX 桥接的新版本 QuantGRU。

```bash
python3 tests/verify_kws_ptq_roundtrip.py \
  --fp-checkpoint /path/to/model_fp_kws.pth \
  --data-dir /datasets/speech_commands_v0.02
```

默认每类取 20 条训练校准及验证样本，输出到运行目录的
`output/kws_ptq_roundtrip/<时间>/`。`torch_ptq/`、`onnx_ptq/` 保存两轮产物，
`comparison.json` 逐项记录 ONNX、编码、GRU 编码和验证结果，`report.md` 提供摘要。
`quantized_outputs.npz` 保存两框架 PTQ/Po2 的逐样本 logits；报告同时记录跨框架的
输出误差和预测差异，Po2 另从最终保存的编码恢复后与 Torch 逐样本核对。
产物不一致时仍保存完整报告并返回 1；`--report-only` 可只生成报告而返回 0。
已有输出目录拒绝覆盖，可用 `--output-dir` 指定新目录。

## 公共 Percentile 校准

`test_percentile_calibration.py` 使用真实量化器和 ORT session，核对 Torch/ONNX 的动态
直方图与网格，覆盖后续大值、批次顺序、常量和零、微小初始范围、逐通道与分块、
重复校准、冻结编码，以及回调异常后的 session 恢复。CUDA 可用时同时核对 CPU/CUDA。
另检查同一设备上最终 Percentile scale 与 Torch 逐值相等。

```bash
python3 -m unittest discover -s tests -p test_percentile_calibration.py -v
```

## 文件说明

`test_run_context.py` 检查内置运行记录：UTC 微秒目录、多进程冲突、配置快照、日志迁移、
终端和文件的独立详细程度、异常堆栈、成功清理和失败保留、历史运行保护、版本来源及模型/线程间的日志隔离。
`test_run_logging.py` 使用真实 Torch/ONNX 和循环算子核对详细日志、数值不变、导出路径及版本记录。
KWS 工作流测试同时检查新目录中的浮点/QAT 检查点、ONNX、编码和恢复结果。

`test_symmetric_defaults.py` 检查两种整数域下的对称构造默认值、Lazy 包装器、公共
JSON1 缺项回退，以及真实 Torch v2/ONNX 模型的默认网格和显式非对称覆盖。

`test_quantization_grid.py` 验证无框架依赖的 INT/UINT 范围、标准零点、两种源 offset
转换、通道数组和窄范围。`test_integer_domains.py` 覆盖 INT/UINT 与对称/非对称四种
组合的真实校准、JSON2 配置、精确重载、CPU/CUDA ORT 导出与恢复、原始 AIMET
协议互换、INT32 非零 offset、共享网格冲突，以及严格重载的符号检查。

`test_standard_zero_point.py` 检查 compiler schema 4 的统一减性零点、无版本及旧版本
混合普通/循环编码迁移、重复规范化、全部 UINT8 整数码和有符号多位宽网格等价，
并用真实 Torch 导出与重载验证非零零点，拒绝未知版本、字段冲突和错误符号。

`test_native_qdq_alignment.py` 使用实际原生量化器及 ORT CPU/CUDA，核对 INT32 小整数码、
逐通道、ties-to-even、非零 offset、非 Po2 倒数和 NaN/Inf 饱和；需使用新构建的 AIMET 原生库。
`test_onnx_ptq_alignment.py` 同时检查 Percentile PTQ 的 bias 启用状态和配置位宽，
以及 Po2 后 INT32 bias 的导出恢复。

```bash
python3 -m unittest discover -s tests -p test_native_qdq_alignment.py -v
```

`test_onnx_operand_precision.py` 验证经过 Transpose 的输入配置、常量 MatMul 的位宽和
输入编号、逐张量网格、Linear/显式权重配置优先级，以及共享网格冲突的原子拒绝。
覆盖真实校准、Po2 和 compiler encodings 导出：

```bash
python3 -m unittest discover -s tests -p test_onnx_operand_precision.py -v
```

`test_onnx_input_quantization.py` 检查 Concat 的独立共享输入网格、校准顺序、配置冲突和
编码重载；`test_onnx_gru_native.py` 覆盖 GRU 中间帧按实际消费位置校准及端口编码导出。
`test_onnx_encoding_tensor_roles.py` 验证网格只沿浮点数据端口传播，覆盖 Gather 整数索引、
BOOL 控制输入、TopK/MaxPool 索引、Dropout mask、NonZero、浮点 Pad 控制和 Split 多输出，
以及非法量化器与被篡改索引端口编码的导出/恢复拒绝。
`test_fx_parameter_quantization.py` 检查展开 BN、自定义参数与普通 Conv 使用一致的校准
方案，覆盖字符串/枚举、CPU/CUDA、百分位值调整后重新校准、整数/FP8/FP16，以及无
参数校准 metadata 特殊适配的 ONNX 再校准、导出和恢复。

`test_dependencies.py` 验证依赖管理配置和生成结果。该测试检查以下内容：

- `pyproject.toml` 的兼容范围覆盖全部发布变体。
- `docker/variants.json` 与生成的 Bake 配置和 lock 文件保持一致。
- 每个 CUDA 变体的运行环境锁完整且自包含。
- 构建工具锁与 manifest 保持一致。
- 缺失 Torch 必需包时配置校验会失败。
- wheelhouse checksum 兼容标准的相对路径格式。

该测试执行本地依赖配置一致性检查。网络下载和 Docker 镜像构建由对应发布验证流程
覆盖。修改依赖范围、变体矩阵、锁生成或 wheelhouse 校验逻辑后执行：

```bash
python3 -m unittest -v tests/test_dependencies.py
```


## 循环算子与客户示例

`test_prepare_onnx_ptq_data.py` 使用真实的小型 ONNX/parquet 文件，在 CPU 上验证
模型、图片和标签检查、模型与数据分离的默认输出布局、样本预处理、重复生成的旧
样本清理以及原始文件保护。需要 runtime 中的 ONNX、PyArrow、Pillow 和 NumPy：

```bash
python3 -m unittest -v tests/test_prepare_onnx_ptq_data.py
```

`test_quant_lstm_integration.py` 使用随机小模型验证 LSTM 的校准、实际 QAT 更新、
参数锁、Po2、GRU/LSTM 共存和 ONNX 编码。`test_kws_example.py` 调用客户 KWS
主入口，使用小尺寸网络和随机音频验证 GRU/LSTM 两种模式的完整流程、导出节点及
检查点保护；另检查校准数据来自训练集且关闭增强。随机数据不用于评价任务精度。

在安装了 rx-met、QuantGRU、QuantLSTM 的 CUDA 环境中执行：

```bash
python3 -m unittest -v tests/test_quant_lstm_integration.py tests/test_kws_example.py
```

面向发布镜像的真实 Speech Commands 数据验收使用
`scripts/release/verify_kws_example.sh`，默认分别运行 GRU 和 LSTM。

## PyTorch 导出产物校验

`test_torch_export_validation.py` 在真实导出流程的末尾注入编码缺失和 ONNX 损坏，
检查导出必须抛出错误；覆盖整层、输入编号、输出、权重通道、非法 scale、未校准，
以及标量输入、Concat 共享量化器和重复算子消除的兼容性。另检查节点名错配、孤立 key、
缺少量化器的计算输入，以及 GELU 展开后删除中间节点或第二路输入编码必须报错。
在安装 rx-met 的环境中可使用 CPU 执行：

```bash
python3 -m unittest discover -s tests -p test_torch_export_validation.py -v
```

`test_native_recurrent_state.py` 另在 CUDA 环境验证 GRU/LSTM 内部编码及权重缺失会使
导出失败，并覆盖关闭量化的循环层无需编码的情况。

`test_quantizable_batchnorm.py` 对照标准 PyTorch BN 检查训练输出、梯度、运行统计量、
单通道 buffer 形状、累计平均和 QAT 冻结。KWS 测试检查自定义 BN 的展开节点、两路
输入编码、所有 ONNX/encoding 名称及检查点重载。导出测试还覆盖参数 reshape 和相同
常量去重后的名称恢复，并通过 ONNX Runtime 验证重命名不改计算结果。

Linear 导出回归覆盖 1～4 维非连续输入、有无偏置、单计算节点、编码名称、ONNX Runtime
浮点数值、QuantSim 不变和编码重载；缺失输入编码仍须报错。

`test_quantizable_gelu.py` 覆盖模块/函数（位置及关键字输入）两种 GELU 近似方式的
输出和梯度、共享实例的独立校准、重复 prepare 的名称稳定性、QAT、ONNX 数值和
编码重载。关闭量化的浮点计算输入仍受严格校验约束。
