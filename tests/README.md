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
  --fp-checkpoint /workspace/output/quick_start_kws/gru/model_fp_kws.pth \
  --data-dir /datasets/speech_commands_v0.02
```

默认每类取 20 条训练校准及验证样本，输出到运行目录的
`output/kws_ptq_roundtrip/<时间>/`。`torch_ptq/`、`onnx_ptq/` 保存两轮产物，
`comparison.json` 逐项记录 ONNX、编码、GRU 编码和验证结果，`report.md` 提供摘要。
产物不一致时仍保存完整报告并返回 1；`--report-only` 可只生成报告而返回 0。
已有输出目录拒绝覆盖，可用 `--output-dir` 指定新目录。

## 文件说明

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
