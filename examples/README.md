# 运行示例

两份代码示例都演示了导出和重新加载，便于在自己的项目中检查量化结果。

- 为 PyTorch 模型编写量化代码时，可从 [quick_start_kws.py](quick_start_kws.py) 看如何把量化接入训练循环。
- 已有静态 ONNX 时，可从 [quick_start_onnx_ptq.py](quick_start_onnx_ptq.py) 看如何用校准数据做 PTQ。

这两份示例在模型外完成数据前处理，导出的 ONNX 接收前处理后的特征。接入自己的模型和数据时，由开发者在代码中实现相应的前处理，将与模型输入形状和 dtype 匹配的数据交给 rx-met 校准。校准与实际推理应使用相同的前处理。

## KWS：GRU / LSTM

从 [quick_start_kws.py](quick_start_kws.py) 的 `_run_kws()` 开始读。示例量化的是 `AttMHRNN` 分类网络。`SpeechFeatures` 和 `FeatureLoader` 将波形转换为 MFCC，供 QuantSim 的 `dummy_input`、校准前向和 ONNX 导出使用。接入自己的项目时，替换模型、浮点权重、数据加载和验证指标。

数据集、MFCC 和输入文件加载位于 [common/kws_data.py](common/kws_data.py)。模型、图准备和检查点读写位于 [common/kws_model.py](common/kws_model.py)。量化和训练步骤位于示例入口。

| 代码位置 | 在自己的量化脚本中如何使用 |
| --- | --- |
| `prepare_model()` 和 `quantsim.QuantizationSimModel` | 参考示例用 `aimet_torch.model_preparer.prepare_model` 准备模型，再用符合模型输入格式的 `dummy_input` 创建 QuantSim。`prepare_model()` 里的 FX 排除项和 `configure_attention_quantizers()` 中按层名共享编码的处理针对 KWS 网络，换模型时按实际结构决定是否需要。 |
| 校准和 PTQ 评估 | 用 `apply_mixed_precision_bitwidth` 为模型应用位宽配置。在 `aimet.nn.compute_encodings(sim.model)` 上下文中，让自己的校准数据经过 `sim.model`，再调用 `align_bias_scale` 并用自己的验证集评估。脚本中的 `aimet` 是 `aimet_torch.v2` 的导入别名。 |
| QAT 微调 | 需要 QAT 时，参考 `freeze_quantizer_parameters` 和 `qat_finetune()`，使用项目自己的训练数据、损失函数和优化器微调 `sim.model`。 |
| 导出和恢复 | 参考 `aimet_torch.rx_export.export_onnx_json` 中的 `export_onnx_json` 导出浮点 ONNX 与配套的量化 JSON。另存模型权重，按相同配置重建 QuantSim，加载权重和编码，用同一输入比较导出前和恢复后的输出。 |

**试跑 GRU 示例。** 按应用包 README 复制示例并启动 CUDA 容器。将 [Speech Commands v0.02](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) 解压到宿主机挂载的数据目录，然后在容器的 `/workspace` 执行：

```bash
python3 examples/quick_start_kws.py \
  --rnn-type gru \
  --data-dir /datasets/speech_commands_v0.02
```

本例默认浮点训练和 QAT 各 1 个 epoch，结果在 `output/quick_start_kws/gru/<execution-id>/`。其中 `exports/att_mh_rnn_kws.onnx` 与 `exports/att_mh_rnn_kws.json` 配套使用。试跑 LSTM 时，将 `gru` 改为 `lstm`。

### 从浮点检查点运行 PTQ

只做校准和 PTQ 时，可阅读同一脚本的 `_run_kws_ptq()`：它先加载浮点权重，再校准、评估、导出并验证恢复结果。用本示例的浮点检查点试跑时，在上面的命令中加上 `--ptq-only --fp-checkpoint /models/model_fp_kws.pth`，模型类型与检查点保持一致。CPU 包支持这一模式，运行时另加 `--device cpu`。

## ONNX PTQ：量化已有模型

从 [quick_start_onnx_ptq.py](quick_start_onnx_ptq.py) 的 `_run_onnx_ptq()` 开始读。它以已有的静态 ONNX 为输入，逐步调用 `aimet_onnx.rx_ptq` 中的接口。示例沿用语音模型只是为了提供具体的输入、输出和精度指标，模型可以来自其他项目。

1. **准备输入。** 示例用 `load_onnx_ptq_inputs` 读取模型和已完成前处理的样本。写自己的数据加载器时，每次前向将处理后的数组组织成 `{输入名: NumPy 数组}` 字典，各输入的形状（含 batch 维）和 dtype 与 ONNX 一致。验证数据和评估指标按自己的任务准备。
2. **创建 QuantSim 并校准。** 参考 `create_onnx_ptq_sim`，用模型和第一份输入创建 QuantSim，再设置位宽和输入输出量化器。脚本用 `sim.compute_encodings(feeds)` 遍历校准输入，随后用 `align_onnx_bias_scale` 对齐 bias。Po2 转换按目标后端需要选用。
3. **导出并验证。** 参考 `export_onnx_json` 导出 ONNX 和编译器量化参数，用 `verify_onnx_ptq_artifacts` 比较恢复前后的输出。示例在有分类验证标签时计算 Top-1。换成其他任务时，可用 `evaluate_onnx_dataset` 返回的输出计算自己的指标。

若项目已有静态 ONNX 和预处理好的校准样本目录，也可从 `aimet_onnx.rx_ptq` 直接导入 `run_onnx_ptq` 调用封装流程。该接口默认使用 CPU provider；示例脚本的 `choose_providers()` 演示了如何选择 CUDA provider。

**试跑 ONNX 示例。** 在包含已复制的 `examples/` 目录和 `output/quick_start_kws/gru/` 成功运行记录的工作目录中，将 Speech Commands v0.02 挂载到 `/datasets/speech_commands_v0.02`，执行：

```bash
python3 examples/quick_start_onnx_ptq.py
```

脚本自动选择 `output/quick_start_kws/gru/` 下最近一次成功运行的 `exports/att_mh_rnn_kws.onnx`，并提取校准和验证用的 MFCC。`--rnn-type lstm` 改为查找 LSTM 运行，`--model /models/model.onnx` 可直接指定已有模型。每次量化运行位于 `output/quick_start_onnx_ptq/<execution-id>/`，模型、编码和评估 metadata 写入 `exports/`。

使用自己的模型和预处理样本时，指定模型及校准目录：

```bash
python3 examples/quick_start_onnx_ptq.py \
  --model /models/model.onnx \
  --calib-dir /datasets/my_model/calib
```

`--calib-dir` 接收已预处理的 NPY/NPZ 样本。脚本默认从校准目录旁的 `val/` 读取验证样本；该目录不存在时，用校准样本比较数值。使用 CPU 包时加上 `--device cpu`。量化配置见[示例配置说明](config/README.md)。

对采用标准 KWS 前处理、输入为 `input: float32 [1,49,20]` 的模型，可省略 `--calib-dir`，用 `--data-dir` 指定 Speech Commands v0.02，默认路径为 `/datasets/speech_commands_v0.02`。脚本从共享的 [common/kws_data.py](common/kws_data.py) 加载数据和 MFCC 前处理，生成的特征分别写入本次运行的 `mfcc_feeds/calib/` 和 `mfcc_feeds/val/`。
