# 使用示例

本目录随 rx-met 应用包交付。按发布包顶层 README 在宿主机将 `examples/`
复制到 `~/rx-met-user/`，再启动匹配的 runtime 并挂载应用、工作目录和数据目录。
以下运行命令均在容器内的 `/workspace` 中执行；数据和输出通过挂载目录保存。

| 入口 | 用途 | 需要准备 |
| --- | --- | --- |
| `quick_start_kws.py` | GRU / LSTM 关键词识别的浮点训练、PTQ、Po2、QAT、导出和恢复 | Speech Commands v0.02 |
| `prepare_onnx_ptq_data.py` | 准备 ONNX PTQ 的模型和校准数据 | ONNX 模型、ImageNette 数据 |
| `onnx_ptq_quick_start.py` | ONNX PTQ、Po2 和编译器制品导出 | 上一步生成的模型和校准数据 |
| `config/` | 基础量化配置、普通层位宽及 GRU / LSTM 内部量化配置 | 按模型需要修改 JSON |

## KWS：选择 GRU 或 LSTM

两种模式采用 Google Research
[`kws_streaming/models/att_mh_rnn.py`](https://github.com/google-research/google-research/blob/72ffe35f5a24a312980e6bb811713b24f31e11f9/kws_streaming/models/att_mh_rnn.py#L68-L73)
支持的循环层选项。本示例是对应拓扑的 PyTorch 实现：

`1 秒音频 → MFCC 特征 → 卷积 → 两层双向 GRU 或 LSTM → 多头注意力 → 12 类分类`

准备并解压 [Speech Commands v0.02](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)，
容器内的数据目录应包含 `yes/`、`no/` 等词目录和 `_background_noise_/`。
两种模式使用同一份数据；只需要切换一个参数：

```bash
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02

# 默认使用 GRU；不传 --rnn_type 时等价于此命令
python3 /workspace/examples/quick_start_kws.py --rnn_type gru

# 使用 LSTM，其余量化步骤相同
python3 /workspace/examples/quick_start_kws.py --rnn_type lstm
```

依次执行浮点训练、PTQ 校准与评估、Power-of-2（Po2）转换、QAT 微调、导出和恢复。
校准使用关闭随机增强的训练集样本，测试集用于比较各阶段准确率。
默认浮点训练和 QAT 各运行 1 epoch，便于学习流程；充分训练后的精度需另外评估，
不能将默认运行结果视为上游论文成绩。训练轮数等参数在脚本开头的“全局配置”中。

卷积后的 BN 使用 `QuantizableBatchNorm2d`。浮点训练维护运行均值和方差；
`prepare_model` 在 eval 模式将其展开为 Sub、Div、Mul、Add，每个算子的两路张量输入
参与校准，QAT 冻结统计量和 affine 参数。不要将该类加入 `module_classes_to_exclude`。
导出的激活编码 key 对应 ONNX 节点名，参数编码 key 对应 initializer 名；参数经过
reshape/常量折叠时，导出器按 FX 参数来源恢复名称，无法对应的编码会报错。

网络中的每个 QuantLSTM 实例都是单层、双向、CUDA float32，两个实例串联组成两层网络。
本示例使用序列输出做整段音频分类；隐藏状态由算子初始化为零，不演示跨音频状态传递。

### 修改量化配置

两种模式共用以下文件，**切换算子不需要手动换 JSON**：

| 文件 / 配置节 | 修改什么 |
| --- | --- |
| `config/mrnn_quantsim_config_custom_mixed_precision_v2.json` | QuantSim 基础量化规则 |
| `config/quick_start_full_quant.json` 的 `layer_type_config` | 卷积、全连接等普通层的位宽 |
| 同一文件的 `GRU_config` | GRU 内部量化点 |
| 同一文件的 `LSTM_config` | LSTM 内部量化点，包括隐藏状态 `output` 和细胞状态 `cell_state` |

启动时会打印配置路径和当前循环层对应的配置节。GRU 和 LSTM 的字段结构不同，
具体字段、默认值和最小 LSTM 配置见 [config/README.md](config/README.md)。
配置在 PTQ 前应用；第 5 步将量化 scale 转成 Po2，QAT 使用固定的量化参数微调权重。

也可以传入修改后的完整配置：

```bash
python3 /workspace/examples/quick_start_kws.py --rnn_type lstm \
  --data-dir /datasets/speech_commands_v0.02 \
  --quant-config /workspace/examples/config/quick_start_full_quant.json \
  --output-dir /workspace/output/kws
```

### 输出和恢复

默认输出根目录是 `/workspace/examples/output/quick_start_kws/`。两种模式分别写入 `gru/` 和 `lstm/`，
避免混用权重和量化参数。上面的命令写入 `/workspace/output/kws/lstm/`。
`--output-dir` 优先于环境变量 `RX_MET_KWS_OUTPUT_DIR`；两者均表示**输出根目录**。

| 主要文件 | 用途 |
| --- | --- |
| `model_fp_kws.pth` | 浮点模型检查点 |
| `att_mh_rnn_kws_qat.pth` | QAT 后的模型检查点，用于重建相同结构的 QuantSim |
| `att_mh_rnn_kws.onnx` | 标准 ONNX 计算图，循环层保留为 GRU 或 LSTM 节点 |
| `att_mh_rnn_kws.encodings` | 配套量化参数，部署及量化模型恢复时使用 |

检查点包含 `rnn_type` 和 `model` 权重字典。恢复时必须使用相同的网络类型、结构和量化配置；
GRU 权重不能加载到 LSTM 网络。脚本会自动演示恢复，并同时检查同一批样本的输出和测试集准确率。
输出“量化流程完成（已通过加载验证）”且退出码为 0，表示该流程通过。
ONNX 表达浮点计算语义，量化参数位于配套 encodings；仅运行 ONNX 不等于运行量化模型。
导出器还可能生成中间文件，客户使用上述配套主文件即可。

如使用 `RX_MET_KWS_FP_MODEL` 指定浮点检查点保存位置，请为两种模式指定不同文件；
已有检查点类型不符或缺少类型信息时会报错，选择新路径后再运行。
查看所有入口参数：`python3 /workspace/examples/quick_start_kws.py --help`。

## ONNX PTQ

### 使用已有模型和校准数据

默认读取 `/datasets/mobilenetv2/mobilenetv2-12.onnx` 和
`/datasets/mobilenetv2/calib/` 中的 NPY/NPZ 校准数据。模型输入应为静态 shape，
校准数据的输入名和 shape 应与模型一致。在容器内执行：

```bash
python3 /workspace/examples/onnx_ptq_quick_start.py
```

使用其他输入路径时，可在命令中指定：

```bash
RX_MET_ONNX_PTQ_MODEL=/datasets/model.onnx \
RX_MET_ONNX_PTQ_CALIB=/datasets/calib \
python3 /workspace/examples/onnx_ptq_quick_start.py
```

输出文件名前缀、量化方式等参数可在复制出的 `onnx_ptq_quick_start.py` 开头修改。

### 准备公开的 MobileNetV2 示例输入（可选）

没有现成输入时，可以使用 ONNX Model Zoo 的 MobileNetV2 和 ImageNette 数据。
先在**宿主机**下载到个人工作目录：

```bash
mkdir -p ~/rx-met-user/inputs
curl -fL --retry 3 \
  -o ~/rx-met-user/inputs/mobilenetv2-12.onnx \
  https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx
curl -fL --retry 3 \
  -o ~/rx-met-user/inputs/imagenette2-320.parquet \
  https://huggingface.co/datasets/johnowhitaker/imagenette2-320/resolve/main/data/train-00000-of-00001.parquet
```

也可在联网机器下载后复制到该目录。如果直接访问 Hugging Face 受限，可使用已获准的镜像源。

按顶层 README 启动容器后，在**容器内**执行：

```bash
python3 /workspace/examples/prepare_onnx_ptq_data.py \
  --onnx /workspace/inputs/mobilenetv2-12.onnx \
  --parquet /workspace/inputs/imagenette2-320.parquet \
  --out-root /workspace/datasets/mobilenetv2
export RX_MET_ONNX_PTQ_ROOT=/workspace/datasets
python3 /workspace/examples/onnx_ptq_quick_start.py
```

准备脚本将模型 batch 固定为 1，并生成各 200 个互不重叠的校准和验证样本。
这里把生成的数据写入可写的工作目录，因此需要设置 `RX_MET_ONNX_PTQ_ROOT`；
该变量在当前容器 shell 中有效，重新进入容器后需再次设置。

### 输出

输出“量化流程完成”及 metadata 路径并返回退出码 0，表示 ONNX PTQ 示例完成。
默认输出到 `/workspace/examples/output/onnx_ptq_quick_start/`，对应宿主机
`~/rx-met-user/examples/output/onnx_ptq_quick_start/`。
目录包含 clean ONNX、配套编译器 encodings、AIMET 原始 encodings 和运行 metadata。
