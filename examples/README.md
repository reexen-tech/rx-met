# 运行示例

在宿主机的应用包目录执行：

```bash
mkdir -p ~/rx-met-user
cp -a examples ~/rx-met-user/
```

按应用包 README 启动 GPU 容器，工作目录使用以下挂载：

```bash
  -v "~/rx-met-user:/workspace" \
  -w /workspace \
```

下面的模型、数据挂载也加入同一条 Docker 命令，将 `/path/to/...` 替换为实际路径。
运行命令在容器内执行。

## KWS：GRU / LSTM

下载并解压 [Speech Commands v0.02](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
到宿主机 `/path/to/datasets/speech_commands_v0.02/`，增加挂载：

```bash
  -v "/path/to/datasets/speech_commands_v0.02:/datasets/speech_commands_v0.02:ro" \
```

进入容器，选择 GRU 或 LSTM 运行：

```bash
python3 examples/quick_start_kws.py --rnn_type gru
python3 examples/quick_start_kws.py --rnn_type lstm
```

默认浮点训练和 QAT 各 1 epoch。结果保存到 `./output/quick_start_kws/gru/` 或 `lstm/`。

## ONNX PTQ：MobileNetV2

### 1. 下载模型和数据（宿主机）

```bash
mkdir -p /path/to/models /path/to/datasets
curl -fL --retry 3 \
  -o /path/to/models/mobilenetv2-12.onnx \
  https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx
curl -fL --retry 3 \
  -o /path/to/datasets/imagenette2-320.parquet \
  https://huggingface.co/datasets/johnowhitaker/imagenette2-320/resolve/main/data/train-00000-of-00001.parquet
```

### 2. 增加模型和数据挂载

```bash
  -v "/path/to/models:/models:ro" \
  -v "/path/to/datasets:/datasets:ro" \
```

### 3. 预处理并量化（容器内，同一目录）

先执行此演示专用的预处理：

```bash
python3 examples/prepare_onnx_ptq_data.py \
  --onnx /models/mobilenetv2-12.onnx \
  --parquet /datasets/imagenette2-320.parquet
```

再运行量化，准备脚本也会打印这一步的命令：

```bash
python3 examples/quick_start_onnx_ptq.py
```

结果保存到 `./output/onnx_ptq_quick_start/`，包含 ONNX、encodings 和 metadata。
日志输出“量化流程完成”表示运行成功。
