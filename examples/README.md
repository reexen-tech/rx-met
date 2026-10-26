# 运行示例

在宿主机的应用包目录执行：

```bash
mkdir -p "$HOME/rx-met-user"
cp -a examples "$HOME/rx-met-user/"
```

按应用包 README 启动 GPU 容器，工作目录使用以下挂载：

```bash
  -v "$HOME/rx-met-user:/workspace" \
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

默认浮点训练和 QAT 各 1 epoch。每次任务保存在
`./output/quick_start_kws/<gru|lstm>/<execution-id>/`，最终模型位于 `exports/`，检查点位于 `checkpoints/`。

### 只校准和 PTQ（CPU / GPU）

使用同一个浮点检查点和固定的校准、验证样本。CPU 仅支持此模式，默认完整训练流程仍使用 CUDA。
首次从 Speech Commands 选择每类 20 个未增强样本并保存为 batch-1 NPZ：

```bash
python3 examples/quick_start_kws.py --rnn-type gru --device cuda --ptq-only \
  --fp-checkpoint /models/model_fp_kws.pth \
  --data-dir /datasets/speech_commands_v0.02 \
  --export-feeds-dir /workspace/shared_kws
```

随后 CPU/GPU 运行都复用这些文件，不重新采样。LSTM 使用 `--rnn-type lstm` 和对应的浮点检查点：

```bash
python3 examples/quick_start_kws.py --rnn-type gru --device cpu --ptq-only \
  --fp-checkpoint /models/model_fp_kws.pth \
  --calib-dir /workspace/shared_kws/calib --validation-dir /workspace/shared_kws/val \
  --output-dir output/kws_cpu
```

此模式执行 FP 评估、校准、Po2、导出和恢复，不执行浮点训练或 QAT。产物为
`att_mh_rnn_kws_ptq.onnx`（不含 Q/DQ）、`att_mh_rnn_kws_ptq.json`、
`outputs.npz` 和 metadata，PTQ 权重保存在 `checkpoints/att_mh_rnn_kws_ptq.pth`。
metadata 记录检查点和实际输入数组的 SHA256、设备、精度及恢复误差。
`--calib-limit N` 和 `--validation-limit N` 可限制共享目录中按文件名排序的前 N 个样本。

将输出的 clean ONNX 传给第二个示例，在 CPU/GPU 上复用同一份图和样本：

```bash
python3 examples/quick_start_onnx_ptq.py --device cpu \
  --model /path/to/exports/att_mh_rnn_kws_ptq.onnx \
  --calib-dir /workspace/shared_kws/calib --validation-dir /workspace/shared_kws/val \
  --prefix att_mh_rnn_kws_ptq --output-dir output/onnx_cpu
```

源码仓库提供逐项精确比较命令；任一数值、结构或字段不同都会写入报告并返回 1，
不自动使用容差或忽略差异：

报告分别记录部署 ONNX、最终 JSON 和 FP32 / affine PTQ / Po2 输出。CPU/CUDA
浮点累加及独立校准网格可能存在差异；最终 Po2 制品相同也不代表前面各阶段逐位相同。
Torch 与 ONNX 的传播节点、常量编码归属也会逐项报告，不能只依据文件数量判定一致。

```bash
python3 scripts/validation/compare_ptq_artifacts.py \
  --left-onnx /cpu/exports/att_mh_rnn_kws_ptq.onnx \
  --right-onnx /cuda/exports/att_mh_rnn_kws_ptq.onnx \
  --left-encodings /cpu/exports/att_mh_rnn_kws_ptq.json \
  --right-encodings /cuda/exports/att_mh_rnn_kws_ptq.json \
  --left-outputs /cpu/exports/outputs.npz --right-outputs /cuda/exports/outputs.npz \
  --report output/comparison.json
```

完整编排命令会运行两个示例在两种设备上的四次任务，并保存日志、CPU/GPU 对照和
Torch/ONNX 对照的四份逐项比较报告：

```bash
python3 scripts/validation/verify_examples_cpu_gpu.py --rnn-type gru \
  --fp-checkpoint /models/model_fp_kws.pth \
  --calib-dir /workspace/shared_kws/calib --validation-dir /workspace/shared_kws/val \
  --cpu-command /cpu-env/bin/python --cuda-command /gpu-env/bin/python \
  --output-dir output/gru_cpu_gpu_comparison
```

两个命令必须使用各自已安装 CPU/GPU wheel 的环境。也可传入以 `python3` 结尾的 Docker 命令；
此时仓库、模型、样本和输出目录须以相同绝对路径挂载到容器。

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

先执行此演示专用的预处理，生成校准、验证数据及标签：

```bash
python3 examples/prepare_onnx_ptq_data.py \
  --onnx /models/mobilenetv2-12.onnx \
  --parquet /datasets/imagenette2-320.parquet
```

再运行量化，准备脚本也会打印这一步的命令：

```bash
python3 examples/quick_start_onnx_ptq.py
```

结果保存到 `./output/onnx_ptq_quick_start/<execution-id>/exports/`，包含 ONNX、encodings 和 metadata。
日志报告 FP32、PTQ、Po2 和恢复后的验证集 Top-1 精度。
日志输出“量化流程完成”表示运行成功。

`execution-id` 自动使用 UTC 时间戳。每次任务的详细日志和运行记录位于同级的
`run.log`、`run.json`，产物位置见终端输出。
参见[运行目录与日志](../docs/Runtime_outputs.md)。
