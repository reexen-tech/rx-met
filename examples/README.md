# 运行示例

本文可随应用包独立阅读。先安装应用并按应用包 README 启动匹配的容器，再准备对应数据：

| 要运行的流程 | 设备 | 必需输入 |
| --- | --- | --- |
| KWS 浮点训练、PTQ、Po2、QAT、恢复 | CUDA | Speech Commands v0.02 |
| KWS 只校准与 PTQ | CPU / CUDA | 同类型浮点检查点，以及 Speech Commands 或已保存的校准和验证波形 |
| 已有 ONNX 的 PTQ | CPU / CUDA | 静态 ONNX 和预处理后的校准 NPY/NPZ；下面以 MobileNetV2 演示 |

在宿主机的应用包目录执行：

```bash
mkdir -p "$HOME/rx-met-user"
cp -a examples "$HOME/rx-met-user/"
```

按应用包 README 启动匹配的 CPU/CUDA 容器，CPU 应用包省略 GPU 参数。
工作目录使用以下挂载：

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

使用 CUDA 容器，选择 GRU 或 LSTM 运行：

```bash
python3 examples/quick_start_kws.py --rnn_type gru
python3 examples/quick_start_kws.py --rnn_type lstm
```

默认浮点训练和 QAT 各 1 epoch。每次任务保存在
`./output/quick_start_kws/<gru|lstm>/<execution-id>/`，最终模型位于 `exports/`，
供下文复用的浮点检查点为 `checkpoints/model_fp_kws.pth`。

完整浮点 MFCC 在分类网络外执行，导出 ONNX 输入为 `[1,49,20]` 特征，输出为 `[1,12]` logits。
Attention 使用广播乘法、归约、指数和除法，Linear 使用等价的 1×1 Conv2d；
仍使用 `config/quick_start_full_quant.json`，修改配置后需要重新校准。

### 只校准和 PTQ（CPU / GPU）

已有同类型的浮点检查点时，可以跳过训练。下例的 `/models/model_fp_kws.pth` 是容器内
的检查点路径；若使用上一步生成的文件，改为其 `/workspace/output/...` 路径。若检查点
存放在宿主机的其他目录，在启动容器时将该目录挂载到 `/models:ro`。

若还没有固定样本，下面的命令从挂载的 Speech Commands 中分别为校准和验证抽取每类
20 个未增强样本，写入 `shared_kws/calib/`、`shared_kws/val/` 和 `shared_kws/manifest.json`。
请使用尚无 `calib/` 和 `val/` 子目录的输出路径。CPU 和 CUDA 均可准备样本和运行 PTQ：

```bash
python3 examples/quick_start_kws.py --rnn-type gru --device cpu --ptq-only \
  --fp-checkpoint /models/model_fp_kws.pth \
  --data-dir /datasets/speech_commands_v0.02 \
  --export-feeds-dir /workspace/shared_kws
```

已保存固定波形时，可直接指定其目录，跳过对 Speech Commands 的挂载和抽样。
每个 NPY 文件保存一条 `[1,16000]` float32 波形；NPZ 文件使用 `input` 键保存同样的数组。
数据须为 16 kHz，按训练时的规则完成长度处理；示例按文件名排序读取，目录中有 NPZ
时只读取 NPZ。验证目录默认是校准目录旁的 `val/`，也可用 `--validation-dir` 指定；
它必须存在且非空，不会自动复用校准集。Top-1 指标还需在样本目录的上一级提供
`manifest.json`，用 `calib_files` / `calib_labels` 和 `val_files` / `val_labels` 配对文件名与
整数类别编号；使用自定义验证目录名时，加载器选择 `calib_*` 字段。上面的样本导出已生成清单。
以下以 GRU 为例：

```bash
python3 examples/quick_start_kws.py --rnn-type gru --device cpu --ptq-only \
  --fp-checkpoint /models/model_fp_kws.pth \
  --calib-dir /workspace/shared_kws/calib --validation-dir /workspace/shared_kws/val \
  --output-dir output/kws_cpu
```

LSTM 将 `--rnn-type gru` 改为 `--rnn-type lstm`，并使用对应的浮点检查点。

此模式执行 FP 评估、校准、Po2、导出和恢复，不执行浮点训练或 QAT。产物为
`att_mh_rnn_kws_ptq.onnx`（不含 Q/DQ）、`att_mh_rnn_kws_ptq.json`、
`outputs.npz` 和 metadata，PTQ 权重保存在 `checkpoints/att_mh_rnn_kws_ptq.pth`。
metadata 记录检查点和实际波形数组的 SHA-256、设备和恢复误差。验证样本带标签时也记录 Top-1。
`--calib-limit N` 和 `--validation-limit N` 可限制共享目录中按文件名排序的前 N 个样本。

### 使用 KWS 的 ONNX 做直量化

已有 clean ONNX 和对应的 MFCC 数据时，可在 CPU/CUDA 上复用同一份图和样本。
这里的 `calib/` 和 `val/` 应包含 `[1,49,20]` 的 MFCC 特征，不能直接使用
KWS PTQ 读入的 `[1,16000]` 波形。前处理使用浮点检查点中的 MFCC 参数；在源码仓库中，
[CPU/GPU 对照脚本](#源码仓库中的-cpugpu-对照)会自动从固定波形生成这两份特征。

```bash
python3 examples/quick_start_onnx_ptq.py --device cpu \
  --model /path/to/exports/att_mh_rnn_kws_ptq.onnx \
  --calib-dir /workspace/mfcc_kws/calib --validation-dir /workspace/mfcc_kws/val \
  --prefix att_mh_rnn_kws_ptq --output-dir output/onnx_cpu
```

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

上面的量化命令默认使用 CUDA；CPU runtime 改为
`python3 examples/quick_start_onnx_ptq.py --device cpu`。

结果保存到 `./output/onnx_ptq_quick_start/<execution-id>/exports/`。部署时配套使用同次导出的
`mobilenetv2_ptq.onnx` 和 `mobilenetv2_ptq.json`；metadata 和 `outputs.npz` 保存各阶段的评估结果。
日志报告 FP32、PTQ、Po2 和恢复后的验证集 Top-1 精度。该指标使用预处理时抽出的独立样本，
不是 ImageNet 官方验证集精度。
日志输出“量化流程完成”表示运行成功。

`execution-id` 自动使用北京时间，格式为 `YYYYMMDD_HHMMSS_ffffff`，例如
`20261027_105308_820401`。每次任务的详细日志和运行记录位于同级的
`run.log`、`run.json`，产物位置见终端输出。

## 源码仓库中的 CPU/GPU 对照

以下命令从源码仓库根目录执行。运行对照工具的 Python 环境需能导入 NumPy、ONNX 和 Protobuf。
应用包用户运行上面的示例即可完成量化。

源码仓库提供逐项精确比较命令；任一数值、结构或字段不同都会写入报告并返回 1，
不自动使用容差或忽略差异：

```bash
python3 scripts/validation/compare_ptq_artifacts.py \
  --left-onnx /cpu/exports/att_mh_rnn_kws_ptq.onnx \
  --right-onnx /cuda/exports/att_mh_rnn_kws_ptq.onnx \
  --left-encodings /cpu/exports/att_mh_rnn_kws_ptq.json \
  --right-encodings /cuda/exports/att_mh_rnn_kws_ptq.json \
  --left-outputs /cpu/exports/outputs.npz --right-outputs /cuda/exports/outputs.npz \
  --report output/comparison.json
```

报告分别记录部署 ONNX、最终 JSON 和 FP32 / affine PTQ / Po2 输出。CPU/CUDA
浮点累加及独立校准网格可能存在差异；最终 Po2 制品相同也不代表前面各阶段逐位相同。
Torch 与 ONNX 的传播节点、常量编码归属也会逐项报告，不能只依据文件数量判定一致。

完整编排命令接受上文保存的固定波形目录和带验证标签的 `manifest.json`。它会在两种设备上分别运行
KWS PTQ，再把同一份 clean ONNX 和从同一组波形生成的 MFCC 特征交给两个设备做 ONNX PTQ。
`--output-dir` 必须是尚未创建的目录，其中将生成日志、四份逐项比较报告和汇总 `comparison.json`：

```bash
python3 scripts/validation/verify_examples_cpu_gpu.py --rnn-type gru \
  --fp-checkpoint /models/model_fp_kws.pth \
  --calib-dir /workspace/shared_kws/calib --validation-dir /workspace/shared_kws/val \
  --cpu-command /cpu-env/bin/python --cuda-command /gpu-env/bin/python \
  --output-dir output/gru_cpu_gpu_comparison
```

两个命令必须使用各自已安装 CPU/GPU wheel 的环境。也可传入以 `python3` 结尾的 Docker 命令；
此时仓库、模型、样本和输出目录须以相同绝对路径挂载到容器。
四份报告均一致时退出码为 0；任一报告有差异时为 1，可按汇总定位报告。
