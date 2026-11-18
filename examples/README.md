# 运行示例

本文可随应用包独立阅读。先安装应用并按应用包 README 启动匹配的容器，再准备对应数据：

| 要运行的流程 | 设备 | 必需输入 |
| --- | --- | --- |
| KWS 浮点训练、PTQ、可选 Po2 演示、QAT、恢复 | CUDA | Speech Commands v0.02 |
| KWS 只校准与 PTQ | CPU / CUDA | 同类型浮点检查点，以及 Speech Commands 或已保存的校准和验证波形 |
| 已有 ONNX 的 PTQ | CPU / CUDA | 默认使用 KWS 导出的浮点 ONNX 和同一份 Speech Commands；也可指定自己的静态 ONNX 和预处理样本 |

Power-of-2（Po2）是按目标后端要求选择的可选转换，不是 PTQ 的必要步骤。KWS 和 ONNX PTQ
示例均执行该步骤，以演示转换前后的精度比较；日志将其标为“可选”阶段。

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

此模式执行 FP 评估、校准、可选步骤 Po2 的演示、导出和恢复，不执行浮点训练或 QAT。产物为
`att_mh_rnn_kws_ptq.onnx`（不含 Q/DQ）、`att_mh_rnn_kws_ptq.json`、
`outputs.npz` 和 metadata，PTQ 权重保存在 `checkpoints/att_mh_rnn_kws_ptq.pth`。
metadata 记录检查点和实际波形数组的 SHA-256、设备和恢复误差。验证样本带标签时也记录 Top-1。
`--calib-limit N` 和 `--validation-limit N` 可限制共享目录中按文件名排序的前 N 个样本。

## ONNX PTQ：复用 KWS 的浮点 ONNX

两个示例独立运行。ONNX PTQ 默认读取 KWS 完整流程已有的最终 ONNX，并自行从同一份
Speech Commands v0.02 抽样和提取 MFCC，无需额外下载模型、数据或执行准备脚本。
KWS 导出的 ONNX 是不含 Q/DQ 的浮点图，量化参数保存在外部 JSON 中；即使经过 QAT，
这份 ONNX 仍可作为新一轮 PTQ 的输入。本流程重新生成量化参数，不加载原来的量化 JSON。

在运行过 KWS 的同一工作目录、使用同一数据挂载执行：

```bash
python3 examples/quick_start_onnx_ptq.py
```

默认使用 GRU，自动选择 `./output/quick_start_kws/gru/` 中最新成功的完整 KWS 运行，读取
`exports/att_mh_rnn_kws.onnx`。若通过 `RX_MET_KWS_OUTPUT_DIR` 修改了 KWS 输出根目录，保持该变量一致。
LSTM 指定 `--rnn-type lstm`。CPU runtime 另加 `--device cpu`；准备 ONNX 的 KWS 完整训练使用 CUDA。
LSTM ONNX PTQ 可直接使用 `cpu`、`cu126` 或 `cu130` 应用；当前 `cu118` 默认构建的
LSTM bridge 与其 ONNX Runtime API 不匹配，需要先用匹配头文件重建算子 wheel。

数据默认读取 `/datasets/speech_commands_v0.02`，可用 `RX_MET_SPEECH_COMMANDS_ROOT` 或
`--data-dir` 覆盖。ONNX 示例分别从训练、验证划分中固定抽取每类 20 个未增强样本，
使用与标准 KWS 示例相同的固定前端提取 `[1,49,20]` MFCC，并保存本次使用的特征和标签，
无需 KWS 检查点。可用 `--samples-per-class` 调整数量。显式选择另一份标准 KWS ONNX 时：

```bash
python3 examples/quick_start_onnx_ptq.py --device cpu \
  --model /path/to/exports/att_mh_rnn_kws.onnx \
  --data-dir /datasets/speech_commands_v0.02
```

Po2 按目标后端要求选用，本示例执行转换演示。导出和恢复接口同时支持 affine PTQ 和 Po2
编码，不要求 scale 必须为 2 的幂。

### 使用自己的模型和预处理样本

通用流程通过 `--model` 和 `--calib-dir` 接收静态 ONNX 及匹配的 NPY/NPZ，
此时不读取 Speech Commands，也不执行 MFCC。输入名称、shape 和 dtype
必须与模型一致；使用 KWS ONNX 时，样本应为 `[1,49,20]` 特征，不能直接使用 `[1,16000]` 波形。

```bash
python3 examples/quick_start_onnx_ptq.py --device cpu \
  --model /models/model.onnx \
  --calib-dir /datasets/my_model/calib --validation-dir /datasets/my_model/val \
  --prefix my_model --output-dir output/my_model
```

模型输入维度须为静态正整数，每份样本包含一次前向所需的全部输入，shape 包含模型声明的
batch 维度，dtype 与模型声明一致，数组不能包含非有限值。单输入可将每份样本保存为一个
NPY；多输入使用 NPZ，每个键与 ONNX 输入名完全一致，不得缺失或增加键。每份 NPZ
保存一次完整前向的输入。文件名可自定，按文件名排序读取；同一目录中有 NPZ 时只读取 NPZ。

验证目录默认使用校准目录旁的 `val/`，也可通过 `--validation-dir` 指定。显式指定的目录
须存在且有效；默认 `val/` 不存在时复用校准样本作数值比较。分类模型使用 batch-1 样本，
并在验证目录上一级的 `manifest.json` 中提供 `val_files` 和 `val_labels` 数组，按文件名
与整数类别编号一一配对；每份前向须保存为单输入 NPY 或完整输入 NPZ。
脚本可计算 `output[0]` 的 Top-1；该输出必须为 `[batch, classes]`，
不会自动查找后续输出。未提供标签时
只比较输出数值；正式精度评估应使用独立于校准数据的验证集。需要调整量化配置时，使用
`--config-file` 和 `--quant-config` 指定 JSON，字段说明见同目录的
[config/README.md](config/README.md)。

默认结果保存到 `./output/onnx_ptq_quick_start/<execution-id>/exports/`。部署时配套使用同次导出的
ONNX 和 JSON；metadata 和 `outputs.npz` 保存各阶段的评估结果。
日志比较 FP32、PTQ、Po2 和恢复后的输出。默认 KWS 示例以及提供
分类验证标签的模型另报告 Top-1；其他模型只报告输出误差。
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
