# 工具脚本

本目录保存环境、应用发布和模型制品验证入口。构建与交付命令在仓库根目录执行。

## 环境与应用发布

| 入口 | 用途 |
| --- | --- |
| `environment/build.sh`、`verify.sh` | 准备和验证公共 CPU/CUDA 环境；默认选择三个 CUDA 变体，CPU 显式指定 `cpu` |
| `environment/export_runtime.sh`、`export.sh` | 导出环境归档及各归档的校验、追踪信息；`export.sh` 可选择 build/runtime 角色 |
| `environment/load.sh` | 校验并导入指定环境归档 |
| `release/build.sh` | 构建应用包并离线安装自检；`--with-env` 可附带 runtime 归档 |
| `release/verify_bundle.sh` | 验收安装目录，CPU 检查 PTQ，CUDA 通过 `--gpu` 启用 GPU 验证 |

完整顺序和参数见[应用包发布](../docs/Release_packaging.md)与
[公共环境](../docs/Shared_environment.md)。

## 模型制品验收

`verify_compiler_artifacts.py` 只读检查 ONNX、schema v4 编码、张量维度和图映射，
默认要求 Po2 和完整计算输入覆盖。使用已安装 NumPy 和 ONNX 的 Python 环境，从仓库根目录执行：

```bash
python3 scripts/verify_compiler_artifacts.py \
  --onnx /path/to/model.onnx \
  --encodings /path/to/model.json \
  --report /path/to/acceptance.json
```

工具不修改输入文件，默认把 JSON 报告打印到标准输出。可选的 `--report` 同时写出报告文件，
可以覆盖已有报告。校验通过返回 0，制品校验失败返回 1；命令参数无效时由参数解析器报错。
`--allow-affine` 允许 affine scale，`--allow-partial` 只检查已声明的编码，不要求全部计算输入覆盖。

## CPU 与 GPU 制品对照

在源码仓库根目录执行以下命令。运行对照工具的 Python 需要 NumPy、ONNX 和 Protobuf。

比较已有的两份 PTQ 导出时，将 `cpu_exports` 和 `cuda_exports` 指向各次运行的 `exports/` 目录。
下面的命令逐项比较 ONNX 图结构和参数、JSON 编码，以及 `outputs.npz` 中各阶段的输出。
仅比较图和编码时，可同时省略两个 `--*-outputs` 参数。

```bash
cpu_exports=/path/to/cpu/exports
cuda_exports=/path/to/cuda/exports
python3 scripts/validation/compare_ptq_artifacts.py \
  --left-onnx "$cpu_exports/att_mh_rnn_kws_ptq.onnx" --right-onnx "$cuda_exports/att_mh_rnn_kws_ptq.onnx" \
  --left-encodings "$cpu_exports/att_mh_rnn_kws_ptq.json" --right-encodings "$cuda_exports/att_mh_rnn_kws_ptq.json" \
  --left-outputs "$cpu_exports/outputs.npz" --right-outputs "$cuda_exports/outputs.npz" \
  --report output/comparison.json
```

从同一检查点重新运行两套示例时，先用 KWS 的 `--export-feeds-dir` 准备固定的校准和验证波形，
保留 `calib/`、`val/` 和含验证标签的 `manifest.json`。分别选择已安装对应 wheel 的 CPU 和 CUDA
Python 环境，并指定尚未创建的输出目录。使用 Docker 命令作为 Python 前缀时，仓库、模型、
样本和输出目录在宿主机和容器中的绝对路径须相同。

```bash
python3 scripts/validation/verify_examples_cpu_gpu.py --rnn-type gru \
  --fp-checkpoint /models/model_fp_kws.pth \
  --calib-dir /workspace/shared_kws/calib --validation-dir /workspace/shared_kws/val \
  --cpu-command /cpu-env/bin/python --cuda-command /gpu-env/bin/python \
  --output-dir output/gru_cpu_gpu_comparison
```

脚本在两种设备上运行 KWS PTQ，再用 CUDA 侧导出的同一份 ONNX 和同一组 MFCC 特征运行两次 ONNX PTQ。
输出目录中的 `comparison.json` 指向四份报告，分别比较同一示例在两种设备上的结果，
以及同一设备上 Torch 和 ONNX 的结果。比较采用精确匹配，全部一致时返回 0，发现差异时返回 1。
CPU 与 CUDA 的浮点累加和独立校准可能产生差异，可按报告中的 FP32、affine PTQ 及可选 Po2 输出定位。

## 实现辅助模块

`environment/common.py` 维护环境定义与兼容检查，`runtime_archives.py` 处理归档。
`release/artifacts.py`、`verify_install.py`、`verify_runtime_archive.py` 实现发布与安装验收。
这些维护者实现不作为独立用户工具交付。
