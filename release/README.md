# rx-met Docker 镜像使用说明

rx-met 以三个独立的 Linux x86_64 GPU 镜像交付：

| 变体 | CUDA | Python | PyTorch | ONNX Runtime GPU | 项目 CUDA 架构 | 最低 NVIDIA Driver | 建议场景 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cu118` | 11.8 | 3.10 | 2.7.1 | 1.20.1 | `sm_80/86/89/90` | 520.61.05 | 驱动或依赖需兼容 CUDA 11.8 |
| `cu126` | 12.6 | 3.10 | 2.8.0 | 1.23.2 | `sm_80/86/89/90` | 560.35.05 | 当前兼容性基线 |
| `cu130` | 13.0 | 3.12 | 2.10.0 | 1.27.0 | `sm_80/86/89/90/120` | 580.126.20 | CUDA 13 和 Blackwell `sm_120` |

宿主机必须安装 NVIDIA Driver、Docker 和 NVIDIA Container Toolkit。宿主机不
需要安装与容器一致的 CUDA Toolkit。选择变体时必须同时满足最低驱动和 GPU
计算能力要求；较新的驱动不能补足镜像中没有编译的 GPU 架构。H200 `sm_90` 可
运行三个变体，RTX 5090 `sm_120` 只能运行 `cu130`。

## 加载镜像

先在制品目录校验完整性，再加载所需变体：

```bash
sha256sum --check --strict SHA256SUMS
zstd -dc rx-met-v@VERSION@-cu126-linux-amd64.tar.zst | docker load
```

## 启动容器

```bash
docker run --gpus device=0 --rm -it \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  --group-add "$(stat -c '%g' /path/to/datasets)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -v /path/to/workspace:/workspace \
  -v /path/to/datasets:/datasets:ro \
  -w /workspace \
  rx-met:@VERSION@-cu126
```

`device=0` 只向容器开放宿主机 GPU 0；多人共用服务器时应改成实际分配的设备号，
不要使用 `--gpus all`。如果程序需要多进程 DataLoader，可保留 `--ipc=host`，或
改用明确的 `--shm-size`。`--group-add` 用于 CIFS/NFS 等仅允许所属组读取的共享
数据目录；`:ro` 仍保证容器不能修改数据。

## 示例源码

交付目录中的 `examples/` 是可直接浏览的示例源码，与产品镜像内
`/opt/rx-met/examples` 的内容一致。源码目录不包含模型、数据集和运行环境；实际
执行时应使用同一批次交付的产品镜像。两个入口分别是：

- `examples/quick_start_kws.py`：KWS 浮点训练、PTQ、Power-of-2、QAT 和导出；
- `examples/onnx_ptq_quick_start.py`：ONNX PTQ、Power-of-2 和 compiler encodings
  导出。

更详细的源码结构和环境变量见 `examples/README.md`。

## KWS 示例

Speech Commands v0.02 在容器内挂载为
`/datasets/speech_commands_v0.02` 后执行：

```bash
cp -a /opt/rx-met/examples /workspace/rx-met-examples
cd /workspace/rx-met-examples
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start_kws.py
```

输出默认写入 `/workspace/rx-met-examples/output/quick_start_kws/`。

## ONNX PTQ 示例

该示例使用公开的 MobileNetV2 ONNX 模型和 ImageNette 数据。若尚未准备输入文件，
先在宿主机下载到启动容器时挂载为 `/workspace` 的目录：

```bash
mkdir -p /path/to/workspace
curl -fL --retry 3 \
  -o /path/to/workspace/mobilenetv2-12.onnx \
  https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx
curl -fL --retry 3 \
  -o /path/to/workspace/imagenette2-320.parquet \
  https://huggingface.co/datasets/johnowhitaker/imagenette2-320/resolve/main/data/train-00000-of-00001.parquet
```

网络受限环境可按部署策略配置代理或替换为可信的公开镜像源，但文件内容应与上述
来源一致。进入容器后生成静态 batch=1 模型和互不重叠的校准/验证样本，再运行
PTQ：

```bash
cp -a /opt/rx-met/examples /workspace/rx-met-examples
cd /workspace/rx-met-examples
python3 prepare_onnx_ptq_data.py \
  --onnx /workspace/mobilenetv2-12.onnx \
  --parquet /workspace/imagenette2-320.parquet \
  --out-root /workspace/datasets/mobilenetv2
export RX_MET_ONNX_PTQ_ROOT=/workspace/datasets
python3 onnx_ptq_quick_start.py
```

该路径默认使用 ONNX Runtime `CUDAExecutionProvider`，provider 缺失时直接报错。
只有 CPU 调试时才显式设置 `RX_MET_ONNX_PTQ_DEVICE=cpu`。数据集、模型和输出不应
打入镜像层，应通过 volume 挂载或写入 `/workspace`。

## 环境检查

```bash
python3 - <<'PY'
import torch
import aimet_torch
import aimet_onnx
import onnxruntime
import quant_gru

print("torch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0))
print("ONNX Runtime", onnxruntime.__version__, onnxruntime.get_available_providers())
assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()
print("rx-met import OK")
PY
```

镜像构建版本和完整 Python 包清单位于 `/opt/rx-met/metadata/`。
