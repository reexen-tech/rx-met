# rx-met Docker 镜像使用说明

rx-met 以三个独立的 Linux x86_64 GPU 镜像交付：

| 变体 | CUDA | Python | PyTorch | ONNX Runtime GPU | 建议场景 |
| --- | --- | --- | --- | --- | --- |
| `cu118` | 11.8 | 3.10 | 2.7.1 | 1.20.1 | 需要 CUDA 11.8 用户态运行库 |
| `cu126` | 12.6 | 3.10 | 2.8.0 | 1.23.2 | 当前兼容性基线 |
| `cu130` | 13.0 | 3.12 | 2.10.0 | 1.27.0 | CUDA 13 和 Blackwell `sm_120` |

宿主机必须安装 NVIDIA Driver、Docker 和 NVIDIA Container Toolkit。宿主机不
需要安装与容器一致的 CUDA Toolkit。

## 加载镜像

先在制品目录校验完整性，再加载所需变体：

```bash
sha256sum --check --strict SHA256SUMS
zstd -dc rx-met-v@VERSION@-cu126-linux-amd64.tar.zst | docker load
```

## 启动容器

```bash
docker run --gpus all --rm -it \
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

如果程序需要多进程 DataLoader，可保留 `--ipc=host`，或改用明确的
`--shm-size`。`--group-add` 用于 CIFS/NFS 等仅允许所属组读取的共享数据目录；
`:ro` 仍保证容器不能修改数据。

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

模型和校准数据默认位于 `/datasets/mobilenetv2/`：

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
