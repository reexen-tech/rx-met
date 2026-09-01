# ADA200 rx-met @DATE_TAG@

面向 ADA200 统一 Docker（`ada200_docker`）的小模型量化软件包。
包内是预编译 wheel 和 examples，**不含 Docker 镜像**。

```text
@BUNDLE_NAME@/
├── README.md
├── ChangeLog.md
├── install.sh
├── wheels/
└── examples/
```

Wheel 版本：`@VERSION@`。

## 环境要求

- 已导入的 `ada200_docker:latest`（Ubuntu 22.04，Python 3.10，`torch==2.8.0+cu128`）
- 宿主机 NVIDIA Driver + NVIDIA Container Toolkit
- 不需要再装 CUDA Toolkit / nvcc

本包会覆盖镜像里的官方 `aimet-torch` / `aimet-onnx`，换成 rx-met 定制 AIMET 和 QuantGRU。
ONNX Runtime 继续用镜像自带的 CPU 版。

## 1. 启动统一 Docker

把发布包和数据集挂进容器：

```bash
docker run --gpus all --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v /path/to/@BUNDLE_NAME@:/opt/rx-met:ro \
  -v /path/to/workspace:/workspace \
  -v /path/to/datasets:/datasets:ro \
  -w /workspace \
  ada200_docker:latest \
  bash
```

`install.sh` 会改系统 Python，若容器用户没有写 `site-packages` 的权限，请去掉 `--user` 用镜像默认用户安装。

## 2. 安装

```bash
cp -a /opt/rx-met /workspace/rx-met
cd /workspace/rx-met
./install.sh
```

脚本会校验 CPython 3.10、Linux x86_64、`torch==2.8.0` 且 `torch.version.cuda == 12.8`，
然后离线安装 `wheels/`。临时跳过检查：`RX_MET_SKIP_ENV_CHECK=1 ./install.sh`。

## 3. 小模型示例（PyTorch / QuantGRU）

将 Speech Commands 放到宿主机数据集目录，容器内路径为
`/datasets/speech_commands_v0.02`。

```bash
cd /workspace/rx-met/examples
# ada200_docker 若未登记 CUDA runtime，先: source /workspace/rx-met/cuda_libs.env
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start.py
```

输出写在 `examples/output/`。量化 JSON 见 `examples/config/README.md`。

## 4. ONNX 直量化示例

默认读 `/datasets/<name>/`，也可用环境变量覆盖：

| 变量 | 默认 |
| --- | --- |
| `RX_MET_ONNX_PTQ_ROOT` | `/datasets` |
| `RX_MET_ONNX_PTQ_EXAMPLE` | `yolo-fastest` |
| `RX_MET_ONNX_PTQ_MODEL` | `$RX_MET_ONNX_PTQ_ROOT/yolo-fastest/yolo-fastest.onnx` |
| `RX_MET_ONNX_PTQ_CALIB` | `$RX_MET_ONNX_PTQ_ROOT/yolo-fastest/input` |

```bash
cd /workspace/rx-met/examples
python3 onnx_ptq_quick_start.py
```

这条路径使用 CPU ONNX Runtime，与 example 默认 `USE_CPU = True` 一致。
