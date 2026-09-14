# rx-met

rx-met 是面向 Linux x86_64 NVIDIA GPU 的小模型量化工具包，包含定制 AIMET
（`aimet_torch`、`aimet_onnx`、`aimet_common`）和 QuantGRU。大模型量化不在
本仓库。

项目自行产出三个独立运行镜像：

| 镜像变体 | CUDA | Python | PyTorch | ONNX Runtime GPU |
| --- | --- | --- | --- | --- |
| `cu118` | 11.8 | 3.10 | 2.7.1 | 1.20.1（CUDA 11 / cuDNN 8） |
| `cu126` | 12.6 | 3.10 | 2.8.0 | 1.23.2（CUDA 12 / cuDNN 9） |
| `cu130` | 13.0 | 3.12 | 2.10.0 | 1.27.0（CUDA 13 / cuDNN 9） |

每个 CUDA 变体维护一对版本化环境镜像：`build-env` 固化编译工具链和第三方
依赖，`runtime-env` 固化运行依赖。产品发布只编译当前 AIMET/QuantGRU 源码并
将项目 wheel 安装到对应 `runtime-env`，不重复准备环境。

## 快速运行

以版本 `1.0.0` 的 CUDA 12.6 变体为例：

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  -v /path/to/workspace:/workspace \
  -v /path/to/datasets:/datasets:ro \
  rx-met:1.0.0-cu126
```

容器内示例位于 `/opt/rx-met/examples`：

```bash
cd /opt/rx-met/examples
export RX_MET_SPEECH_COMMANDS_ROOT=/datasets/speech_commands_v0.02
python3 quick_start_kws.py
```

详细使用方法见 [docs/User_guide.md](docs/User_guide.md)。

## 构建与发布

```bash
# 默认构建 cu118、cu126、cu130，并导出三个 tar.zst
./scripts/release_build.sh

# 只构建一个变体，不导出归档
./scripts/release_build.sh --no-export cu126

# 使用指定目录中已校验的成对环境镜像归档
./scripts/release_build.sh --environment-dir /path/to/environment-images/deps-v1

# 在 GPU 机器上执行发布验收
./scripts/verify_bundle.sh --gpu --gpu-device 0 cu126

# 使用真实数据完整运行两个 example（路径按实际环境填写）
./scripts/verify_kws_example.sh --dataset-dir /path/to/speech_commands_v0.02 cu126
./scripts/verify_onnx_ptq_example.sh \
  --model /path/to/model.onnx --dataset-dir /path/to/calib cu126
```

默认制品目录为 `.release/export/`。完整构建、验收流程和可配置环境变量见
[docs/Release_packaging.md](docs/Release_packaging.md)，各脚本的职责和调用关系见
[scripts/README.md](scripts/README.md)。

## 主要目录

- `docker/Dockerfile.environment`：稳定 build-env/runtime-env 构建
- `docker/Dockerfile`：基于稳定环境编译源码并组装产品镜像
- `docker/docker-bake.hcl`：三个 CUDA 变体的唯一构建矩阵
- `docker/requirements/`：公共依赖锁、三份 PyTorch/CUDA 锁和三份 ONNX Runtime GPU 锁
- `scripts/build_environment_images.sh`：按需构建稳定环境镜像，默认产出三个 CUDA 变体的本地归档
- `scripts/export_environment_images.sh`：在本地生成可人工搬运的环境文件
- `scripts/load_environment_images.sh`：校验并加载环境归档
- `scripts/release_build.sh`：环境解析、产品构建、验收和导出
- `scripts/verify_bundle.sh`：最终镜像静态或 GPU 验收
- `scripts/verify_kws_example.sh`：使用真实 Speech Commands 数据运行完整 KWS 用例
- `scripts/verify_onnx_ptq_example.sh`：使用指定模型和校准数据运行完整 ONNX PTQ 用例
- `examples/`：KWS 和 ONNX PTQ 快速示例

## 许可证

BSD-3-Clause

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
