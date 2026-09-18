# rx-met

rx-met 是面向 Linux x86_64 NVIDIA GPU 的小模型量化工具包，包含定制 AIMET
（`aimet_torch`、`aimet_onnx`、`aimet_common`）和 QuantGRU。本仓库的功能范围是
小模型量化。

项目自行产出三个独立运行镜像：

| 镜像变体 | CUDA | Python | PyTorch | ONNX Runtime GPU |
| --- | --- | --- | --- | --- |
| `cu118` | 11.8 | 3.10 | 2.7.1 | 1.20.1（CUDA 11 / cuDNN 8） |
| `cu126` | 12.6 | 3.10 | 2.8.0 | 1.23.2（CUDA 12 / cuDNN 9） |
| `cu130` | 13.0 | 3.12 | 2.10.0 | 1.27.0（CUDA 13 / cuDNN 9） |

每个 CUDA 变体维护一对版本化环境镜像：`build-env` 固化编译工具链和第三方
依赖，`runtime-env` 固化运行依赖。产品发布复用环境镜像，编译当前 AIMET/QuantGRU
源码，并将项目 wheel 安装到对应 `runtime-env`。

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
./scripts/release/build.sh

# 构建 cu126 并跳过归档导出
./scripts/release/build.sh --no-export cu126

# 使用指定目录中已校验的成对环境镜像归档
./scripts/release/build.sh --environment-dir /path/to/environment-images/deps-v1

# 在 GPU 机器上执行发布验收
./scripts/release/verify_bundle.sh --gpu --gpu-device 0 cu126

# 使用真实数据完整运行两个 example（路径按实际环境填写）
./scripts/release/verify_kws_example.sh --dataset-dir /path/to/speech_commands_v0.02 cu126
./scripts/release/verify_onnx_ptq_example.sh \
  --model /path/to/model.onnx --dataset-dir /path/to/calib cu126
```

默认制品目录为 `.release/export/`，其中同时包含产品镜像归档和可直接浏览的
`examples/` 示例源码。完整构建、验收流程和可配置环境变量见
[docs/Release_packaging.md](docs/Release_packaging.md)，各脚本的职责和调用关系见
[scripts/README.md](scripts/README.md)。

## 参与贡献

开发环境、测试范围和提交要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 主要目录

- `src/aimet_common/`：Torch 和 ONNX 共用的 AIMET Python 包
- `src/aimet_onnx/`：ONNX 量化前端和 RX PTQ 流程
- `src/aimet_torch/`：PyTorch 量化前端和 RX 导出流程
- `native/`：AIMET C++/CUDA 运行库及 Python bindings
- `operators/quant-gru/`：量化流程替换 GRU 时使用的 QuantGRU 算子
- `packaging/release-bundle/`：产品镜像和导出目录使用的文档模板
- `docker/Dockerfile.environment`：稳定 build-env/runtime-env 构建
- `docker/Dockerfile`：基于稳定环境编译源码并组装产品镜像
- `docker/variants.json`：三个 CUDA 变体和精确依赖的权威配置
- `docker/docker-bake.hcl`：由依赖工具生成的 Buildx Bake 配置
- `docker/requirements/`：构建工具锁和三份自包含运行环境锁
- `scripts/dependencies.py`：生成、检查并下载锁定依赖
- `scripts/environment/build.sh`：按需构建稳定环境镜像，默认产出三个 CUDA 变体的本地归档
- `scripts/environment/export.sh`：在本地生成可复制的环境文件
- `scripts/environment/load.sh`：校验并加载环境归档
- `scripts/environment/verify.sh`：验证环境镜像身份、工具链和运行依赖
- `scripts/release/build.sh`：环境解析、产品构建、验收和导出
- `scripts/release/verify_bundle.sh`：最终镜像静态或 GPU 验收
- `scripts/release/verify_kws_example.sh`：使用真实 Speech Commands 数据运行完整 KWS 用例
- `scripts/release/verify_onnx_ptq_example.sh`：使用指定模型和校准数据运行完整 ONNX PTQ 用例
- `examples/`：KWS 和 ONNX PTQ 快速示例

源码目录采用 Python `src` layout。安装后的公开 import 名保持为
`aimet_common`、`aimet_onnx` 和 `aimet_torch`；`native/` 与
`operators/` 作为 Python namespace 之外的实现目录。

## 许可证

BSD-3-Clause

Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
