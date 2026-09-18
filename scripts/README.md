# 工具脚本

本目录保存项目构建、环境管理、产品发布和验证命令。公开命令按职责分组，辅助实现
位于 `internal/`。

## 产品发布

- `release/build.sh` 解析环境、构建产品镜像、执行验收并导出发布制品。
- `release/verify_bundle.sh` 验证最终镜像和可选 GPU 流程。
- `release/verify_kws_example.sh` 使用真实 Speech Commands 数据验证 KWS 流程。
- `release/verify_onnx_ptq_example.sh` 使用指定模型和校准数据验证 ONNX PTQ 流程。

常用入口为：

```bash
./scripts/release/build.sh
./scripts/release/verify_bundle.sh --gpu --gpu-device 0 cu126
```

## 环境镜像

- `environment/build.sh` 构建并验证稳定 build-env 和 runtime-env。
- `environment/load.sh` 校验并加载已导出的环境归档。
- `environment/export.sh` 导出可搬运的环境归档和校验文件。
- `environment/verify.sh` 验证环境身份、工具链和运行依赖。

环境镜像使用独立版本，可以被多个产品版本复用。

## 跨流程工具

- `build_aimet_native.sh` 独立构建 AIMET 原生运行库。
- `dependencies.py` 生成、检查并下载变体依赖和 lock 文件。
- `lib/environment_images.sh` 提供环境镜像命名和 Buildx 检查函数。

## 辅助实现

- `internal/build_wheels_in_container.sh` 在 builder 镜像中调度项目 wheel 构建。
- `internal/build_quant_gru_wheel.sh` 构建 QuantGRU 原生库和 wheel。
- `internal/prepare_onnxruntime_headers.sh` 准备并校验 ONNX Runtime 头文件。
- `internal/prepare_packaging.py` 在临时源码副本中设置发布版本。
- `internal/verify_dependency_wheelhouse.py` 校验离线 wheelhouse 的包名、版本和 ABI。

完整发布参数和操作流程见
[`docs/Release_packaging.md`](../docs/Release_packaging.md)。
