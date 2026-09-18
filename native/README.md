# AIMET 原生运行库

本目录保存 rx-met 维护的 AIMET C++ 和 CUDA 实现。源码最初来自 Qualcomm AIMET
`2.17.0` tag，对应 commit `0e679b705818df7f408971e34693b8098f0971a9`。

## 目录说明

- `common/` 保存公共量化实现和 `_libpymo` Python binding。
- `onnx/` 保存 `libquant_info` 和 ONNX Runtime custom op。
- `third_party/onnxruntime/` 保存固定版本的 ONNX Runtime 头文件和许可证。
- `tests/` 保存已安装原生运行接口的冒烟测试。

## 文件说明

- `CMakeLists.txt` 统一编排 common 和 ONNX 原生库构建。
- `LICENSE` 保留上游 AIMET 许可证。

构建结果会安装到 `src/aimet_common/`，包括 `_libpymo`、
`libquant_info` 和 `libaimet_onnxrt_ops.so`。统一构建入口为：

```bash
./scripts/build_aimet_native.sh
```

本目录按 rx-met 的运行职责维护，不继续镜像上游 AIMET 的完整目录结构。
