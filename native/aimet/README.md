# AIMET 原生运行库

本目录包含由 rx-met 维护的 AIMET 原生实现。目录按运行职责组织，不继续复制上游
AIMET 仓库布局。

## 源码来源

最初的 C++ 和 CUDA 实现来自 Qualcomm AIMET `2.17.0` tag，导入时未修改内容，
对应 commit：

```text
0e679b705818df7f408971e34693b8098f0971a9
```

导入的上游模块包括：

- `ModelOptimizations/DlQuantization`
- `ModelOptimizations/PyModelOptimizations`
- `TrainingExtensions/onnx/src`

导入后，这些文件作为 rx-met 源码直接维护。rx-met 可以按自身行为要求修改它们；
与原始 AIMET 仓库同步不属于接口或维护契约。

## 目录结构

- `common/`：公共量化实现和 `_libpymo` binding；
- `onnx/`：`libquant_info` 和 ONNX Runtime custom-op 实现；
- `third_party/onnxruntime/`：不可变的 ONNX Runtime 头文件和许可证；
- `tests/`：已安装原生运行接口的冒烟测试。

构建会向 `aimet_common` Python 包安装：

- `_libpymo<EXT_SUFFIX>`
- `libquant_info<EXT_SUFFIX>`
- `libaimet_onnxrt_ops.so`

统一使用 `scripts/build_aimet_native.sh` 作为构建入口。支持 Linux x86_64 上的
CPython 3.10-3.12；通过 `RX_MET_ENABLE_CUDA` 选择 CPU 或 CUDA 实现。
