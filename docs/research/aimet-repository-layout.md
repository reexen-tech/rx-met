# AIMET 官方仓库布局与 rx-met 对照

检索日期：2026-09-17

检索范围：Qualcomm 官方 [AIMET 仓库](https://github.com/qualcomm/aimet)、
当前默认分支 `develop`，以及
[`2.17.0`](https://github.com/qualcomm/aimet/tree/2.17.0) tag（commit
[`0e679b7`](https://github.com/qualcomm/aimet/commit/0e679b705818df7f408971e34693b8098f0971a9)）。

## 结论

rx-met 使用面向发行物的裁剪布局：

```text
src/
├── aimet_common/
├── aimet_onnx/
└── aimet_torch/
native/
├── common/
├── onnx/
├── third_party/
└── tests/
```

`src/` 下三个目录是 Python import package；`native/` 是为这些包生成动态库的
C++/CUDA 实现。源码位置变化不影响安装后的 `import aimet_torch` 等公开接口。

## 官方 AIMET 2.17.0

官方仓库按实现层和框架组织，而不是按最终 import 名平铺：

```text
aimet/
├── ModelOptimizations/
│   ├── DlQuantization/
│   └── PyModelOptimizations/
└── TrainingExtensions/
    ├── common/
    │   └── src/python/aimet_common/
    ├── onnx/
    │   ├── src/*.cpp
    │   └── src/python/aimet_onnx/
    └── torch/
        └── src/python/aimet_torch/
```

一手目录来源：

- [TrainingExtensions](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/TrainingExtensions)
- [aimet_common](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/TrainingExtensions/common/src/python/aimet_common)
- [aimet_onnx](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/TrainingExtensions/onnx/src/python/aimet_onnx)
- [aimet_torch](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/TrainingExtensions/torch/src/python/aimet_torch)
- [ModelOptimizations](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/ModelOptimizations)
- [ONNX native source](https://github.com/qualcomm/aimet/tree/0e679b705818df7f408971e34693b8098f0971a9/TrainingExtensions/onnx/src)

源码路径与安装路径不是同一概念。官方 CMake 会把
`PyModelOptimizations` 生成的 `_libpymo` 和 ONNX 原生产物安装到 Python
package 中。

## rx-met 来源映射

| rx-met 路径 | AIMET 2.17.0 主要来源 | 当前角色 |
| --- | --- | --- |
| `src/aimet_common/` | `TrainingExtensions/common/src/python/aimet_common/` | 共享 Python 代码及原生库承载位置 |
| `src/aimet_onnx/` | `TrainingExtensions/onnx/src/python/aimet_onnx/` | ONNX 前端和 RX PTQ |
| `src/aimet_torch/` | `TrainingExtensions/torch/src/python/aimet_torch/` | PyTorch 前端和 RX 导出 |
| `native/common/` | `ModelOptimizations/DlQuantization/` | 通用 C++/CUDA 量化核心 |
| `native/common/bindings/python/` | `ModelOptimizations/PyModelOptimizations/` | `_libpymo` binding |
| `native/onnx/src/` | `TrainingExtensions/onnx/src/` | ONNX Runtime custom op |

不复制官方完整的 `TrainingExtensions/` 和 `ModelOptimizations/` 层级，是因为
rx-met 只维护明确裁剪后的 Torch、ONNX 和原生运行范围。来源映射和固定基线比机械
复刻上游目录更重要。
