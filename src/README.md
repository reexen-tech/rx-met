# Python 源码

本目录采用 Python `src` layout，保存 rx-met wheel 安装的三个公开 import package。
目录层级不改变安装后的包名。

## 包说明

- `aimet_common/` 保存 Torch 和 ONNX 共用的量化逻辑、配置和原生库加载入口。
- `aimet_onnx/` 保存 ONNX 量化、图处理、PTQ 和编译器制品导出逻辑。
- `aimet_torch/` 保存 PyTorch 模型准备、QuantSim、QAT、导出和 QuantGRU 集成逻辑。

`native/` 构建的动态库会安装到 `aimet_common`。自定义循环算子的实现位于
`operators/`，不属于本目录。
