# QuantGRU CPU 定点参考实现

本目录提供独立的 C++17 GRU 定点前向实现，使用 int32 保存量化输入、权重和输出，
通过移位参数完成重缩放，激活使用 16 段分段线性 LUT。

该参考实现维护自身的 `GRUQuantParams` 和 `shift_*` 参数接口。主 CUDA 实现使用
`QuantParam` 和按类型派生的重缩放参数，两者通过参数转换和数值对比验证一致性。

## 构建和运行

环境需要 CMake 3.14 及以上版本和 C++17 编译器。以下命令从 QuantGRU 目录执行：

```bash
cmake -S operators/quant-gru/quant-gru-cpu-only \
  -B quant-gru-cpu-only/build -DCMAKE_BUILD_TYPE=Release
cmake --build quant-gru-cpu-only/build --parallel 2
./quant-gru-cpu-only/build/gru_cpu_example
```

构建结果包含：

- `libquant_gru_cpu_static.a`：静态库。
- `libquant_gru_cpu_shared.so`：Linux 共享库。
- `gru_cpu_example`：零输入与零权重的前向冒烟示例。

示例打印 `Output h[640]: 0 0 0 0 0 ...` 和 `=== Done ===`，退出码为 0。
该结果验证构建、链接和基础执行。量化精度和 CPU/CUDA 一致性使用实际权重、校准参数
及代表性数据单独评估。

## 接口与文件

| 文件 | 用途 |
| --- | --- |
| [include/gru_quant_cpu.h](include/gru_quant_cpu.h) | `ForwardPassQuantCPU` 前向接口 |
| [include/quantize_param_types.h](include/quantize_param_types.h) | 移位量、零点和执行参数 |
| [include/quantize_bitwidth_config.h](include/quantize_bitwidth_config.h) | 位宽与量化范围 |
| [include/quantize_lut_types.h](include/quantize_lut_types.h) | 分段线性 LUT 结构 |
| [include/quantize_ops_helper.h](include/quantize_ops_helper.h) | 重缩放和门计算 |
| [src/gru_forward_cpu_quant.cc](src/gru_forward_cpu_quant.cc) | CPU 矩阵乘和递推实现 |
| [src/quantize_lut.cc](src/quantize_lut.cc) | LUT 生成 |
| [example/main.cc](example/main.cc) | 参数初始化和前向调用示例 |

调用方通过 `setRescaleParam()` 设置量化参数，通过 `Run()` 提供量化权重、偏置、
输入和隐藏状态缓冲区。接入真实模型时，调用方负责填充与输入输出网格一致的 scale、
移位量、零点和 LUT。

主模块文档见 [QuantGRU README](../README.md) 和
[量化计算流程](../docs/GRU量化计算流程.md)。
