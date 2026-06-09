# `tools/reex/` — TurboQuant 离线工具

本目录存放 TurboQuant 的离线辅助小程序，**编译期使用，不进运行时**。

## 编译开关

仅在 `cmake -DREEX_TURBOQUANT=ON` 时编译。

## 工具清单

| 工具 | 用途 |
| --- | --- |
| `reex_turboquant_gen_wht_signs.cpp` | 用 `seed=42` PRNG 一次性生成 sign 表（`s1[128] / s2[128] / s1[64] / s2[64]`），输出到 `ggml/src/ggml-cpu/reex/reex_turboquant_wht_signs.inc`，由 CPU/CUDA/Metal/Vulkan/HIP 各 backend `#include` 共享同一份常量。手工运行一次即可，结果稳定可复现。 |

## 命名约定

`reex_turboquant_<功能>.cpp`，与 `src/reex/`、`ggml/src/ggml-*/reex/` 下的命名前缀一致。
