# `src/reex/` — 本仓库 TurboQuant 接入的 llama 层源码

本目录存放 TurboQuant 在 llama 层（`src/llama-*`）的本地适配代码。

## 编译开关

整个目录下的 `.cpp` / `.h` 仅在 `cmake -DREEX_TURBOQUANT=ON` 时被编译进 `libllama` / `libggml`。
`OFF` 是**默认**，OFF 时本目录在产物里不可见，与 upstream `llama.cpp` 二进制等价。

## 命名约定

- 文件名：`reex_turboquant_<功能>.{h,cpp}`，例如 `reex_turboquant_layer_spec.cpp`、`reex_turboquant_innerq.cpp`、`reex_turboquant_trace.h`。
- C++ 命名空间：`reex_turboquant::`。
- 不在本目录使用任何 `turbo_*`、`block_turbo*` 等可能与 TheTom / 上游撞名的符号。

## 模块清单（Phase 1-2 落地后填）

| 文件 | 角色 |
| --- | --- |
| `reex_turboquant_layer_spec.{h,cpp}` | `resolve_layer_spec(hparams, type_k, type_v, il)` 纯函数，依据用户全局 `-ctk` / `-ctv` 选择 + head_dim WHT 整除性，决定哪些层最终用 `TQ_K3 / TQ_V2 / TQ_V4`。 |
| `reex_turboquant_innerq.{h,cpp}` | InnerQ per-channel 方差统计与 scale_inv 张量构造。 |
| `reex_turboquant_trace.h` | `LLAMA_REEX_TRACE` 控制的轻量打点。 |

更多设计细节见 `docs/turboquant/01_设计与实现计划.md`。
