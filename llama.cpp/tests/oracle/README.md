# `tests/oracle/` — 算法 Reference 实现（仅供测试）

本目录存放 TurboQuant 算法的"参考实现"，用于在测试里给生产路径打 oracle。

## 编译开关

仅在 `cmake -DREEX_TURBOQUANT=ON -DLLAMA_BUILD_TESTS=ON` 时编译。
**不进入** `libllama` / `libggml`，**不进入**任何生产二进制目标。

## 内容来源

`reex_turboquant_oracle_core.{h,cpp}` 来自旧 `turboquant` 分支的 `src/reex/reex_turboquant_core.{h,cpp}`，提取其中算法核心约 600 行：

- `Codebook`（Lloyd-Max 优化得到的 2/3/4-bit centroid 表生成）
- `TurboQuantMSE`（基于显式 `Pi` 矩阵的 Lloyd-Max 标量量化）
- `TurboQuantProd`（基于 PolarQuant 的内积保持量化 + QJL 残差 sign）

这是与 `/home/zcx/CLionProjects/turboquant` Python 参考实现位级对齐的 C++ 移植，是后续 backend kernel 的金标准。

## 用途

1. 单元测试里以 `RotationDense + Pi` 路径生成 golden，对生产路径（`RotationWHT + ggml block`）的 `vec_dot / from_float / to_float` 等做 cos_sim 对拍；
2. 多 backend 实现（CUDA / Metal / Vulkan）落地时，golden 也由这里产生；
3. 在论文角度证明实现正确性时的参照。

## 警告

- 不在本目录引入对 `ggml-cuda` / `ggml-metal` / 其它 backend 的依赖；本目录是**纯 host C++**。
- 不在生产路径调用本目录里的任何函数。

更多设计细节见 `docs/turboquant/01_设计与实现计划.md` §0.4 / §1.P7。
