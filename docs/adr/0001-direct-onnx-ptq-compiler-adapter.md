# 使用独立的 ONNX PTQ 编译器适配器

- 状态：已接受
- 日期：2026-09-01

## 背景

`src/aimet_torch/rx_export` 面向 `aimet_torch` QuantSim、单输入模型和特定 GRU
导出规则。已有 ONNX 模型直接以 ONNX 图作为输入。复用 PyTorch 导出链会把 PyTorch
节点命名、GRU 特化规则和 ONNX PTQ 耦合在一起。

ONNX PTQ 还需要保留原模型的输入输出契约，并同时生成 Clean ONNX、AIMET 原始
encodings 和编译器 encodings。

## 决策

已有 ONNX 模型使用独立的 `src/aimet_onnx/rx_ptq` 适配器。该适配器直接处理 ONNX
图，并负责校准、Power-of-2 对齐、编译器格式转换和制品导出。

该功能保持当前 AIMET 2.17 原生运行库基线。升级 AIMET 或调整原生 ABI 应作为独立
决策处理。

## 理由

- ONNX PTQ 可以按 ONNX 图直接建立张量、节点输入输出和参数之间的映射。
- PyTorch 和 ONNX 流程分别演进，各自维护对应的图语义。
- 两条流程仍可复用 `aimet_common` 中的 Power-of-2 算法和相同的量化配置。

## 考虑过的方案

- 扩展 PyTorch 导出器：会把 PyTorch 节点命名、GRU 特化和 ONNX 直量化混在一起。
- 同时升级 AIMET：会引入 Python/原生 ABI、构建和 encoding 行为变化，扩大变更范围。
- 使用独立 ONNX 适配器：边界清晰，可以保留原始 ONNX 图契约。

## 影响与限制

- PyTorch 导出和 ONNX PTQ 是两个公开入口，需要分别测试。
- 共享量化算法放在 `aimet_common`，确保两条流程采用相同实现。
- ONNX 流程适用于 PTQ；QAT 使用 PyTorch 流程。
- 适配器输出必须通过 Clean ONNX 重载、编译器覆盖率和孤立键检查。

使用方法和制品定义见 [`docs/ONNX_PTQ_COMPILER.md`](../ONNX_PTQ_COMPILER.md)。
