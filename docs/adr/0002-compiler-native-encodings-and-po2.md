# 输出使用真实 Power-of-2 scale 的编译器 encodings

- 状态：已接受
- 日期：2026-09-01

## 背景

AIMET 原始 encodings 用于量化模拟器调试和重载。目标编译器使用按节点输入、节点
输出和参数组织的格式。目标部署要求实际量化 scale 和 JSON 中的指数共同对应
`1/2^n`。

Conv、Gemm 和 MatMul 的 bias 需要满足 `Sb = Sx * Sw`，才能与输入和权重的定点
计算保持一致。

## 决策

最终 `.encodings` 使用 `src/aimet_onnx/rx_ptq/compiler_encodings.py` 实现的编译器
格式。AIMET 原始 encodings 作为独立文件保留，用于调试、重载和结果追溯。

Power-of-2 对齐修改实际量化器中的 scale，`n` 从修改后的 scale 计算。默认 ONNX
流程使用项目 PyTorch 导出流程相同的 `cover_range` 规则。Conv、Gemm 和 MatMul
的 bias 在其他量化器校准后通过项目对齐流程设置为 `Sb = Sx * Sw`。

## 理由

- 编译器直接消费节点级格式，AIMET 原始格式用于调试和重载。
- 模拟、导出和部署使用相同的真实 scale，JSON 与实际推理行为保持一致。
- bias scale 从输入和权重 scale 推导，可以明确验证定点计算关系。

## 考虑过的方案

- 直接输出 AIMET 1.0.0/2.0.0 格式：编译器需要增加 AIMET 格式支持，节点级规则仍需扩展。
- 给 encoding 增加 `n` 并保留原 scale：`n` 与实际 scale 会产生差异。
- 将 Po2 作为可选后处理：发布制品可能偏离部署定点约束。

## 影响与限制

- Clean ONNX 和编译器 encodings 必须作为一组制品使用。
- 编译器格式是项目接口；字段或 schema 版本变化需要兼容性说明和回归测试。
- Power-of-2 和 bias 对齐可能改变量化误差，发布验证必须比较 FP32、PTQ、Po2 和
  Clean ONNX 输出。
- 任一量化器缺少节点输入、节点输出或参数映射时，导出立即失败。

使用方法和字段说明见 [`docs/ONNX_PTQ_COMPILER.md`](../ONNX_PTQ_COMPILER.md)。
