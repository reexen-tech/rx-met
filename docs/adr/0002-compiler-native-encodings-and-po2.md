# Emit compiler-native encodings with real Power-of-2 scales

<!-- Status: accepted -->

AIMET raw encodings 只作为内部重载和追溯产物；最终 `.encodings` 直接生成现有编译器要求的 node-IO/parameter schema。Po2 必须复用 Torch RX 的 `cover_range`：scale 写入 live quantizer，`n` 从真实 scale 计算。Conv/Gemm/MatMul 的 bias 在独立校准后再对齐为 `Sb = Sx * Sw`，不走官方 INT32 concretize。

## Considered Options

- 直接交付 AIMET 1.0.0/2.0.0：编译器需要额外理解 AIMET 原生格式，且无法表达现有 Torch compiler schema 的节点级规则。
- 只给 encoding 增加 `n`：会造成 `n` 与实际 scale 不一致。
- 先普通 PTQ、再可选 Po2：无法保证首版产物满足部署定点约束。
