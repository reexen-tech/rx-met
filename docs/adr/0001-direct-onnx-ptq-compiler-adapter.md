# Use a dedicated direct-ONNX PTQ compiler adapter

<!-- Status: accepted -->

已有 `aimet_torch/rx_export` 是面向 `aimet_torch` sim、单输入模型和 BSRNN/GRU 历史规则的导出链，不能作为已有 ONNX 模型的通用入口。因此 ONNX PTQ 使用独立的 `aimet_onnx/rx_ptq` adapter：保留原 ONNX 图契约，输出 clean ONNX 与 compiler encodings，并冻结当前 AIMET 2.17/native 基线；Torch 导出器和 native 版本暂不改造。

## Considered Options

- 强行扩展 Torch 导出器：会把 Torch 节点命名、GRU 特化和 ONNX 直量化混在一起。
- 升级到 official 2.36.0：会同时引入 Python/native ABI、构建和 encoding 行为变化，超出本次 PTQ 目标。
- 独立 ONNX adapter：边界清晰，可按 ONNX graph 建立 node-IO 到 compiler schema 的映射。
