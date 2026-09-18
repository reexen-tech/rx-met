# Quantized operators

This directory contains first-party operators that replace framework modules
during an rx-met quantization workflow.

Each operator owns its native implementation, Python binding, build metadata,
tests, and operator-specific documentation. Model discovery and replacement
remain in `src/aimet_torch`; operator execution and quantization internals stay
behind the operator's Python interface in this directory.

Current operators:

- `quant-gru/`: replacement implementation for PyTorch GRU modules.

Future recurrent operators, such as QuantLSTM, should use a sibling directory
and share implementation only when the shared code has a stable interface.
