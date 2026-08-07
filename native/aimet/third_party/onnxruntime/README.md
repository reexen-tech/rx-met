# ONNX Runtime headers

The headers in this directory come from ONNX Runtime `1.23.2`. Public headers
were extracted from the official `onnxruntime-linux-x64-1.23.2` release archive.
The CUDA provider headers come from tag `v1.23.2`:

- `include/core/providers/cuda/cuda_context.h`
- `include/core/providers/cuda/cuda_resource.h`

The files are vendored so CPU and CUDA native builds require no network access.
They are external, immutable build inputs and are checked by `SHA256SUMS` before
compilation. ONNX Runtime is licensed under the MIT license in `LICENSE`; its
notices are retained in `ThirdPartyNotices.txt`.
