# AIMET native runtime

This directory contains the AIMET native implementation maintained by rx-met.
It is organized by runtime responsibility rather than by the layout of the
original AIMET repository.

## Source origin

The initial C++ and CUDA implementation was imported without content changes
from Qualcomm AIMET tag `2.17.0`, commit:

```text
0e679b705818df7f408971e34693b8098f0971a9
```

The imported source came from these AIMET modules:

- `ModelOptimizations/DlQuantization`
- `ModelOptimizations/PyModelOptimizations`
- `TrainingExtensions/onnx/src`

After import, these files are owned and maintained as rx-met source. They may be
changed directly when rx-met behavior requires it; synchronization with the
original AIMET repository is not an interface or maintenance requirement.

## Layout

- `common/`: shared quantization implementation and the `_libpymo` binding.
- `onnx/`: `libquant_info` and the ONNX Runtime custom-op implementation.
- `third_party/onnxruntime/`: immutable ONNX Runtime headers and license files.
- `tests/`: smoke tests for the installed native runtime interface.

The build installs these files into the `aimet_common` Python package:

- `_libpymo.cpython-310-x86_64-linux-gnu.so`
- `libquant_info.cpython-310-x86_64-linux-gnu.so`
- `libaimet_onnxrt_ops.so`

Use `scripts/build_aimet_native.sh` as the build interface. The supported ABI is
CPython 3.10 on Linux x86_64, with CPU and CUDA implementations selected through
`RX_MET_ENABLE_CUDA`.
