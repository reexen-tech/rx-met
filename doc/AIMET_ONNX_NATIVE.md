# AIMET ONNX native runtime

rx-met builds the native runtime required by its `aimet_common` and
`aimet_onnx` Python packages directly from repository source. No prebuilt
AIMET shared library is downloaded, copied into the repository, or added to
the wheel.

## Ownership and source baseline

The implementation under `native/aimet` is rx-met-maintained source. Its
initial C++ and CUDA implementation was imported without content changes from
Qualcomm AIMET tag `2.17.0`, commit
`0e679b705818df7f408971e34693b8098f0971a9`, matching the fork's Python
baseline.

The initial source mapping was:

- `ModelOptimizations/DlQuantization` -> `native/aimet/common`;
- `ModelOptimizations/PyModelOptimizations` ->
  `native/aimet/common/bindings/python`;
- `TrainingExtensions/onnx/src` -> `native/aimet/onnx/src`.

This mapping records provenance; it is not a synchronization contract. The
native implementation may be changed directly for rx-met requirements and is
organized by runtime responsibility instead of mirroring the original AIMET
repository layout. Git tracks changes to this source, so there is no separate
AIMET source manifest.

The ONNX Runtime 1.23.2 headers under
`native/aimet/third_party/onnxruntime` remain third-party inputs. Their license,
notices, version, and `SHA256SUMS` are kept with the headers and verified before
every build.

## Outputs

The native build installs three runtime files into `aimet_common`:

- `_libpymo.cpython-310-x86_64-linux-gnu.so`;
- `libquant_info.cpython-310-x86_64-linux-gnu.so`;
- `libaimet_onnxrt_ops.so`.

The supported ABI is CPython 3.10 on Linux x86_64. Build dependencies are a
C++17 compiler, CMake, Ninja, Eigen3, pybind11, and the matching ONNX Runtime
headers. CUDA builds additionally require the CUDA Toolkit.

## Build and smoke test

CPU:

```bash
RX_MET_ENABLE_CUDA=0 ./scripts/build_aimet_native.sh
python3 native/aimet/tests/smoke_onnx_runtime.py \
    --provider CPUExecutionProvider
```

CUDA:

```bash
RX_MET_ENABLE_CUDA=1 \
RX_MET_CUDA_ARCHITECTURES='80;90' \
./scripts/build_aimet_native.sh
python3 native/aimet/tests/smoke_onnx_runtime.py \
    --provider CUDAExecutionProvider
```

The build does not access the network. Its supported environment variables
are:

- `RX_MET_ENABLE_CUDA`: `0` for CPU or `1` for CUDA;
- `RX_MET_AIMET_BUILD_DIR`: CMake build directory;
- `RX_MET_AIMET_INSTALL_DIR`: destination package directory, defaulting to
  `aimet_common` in the repository;
- `RX_MET_PYTHON`: Python interpreter, defaulting to `python3`;
- `RX_MET_CUDA_ARCHITECTURES`: CMake CUDA architecture list;
- `RX_MET_ONNXRUNTIME_ROOT`: deliberate override for testing another ONNX
  Runtime header tree.

`scripts/build_wheel.sh` invokes this native build automatically. The resulting
wheel is platform-specific (`cp310-cp310-linux_x86_64`) and the wheel build
fails if any required native output is absent.
