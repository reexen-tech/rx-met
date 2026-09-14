#!/usr/bin/env python3
"""Exercise AIMET ONNX native imports, custom ops, and calibration."""

import argparse
import platform
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "aimet_common").is_dir() and (_parent / "aimet_onnx").is_dir():
        sys.path.insert(0, str(_parent))
        break

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def _make_model() -> onnx.ModelProto:
    model_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    model_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    weight = numpy_helper.from_array(
        np.array(
            [
                [0.25, -0.50, 0.75],
                [1.00, 0.50, -0.25],
                [-0.75, 0.25, 0.50],
                [0.50, 1.00, -1.00],
            ],
            dtype=np.float32,
        ),
        name="weight",
    )
    matmul = helper.make_node("MatMul", ["input", "weight"], ["linear"], name="matmul")
    relu = helper.make_node("Relu", ["linear"], ["output"], name="relu")
    graph = helper.make_graph(
        [matmul, relu],
        "aimet_onnx_native_smoke",
        [model_input],
        [model_output],
        [weight],
    )
    model = helper.make_model(
        graph,
        producer_name="rx-met-native-smoke",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main(provider: str) -> None:
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        raise RuntimeError(
            "AIMET ONNX native smoke requires CPython 3.10-3.12 on Linux x86_64"
        )

    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise RuntimeError(
            f"{provider} is unavailable; ONNX Runtime providers: {available_providers}"
        )

    from aimet_common import _libpymo, libquant_info
    from aimet_onnx import QuantizationSimModel
    from aimet_onnx.utils import create_ort_session_options_with_aimet_custom_ops

    native_paths = [Path(_libpymo.__file__), Path(libquant_info.__file__)]
    custom_op = native_paths[1].parent / "libaimet_onnxrt_ops.so"
    native_paths.append(custom_op)
    missing = [str(path) for path in native_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing AIMET ONNX native files: {missing}")

    create_ort_session_options_with_aimet_custom_ops()
    libquant_info.QcQuantizeInfo()

    sample = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)
    sim = QuantizationSimModel(
        _make_model(),
        dummy_input={"input": sample},
        providers=[provider],
    )
    sim.compute_encodings([{"input": sample}, {"input": sample * 0.5}])
    enabled_quantizers = [
        quantizer
        for quantizer in sim.qc_quantize_op_dict.values()
        if quantizer.enabled
    ]
    if not enabled_quantizers or not all(
        quantizer.is_initialized() for quantizer in enabled_quantizers
    ):
        raise RuntimeError("AIMET ONNX quantizers were not initialized")

    (output,) = sim.session.run(None, {"input": sample})
    if output.shape != (1, 3) or not np.isfinite(output).all():
        raise RuntimeError(f"unexpected QuantSim output: shape={output.shape}")

    print("AIMET ONNX native smoke passed")
    print("  provider:", provider)
    print("  custom-op domain:", sim._op_domain)
    print("  _libpymo:", native_paths[0])
    print("  libquant_info:", native_paths[1])
    print("  custom op:", native_paths[2])
    print("  quantizers:", len(enabled_quantizers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
        default="CPUExecutionProvider",
    )
    main(parser.parse_args().provider)
