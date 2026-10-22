"""Execution-provider policy shared by ONNX PTQ calibration and evaluation."""


def ptq_execution_providers(providers):
    """Use full FP32 CUDA matmul unless the caller explicitly requests TF32."""
    normalized = []
    for provider in providers:
        name, options = provider if isinstance(provider, tuple) else (provider, {})
        if name == "CUDAExecutionProvider":
            options = dict(options)
            options.setdefault("use_tf32", "0")
            normalized.append((name, options))
        else:
            normalized.append(provider)
    return normalized
