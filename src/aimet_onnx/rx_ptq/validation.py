"""Validate live ONNX quantizers without changing their configuration."""

import math

from aimet_common.defs import QuantizationDataType


def validate_quantizer_state(sim, *, allow_pending_bias: bool = False) -> None:
    """Reject missing required quantizers and invalid enabled integer grids."""
    quantizers = sim.qc_quantize_op_dict
    native = getattr(sim, "_rx_native_grus", {})
    missing_native = [name for name in getattr(sim, "_rx_expected_native_grus", ())
                      if name not in native or not native[name].module.use_quantization]
    if missing_native:
        raise RuntimeError("Required native GRU quantization is missing or disabled: " + ", ".join(missing_native))
    for name, adapter in getattr(sim, "_rx_native_grus", {}).items():
        if adapter.module.use_quantization and not adapter.module.is_calibrated():
            raise RuntimeError(f"GRU {name}: internal quantizers are not calibrated")
    pending_bias = set(getattr(sim, "_rx_calibration_bias_quantizers", ()))
    if pending_bias and not allow_pending_bias:
        raise RuntimeError("Bias quantizers still await scale alignment: " + ", ".join(sorted(pending_bias)))
    missing = [
        name for name in getattr(sim, "_rx_expected_quantizers", ())
        if name not in quantizers or (
            not quantizers[name].enabled and not (allow_pending_bias and name in pending_bias)
        )
    ]
    if missing:
        raise RuntimeError("Required quantizers are missing or disabled: " + ", ".join(sorted(missing)))
    for name, quantizer in quantizers.items():
        if not quantizer.enabled or quantizer.data_type != QuantizationDataType.int:
            continue
        encodings = quantizer.get_encodings()
        if not encodings:
            raise RuntimeError(f"Enabled quantizer has no encodings: {name}")
        for channel, encoding in enumerate(encodings):
            if not math.isfinite(encoding.delta) or encoding.delta <= 0:
                raise ValueError(f"{name}[{channel}]: scale must be finite and positive")
            if not all(math.isfinite(value) for value in (encoding.min, encoding.max, encoding.offset)):
                raise ValueError(f"{name}[{channel}]: encoding range and offset must be finite")
