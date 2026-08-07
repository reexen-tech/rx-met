"""SmoothQuant Step 1 (calibration + smoothing) for the REEX Q64 pipeline.

The target weight quantization is handled later by ``llama-quantize``, so this
package only rewrites the float checkpoint.
"""

# === REEX_SMOOTHQUANT BEGIN: package exports for SmoothQuant Step 1 ===

from .adapters import (
    CalibrationTargets,
    RoutingTarget,
    SmoothGroup,
    is_decoder_layer,
    qwen35_calibration_targets,
    smooth_groups,
)
from .calibration import (
    CalibrationState,
    calibrate_progressively,
    expert_coverage_summary,
    get_act_scales,
    resolve_calib_dataset,
)
from .defaults import DEFAULT_ALPHA, default_alpha
from .runner import SmoothQuantResult, run_smoothquant
from .smooth import chunked_absmax_last_dim, smooth_ln_fcs, smooth_lm

__all__ = [
    "DEFAULT_ALPHA",
    "CalibrationState",
    "CalibrationTargets",
    "RoutingTarget",
    "SmoothGroup",
    "SmoothQuantResult",
    "calibrate_progressively",
    "chunked_absmax_last_dim",
    "default_alpha",
    "expert_coverage_summary",
    "get_act_scales",
    "is_decoder_layer",
    "qwen35_calibration_targets",
    "resolve_calib_dataset",
    "run_smoothquant",
    "smooth_groups",
    "smooth_ln_fcs",
    "smooth_lm",
]

# === REEX_SMOOTHQUANT END ===
