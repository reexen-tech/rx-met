"""SmoothQuant Step 1 (calibration + smoothing) for the REEX Q64 pipeline.

The target weight quantization is handled later by ``llama-quantize``, so this
package only rewrites the float checkpoint.
"""

# === REEX_SMOOTHQUANT BEGIN: package exports for SmoothQuant Step 1 ===

from .adapters import SmoothGroup, is_decoder_layer, smooth_groups
from .calibration import get_act_scales, resolve_calib_dataset
from .defaults import DEFAULT_ALPHA, default_alpha
from .runner import SmoothQuantResult, run_smoothquant
from .smooth import smooth_ln_fcs, smooth_lm

__all__ = [
    "DEFAULT_ALPHA",
    "SmoothGroup",
    "SmoothQuantResult",
    "default_alpha",
    "get_act_scales",
    "is_decoder_layer",
    "resolve_calib_dataset",
    "run_smoothquant",
    "smooth_groups",
    "smooth_ln_fcs",
    "smooth_lm",
]

# === REEX_SMOOTHQUANT END ===
