# -*- mode: python -*-
# =============================================================================
"""Classify compute vs data-movement / index / control ops for QuantSim defaults.

Fixed-point compilers need input and output encodings on every compute node.
AIMET JSON cannot default ``is_input_quantized`` (semantic validation rejects it),
so configurators call ``is_compute_op`` / ``apply_compute_op_io_defaults`` instead
of enumerating every ONNX type in ``op_type``.
"""

from typing import Dict, Optional, Sequence

# Same membership as aimet_common.onnx._utils._is_grid_preserving_op.
# Kept here so this module does not import onnx.
_GRID_PRESERVING_OPS = frozenset(
    {
        "BatchToSpace",
        "Col2Im",
        "Compress",
        "DepthToSpace",
        "Dropout",
        "Expand",
        "Flatten",
        "Gather",
        "GatherElements",
        "GatherND",
        "Identity",
        "MaxPool",
        "MaxRoiPool",
        "NonZero",
        "Pad",
        "ReduceMax",
        "ReduceMin",
        "Reshape",
        "Slice",
        "SpaceToBatch",
        "SpaceToDepth",
        "Split",
        "SplitToSequence",
        "Squeeze",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
    }
)


# Index, shape, control-flow, boolean, and integer-address math.
# Grid-preserving ops are handled separately via ``is_grid_preserving_op``.
_INDEX_CONTROL_BOOL_OPS = frozenset(
    {
        "And",
        "ArgMax",
        "ArgMin",
        "BitwiseAnd",
        "BitwiseNot",
        "BitwiseOr",
        "BitwiseXor",
        "Cast",
        "Ceil",
        "Constant",
        "ConstantOfShape",
        "ElementWiseAnd",
        "ElementWiseEqual",
        "ElementWiseFloorDiv",
        "ElementWiseGreater",
        "ElementWiseGreaterEqual",
        "ElementWiseLess",
        "ElementWiseLessEqual",
        "ElementWiseNot",
        "ElementWiseOr",
        "Equal",
        "EyeLike",
        "Floor",
        "FloorDiv",
        "Greater",
        "GreaterEqual",
        "GreaterOrEqual",
        "If",
        "Less",
        "LessEqual",
        "LessOrEqual",
        "Loop",
        "NonMaxSuppression",
        "Not",
        "OneHot",
        "OptionalGetElement",
        "Or",
        "Range",
        "Round",
        "Scan",
        "SequenceAt",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
        "Shape",
        "Size",
        "Where",
        "Xor",
    }
)

# Own their I/O encodings; do not apply generic compute-op defaults.
_SKIP_MODULE_TYPE_NAMES = frozenset({"QuantGRU"})


def is_grid_preserving_op(op_type: str) -> bool:
    """Return True if ``op_type`` preserves the quantization grid (reshape/index)."""
    return bool(op_type) and op_type in _GRID_PRESERVING_OPS


def is_non_compute_op(op_type: str) -> bool:
    """Return True if ``op_type`` should not get default compute I/O quantization."""
    if not op_type:
        return True
    return is_grid_preserving_op(op_type) or op_type in _INDEX_CONTROL_BOOL_OPS


def is_compute_op(op_type: str) -> bool:
    """Return True if ``op_type`` is a value-changing compute operator."""
    if not op_type:
        return False
    return not is_non_compute_op(op_type)


def skips_generic_quantizers(module) -> bool:
    """True if the module owns encodings and must not get AIMET I/O/param defaults.

    QuantGRU (and its ``QuantizedQuantGRU`` wrapper) keep encodings in
    ``GRU_config``; generic QuantSim defaults must not re-enable them.
    """
    if module is None:
        return False
    for klass in type(module).mro():
        name = getattr(klass, "__name__", "")
        if name in _SKIP_MODULE_TYPE_NAMES:
            return True
        if name.startswith("Quantized") and name[len("Quantized") :] in _SKIP_MODULE_TYPE_NAMES:
            return True
    return False


def is_compute_module(
    module, op_types: Optional[Sequence[str]] = None
) -> bool:
    """Classify a torch/ONNX-mapped module.

    * Skip QuantGRU (owns encodings via GRU_config).
    * If any mapped ONNX/backend type is compute, treat the module as compute.
    * No mapping: fail-open as compute so custom leaves are not missed.
    """
    if module is None or skips_generic_quantizers(module):
        return False
    if op_types:
        return any(is_compute_op(op_type) for op_type in op_types)
    return True


def apply_compute_op_io_defaults(
    op_type: str, op_config: Optional[Dict] = None
) -> Dict:
    """Fill missing input/output flags for a compute op; leave existing keys intact."""
    result = dict(op_config or {})
    if is_compute_op(op_type):
        result.setdefault("is_input_quantized", True)
        result.setdefault("is_output_quantized", True)
    return result
