from __future__ import annotations

import logging
from typing import Tuple

from .encodings_ops import postprocess_encodings_inplace
from .io_utils import load_encodings, save_encodings
from .onnx_graph import fix_slice_end_constants, postprocess_onnx_graph_inplace
from .renaming import rename_gru_initializers_and_change_gru_json_format

logger = logging.getLogger(__name__)


def postprocess_encodings_file(
    encodings_path: str,
    output_encodings_path: str,
    verbose: bool = True,
) -> str:
    enc = load_encodings(encodings_path)
    enc, changed = postprocess_encodings_inplace(enc, verbose=verbose)
    if changed or output_encodings_path != encodings_path:
        save_encodings(enc, output_encodings_path)
    return output_encodings_path


def postprocess_all(
    *,
    onnx_path: str,
    encodings_path: str,
    input_shape: Tuple[int, int, int],
    output_onnx_path: str | None = None,
    output_encodings_path: str | None = None,
    verbose: bool = True,
) -> Tuple[str, str]:
    if verbose:
        logger.info("=== Pipeline G1: postprocess_onnx_graph_inplace ===")
    postprocess_onnx_graph_inplace(onnx_path, input_shape)

    if verbose:
        logger.info("=== Pipeline G2: fix_slice_end_constants ===")
    fix_slice_end_constants(onnx_path, verbose=verbose)

    if verbose:
        logger.info("=== Pipeline E*: load_encodings ===")
    enc = load_encodings(encodings_path)
    enc, changed_any = postprocess_encodings_inplace(enc, verbose=verbose)

    enc_out = output_encodings_path or encodings_path
    if changed_any or enc_out != encodings_path:
        if verbose:
            logger.info("=== Pipeline E*: save_encodings -> %s ===", enc_out)
        save_encodings(enc, enc_out)
    else:
        enc_out = encodings_path

    if verbose:
        logger.info("=== Pipeline X2: rename_gru_initializers_and_change_gru_json_format ===")
    onnx_out = output_onnx_path or onnx_path
    return rename_gru_initializers_and_change_gru_json_format(
        onnx_path=onnx_path,
        encodings_path=enc_out,
        output_onnx_path=onnx_out,
        output_encodings_path=enc_out,
        verbose=verbose,
    )
