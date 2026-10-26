"""Strict read-only acceptance of saved compiler encoding artifacts."""

import hashlib
import json
import math
from pathlib import Path

from .compiler_encodings import validate_compiler_encodings


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate compiler JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite compiler JSON number: {value}")
    return number


def _invalid_constant(value):
    raise ValueError(f"Invalid compiler JSON constant: {value}")


def load_compiler_encodings(path, *, require_po2=False):
    """Read a current-schema artifact without migration, repair or broadcasting."""
    document = json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=_unique_object, parse_float=_finite_float,
                          parse_constant=_invalid_constant)
    validate_compiler_encodings(document, require_po2=require_po2, strict=True)
    return document


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_compiler_artifacts(onnx_path, encodings_path, *, require_po2=True, require_complete=True):
    """Validate the saved schema, graph mapping and shape metadata as-is.

    This does not invoke a downstream compiler or claim numerical execution
    equivalence. Runtime reload checks remain a separate workflow step.
    """
    import onnx
    from .compiler_graph_validation import validate_compiler_encoding_graph
    from .onnx_shapes import validate_onnx_shapes

    onnx_path, encodings_path = Path(onnx_path), Path(encodings_path)
    document = load_compiler_encodings(encodings_path, require_po2=require_po2)
    model = onnx.load(str(onnx_path))
    shapes = validate_onnx_shapes(model)
    onnx.checker.check_model(str(onnx_path), full_check=True)
    validate_compiler_encoding_graph(model, document, require_complete=require_complete)
    return {"status": "passed", "schema_version": document["schema_version"],
            "require_po2": require_po2, "require_complete": require_complete,
            "onnx": str(onnx_path.resolve()),
            "encodings": str(encodings_path.resolve()),
            "sha256": {"onnx": _sha256(onnx_path), "encodings": _sha256(encodings_path)},
            "onnx_shapes": shapes, "activation_entries": len(document["activation_encodings"]),
            "parameter_entries": len(document["param_encodings"])}
