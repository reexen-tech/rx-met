"""Compare CPU/GPU PTQ artifacts exactly; write differences before returning 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from google.protobuf.json_format import MessageToDict
import numpy as np
import onnx
from onnx import numpy_helper


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_json(left, right):
    differences = []

    def visit(a, b, path):
        if type(a) is not type(b):
            differences.append({"path": path, "left": a, "right": b, "kind": "type"})
        elif isinstance(a, dict):
            for key in sorted(a.keys() | b.keys()):
                child = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if key not in a or key not in b:
                    differences.append({"path": child, "kind": "missing",
                                        "left_present": key in a, "right_present": key in b})
                else:
                    visit(a[key], b[key], child)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append({"path": path, "kind": "length", "left": len(a), "right": len(b)})
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + f"/{index}")
        elif a != b:
            differences.append({"path": path, "left": a, "right": b, "kind": "value"})

    visit(left, right, "")
    return {"equal": not differences, "differences": differences}


def compare_arrays(left, right):
    differences = []

    def scalar(value):
        value = value.item() if isinstance(value, np.generic) else value
        if isinstance(value, float) and not np.isfinite(value):
            return {"nonfinite": str(value)}
        if isinstance(value, complex):
            return {"real": scalar(value.real), "imag": scalar(value.imag)}
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        return value

    for name in sorted(left.keys() | right.keys()):
        if name not in left or name not in right:
            differences.append({"name": name, "kind": "missing",
                                "left_present": name in left, "right_present": name in right})
            continue
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.shape != b.shape or a.dtype != b.dtype:
            differences.append({"name": name, "kind": "shape_or_dtype",
                                "left_shape": list(a.shape), "right_shape": list(b.shape),
                                "left_dtype": str(a.dtype), "right_dtype": str(b.dtype)})
        elif not np.array_equal(a, b):
            item = {"name": name, "kind": "value", "shape": list(a.shape),
                    "dtype": str(a.dtype), "different_elements": int(np.count_nonzero(a != b)),
                    "element_differences": [
                        {"index": [int(value) for value in index], "left": scalar(a[index]), "right": scalar(b[index])}
                        for index in zip(*np.nonzero(a != b))
                    ] if a.ndim else [{"index": [], "left": scalar(a[()]), "right": scalar(b[()])}]}
            if a.dtype.kind in "biufc":
                delta = np.abs(a.astype(np.complex128 if a.dtype.kind == "c" else np.float64)
                               - b.astype(np.complex128 if b.dtype.kind == "c" else np.float64))
                item["nonfinite_elements"] = int(np.count_nonzero(~np.isfinite(delta)))
                item["max_abs"] = (float(delta.max()) if delta.size else 0.0) if not item["nonfinite_elements"] else None
                item["rmse"] = (float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0) if not item["nonfinite_elements"] else None
            differences.append(item)
    return {"equal": not differences, "differences": differences,
            "left_count": len(left), "right_count": len(right)}


def compare_onnx(left, right):
    models = [onnx.load(left), onnx.load(right)]
    structures, initializers = [], []
    for model in models:
        tensors = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
        structure = MessageToDict(model, preserving_proto_field_name=True)
        structure["graph"]["initializer"] = [
            {"name": tensor.name, "data_type": tensor.data_type, "dims": list(tensor.dims)}
            for tensor in model.graph.initializer
        ]
        structures.append(structure)
        initializers.append(tensors)
    structure = compare_json(*structures)
    values = compare_arrays(*initializers)
    return {"equal": structure["equal"] and values["equal"], "structure": structure,
            "initializers": values, "file_equal": sha256(left) == sha256(right)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-onnx", type=Path, required=True)
        parser.add_argument(f"--{side}-encodings", type=Path, required=True)
        parser.add_argument(f"--{side}-outputs", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.left_outputs) != bool(args.right_outputs):
        parser.error("both --left-outputs and --right-outputs are required together")
    report = {
        "comparison": "exact", "onnx": compare_onnx(args.left_onnx, args.right_onnx),
        "encodings": compare_json(json.loads(args.left_encodings.read_text()),
                                  json.loads(args.right_encodings.read_text())),
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                   for name, path in vars(args).items() if name != "report" and path is not None},
    }
    if args.left_outputs:
        with np.load(args.left_outputs, allow_pickle=False) as left, \
             np.load(args.right_outputs, allow_pickle=False) as right:
            report["outputs"] = compare_arrays(dict(left), dict(right))
    report["equal"] = all(report[name]["equal"] for name in ("onnx", "encodings", "outputs") if name in report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Artifacts {'match' if report['equal'] else 'differ'}: {args.report.resolve()}")
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
