"""Read-only test checks for the feature-input KWS model and its ADA200 encodings.

These checks cover the reviewed importer/lowering contracts. They do not run
the compiler, memory planner, LUT evaluator, or device inference.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from aimet_common.compiler_artifacts import (
    load_compiler_encodings,
    validate_compiler_artifacts,
)


_OPERATORS = {
    "Add", "Cast", "Clip", "Concat", "Constant", "Conv", "Div", "Exp",
    "Flatten", "Gather", "GRU", "Identity", "Mul", "ReduceMax", "ReduceSum",
    "Relu", "Reshape", "Shape", "Slice", "Squeeze", "Sub", "Transpose",
    "Unsqueeze",
}


def _attributes(node):
    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


def _shape(value):
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def validate_ada200_artifacts(onnx_path, encodings_path):
    """Reject known unsupported graph forms and inconsistent quantized operands."""
    onnx_path, encodings_path = Path(onnx_path), Path(encodings_path)
    common = validate_compiler_artifacts(
        onnx_path, encodings_path, require_po2=True, require_complete=True,
    )
    model = onnx.load(str(onnx_path))
    document = load_compiler_encodings(encodings_path, require_po2=True)
    parameters = document["param_encodings"]
    activations = document["activation_encodings"]
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            value = _attributes(node).get("value")
            if isinstance(value, onnx.TensorProto):
                constants[node.output[0]] = numpy_helper.to_array(value)
    values = {value.name: value for value in (
        *model.graph.input, *model.graph.value_info, *model.graph.output,
    )}
    consumers = Counter(name for node in model.graph.node for name in node.input if name)
    graph_outputs = {value.name for value in model.graph.output}
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    inputs = [value for value in model.graph.input if value.name not in constants]
    require(len(inputs) == 1 and _shape(inputs[0]) == [1, 49, 20],
            "Expected one static MFCC input with shape [1,49,20]")
    require(len(inputs) == 1 and inputs[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT,
            "Expected FLOAT32 MFCC input")
    require(len(model.graph.output) == 1 and _shape(model.graph.output[0]) == [1, 12],
            "Expected one classification-logit output with shape [1,12]")
    counts = Counter(node.op_type for node in model.graph.node)
    query_convs = []
    for node in model.graph.node:
        require(node.domain in ("", "ai.onnx"), f"{node.name}: unsupported domain {node.domain!r}")
        require(node.op_type in _OPERATORS,
                f"{node.name}: unreviewed or unsupported ADA200 KWS operator {node.op_type}")
        require("speech_features" not in node.name,
                f"{node.name}: preprocessing must be outside the deployment graph")
        attributes = _attributes(node)
        entry = activations.get(node.name, {})
        if node.op_type == "Conv":
            require(attributes.get("group", 1) == 1, f"{node.name}: expected group=1")
            weight = constants.get(node.input[1]) if len(node.input) > 1 else None
            require(weight is not None and weight.ndim == 4,
                    f"{node.name}: Conv requires a constant rank-4 weight")
            if "query_projs" in node.name:
                query_convs.append(node.name)
                require(weight is not None and tuple(weight.shape[2:]) == (1, 1),
                        f"{node.name}: query projection must be a 1x1 convolution")
            if len(node.input) > 2 and node.input[2]:
                bias = parameters.get(node.input[2], {})
                input_record = entry.get("input", {}).get("0", {})
                weight_record = parameters.get(node.input[1], {})
                require(bias.get("dtype") == "INT32",
                        f"{node.name}: aligned Conv bias must use INT32")
                if all("scale" in record for record in (bias, input_record, weight_record)):
                    expected = np.asarray(input_record["scale"]) * np.asarray(weight_record["scale"])
                    require(np.allclose(np.asarray(bias["scale"]), expected, rtol=1e-7, atol=0),
                            f"{node.name}: bias scale must equal input scale times weight scale")
                else:
                    require(False, f"{node.name}: missing input, weight, or bias scale")
        elif node.op_type == "GRU":
            require(attributes.get("linear_before_reset", 0) == 1,
                    f"{node.name}: ADA200 requires linear_before_reset=1")
            require(attributes.get("layout", 0) == 0,
                    f"{node.name}: ADA200 requires sequence-first ONNX GRU layout")
            require(attributes.get("direction", b"forward") == b"bidirectional",
                    f"{node.name}: expected the original bidirectional GRU")
            for index in (1, 2, 3, 5):
                if index < len(node.input) and node.input[index]:
                    require(node.input[index] in constants,
                            f"{node.name}: GRU input {index} must be constant")
            if len(node.input) > 4:
                require(not node.input[4], f"{node.name}: variable sequence lengths are not supported")
            for output in node.output[1:]:
                require(not output or (not consumers[output] and output not in graph_outputs),
                        f"{node.name}: bidirectional GRU hidden-state consumers are unsupported")
        elif node.op_type in ("ReduceMax", "ReduceSum"):
            require(attributes.get("keepdims", 1) == 1,
                    f"{node.name}: keep dimensions for ADA200 reductions")
            source = values.get(node.input[0])
            require(source is not None and len(_shape(source)) == 3,
                    f"{node.name}: Attention reduction input must have static rank 3")
            axes = attributes.get("axes")
            if axes is None and len(node.input) > 1:
                axes = constants.get(node.input[1])
            require(axes is not None and np.asarray(axes).size == 1,
                    f"{node.name}: expected one constant reduction axis")
        for kind in ("input", "output"):
            for record in entry.get(kind, {}).values():
                require(record["dtype"].startswith(("INT", "UINT")) and record["bitwidth"] in (8, 16),
                        f"{node.name}: deployment activations must use integer 8/16-bit encodings")

    require(counts["GRU"] == 2, "Expected the original two GRU layers")
    require(len(query_convs) == 4, "Expected four independent query projections")
    require(counts["Exp"] == 4, "Expected four explicitly decomposed attention heads")
    require(counts["Div"] >= 4, "Expected explicit normalization of each attention head")
    for name in constants:
        require("speech_features" not in name and not name.endswith("_fix"),
                f"Unexpected preprocessing or importer-derived weight: {name}")
    if errors:
        raise ValueError("ADA200 KWS artifact checks failed:\n  - " + "\n  - ".join(errors))
    return {
        "status": "passed", "scope": "ada200_kws_source_contract",
        "compiler_execution": "not_run", "device_execution": "not_run",
        "input_shape": [1, 49, 20], "operator_counts": dict(sorted(counts.items())),
        "query_projections": query_convs,
        "artifact_validation": common,
        "remaining_runtime_checks": [
            "Compiler lowering, physical layout, memory planning, and code generation",
            "Exp/reciprocal LUT error, reduction accumulator range, and device accuracy",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--encodings", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.report and args.report.resolve() in (args.onnx.resolve(), args.encodings.resolve()):
        parser.error("The report must not overwrite an input artifact")
    try:
        report = validate_ada200_artifacts(args.onnx, args.encodings)
    except (ValueError, RuntimeError, OSError) as error:
        report = {"status": "failed", "error": str(error)}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
