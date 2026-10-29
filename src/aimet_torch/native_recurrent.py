"""Single registration seam for operator-owned recurrent quantization.

Operators implement load_bitwidth_config, calibration_context, enable_pot2,
export/load_quant_params_{to/from}_aimet_format and set_quant_params_locked.
This module knows discovery and dispatch, not gates, grids or native JSON.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Callable
import json


@dataclass(frozen=True)
class RecurrentOperator:
    module_type: type
    package: str
    onnx_type: str
    register_onnx: Callable[..., None]
    normalize_onnx: Callable[[str], None]
    custom_opsets: Callable[[], dict[str, int]] | None = None
    prune_encodings: Callable[[str], None] | None = None


@lru_cache(maxsize=1)
def recurrent_operators() -> tuple[RecurrentOperator, ...]:
    """Installed optional operators. Keep all package-specific hooks here."""
    operators = []
    for package, class_name, onnx_type, normalize, opsets, prune in (
        ("quant_gru", "QuantGRU", "GRU", "normalize_quant_gru_onnx_to_optimized_baseline",
         "get_quant_gru_custom_opsets", "prune_quant_gru_raw_l0_param_encodings"),
        ("quant_lstm", "QuantLSTM", "LSTM", "normalize_quant_lstm_onnx", None, None),
    ):
        try:
            library = import_module(package)
        except ImportError:
            continue
        operators.append(RecurrentOperator(
            getattr(library, class_name), package, onnx_type,
            getattr(library, f"ensure_{package}_onnx_registered"),
            getattr(library, normalize),
            getattr(library, opsets) if opsets else None,
            getattr(library, prune) if prune else None,
        ))
    return tuple(operators)


def native_recurrent_types() -> tuple[type, ...]:
    return tuple(op.module_type for op in recurrent_operators())


def named_native_recurrent(model):
    """Yield (module path, module, registration), including QuantSim subclasses."""
    for name, module in model.named_modules():
        for op in recurrent_operators():
            if isinstance(module, op.module_type):
                yield name, module, op
                break


def validate_native_po2_targets(model):
    """Find enabled native operators and reject missing calibration before Po2."""
    targets = []
    for name, module, op in named_native_recurrent(model):
        if not module.use_quantization:
            continue
        if not module.is_calibrated():
            raise RuntimeError(f"{op.module_type.__name__} is not calibrated: {name!r}")
        targets.append((name, module, op))
    return targets


def apply_native_po2(targets, method, tolerance):
    """Let each operator enforce its locking rules and derive its own Po2 grids."""
    for _, module, _ in targets:
        module.enable_pot2(method=method, tolerance=tolerance)


def verify_native_po2(targets):
    """Synchronize public encodings and validate effective grids in every direction."""
    from aimet_common.compiler_encodings import finalize_compiler_encodings
    from aimet_common.compiler_graph_validation import validate_native_recurrent_encoding_coverage

    results = []
    for name, module, op in targets:
        # GRU resolves pending mode changes when exporting its public encodings.
        document = module.export_quant_params_to_aimet_format(
            {}, module_name=name, for_onnx=False,
        )
        activations = document.get("activation_encodings", {})
        entry = activations.get(name)
        validate_native_recurrent_encoding_coverage(
            entry, op.onnx_type, bidirectional=module.bidirectional, name=name,
        )
        directions = ("internal_ops", "internal_ops_reverse")[:2 if module.bidirectional else 1]
        parameters = document.get("param_encodings", {})
        suffixes = ["weight_ih.weight", "weight_hh.weight"]
        if module.bias:
            suffixes.append("bias")
        for suffix in suffixes:
            if f"{name}.{suffix}" not in parameters:
                raise ValueError(f"Missing native parameter encoding: {name}.{suffix}")

        # Reuse the compiler's integer-grid contract, including exact Po2 scales,
        # zero points and per-channel ranges; floating-point records are exempt.
        normalized = finalize_compiler_encodings(document, require_po2=True)
        entry = normalized["activation_encodings"][name]
        records = [record for field in ("input", "output")
                   for record in entry[field].values()]
        records.extend(operation["output"][0] for field in directions
                       for operation in entry[field].values())
        records.extend(normalized["param_encodings"].values())
        integer_records = [record for record in records
                           if not record["dtype"].startswith("FLOAT")]

        def scale_count(value):
            return sum(map(scale_count, value)) if isinstance(value, list) else 1

        if not integer_records:
            raise ValueError(f"No native integer encodings were exported: {name!r}")
        results.append({
            "name": name,
            "operator": op.onnx_type,
            "encoding_count": len(integer_records),
            "scale_count": sum(scale_count(record["scale"]) for record in integer_records),
        })
    return results


@contextmanager
def calibrate_native_recurrent(model):
    """Let each operator own its calibration lifecycle and failure cleanup."""
    with ExitStack() as stack:
        for _, module, _ in named_native_recurrent(model):
            stack.enter_context(module.calibration_context())
        yield


def configure_recurrent_calibration(model, quant_scheme, percentile):
    scheme = getattr(quant_scheme, "name", str(quant_scheme)).lower()
    method = ("percentile" if "percentile" in scheme else
              "sqnr" if "enhanced" in scheme else "minmax")
    for _, module, _ in named_native_recurrent(model):
        module.calibration_method = method
        module.reset_calibration()
        if method == "percentile":
            module.percentile_value = percentile


@contextmanager
def recurrent_export_mode(sim_model, original_model, opset):
    """Validate copied modules, register symbolics and temporarily enable export."""
    sim_modules = list(named_native_recurrent(sim_model))
    original_modules = list(named_native_recurrent(original_model))
    if [(n, op.package) for n, _, op in sim_modules] != [
        (n, op.package) for n, _, op in original_modules
    ]:
        raise RuntimeError("Native recurrent module paths changed during export")
    custom_opsets = {}
    with ExitStack() as stack:
        for (name, sim_module, op), (_, module, _) in zip(sim_modules, original_modules):
            for attr in ("input_size", "hidden_size", "num_layers", "bidirectional", "batch_first", "bias"):
                if getattr(sim_module, attr, None) != getattr(module, attr, None):
                    raise RuntimeError(f"Native recurrent export mismatch: {name}.{attr}")
            sim_module.set_module_name(name)
            module.set_module_name(name)
            op.register_onnx(opset=opset)
            if op.custom_opsets:
                custom_opsets.update(op.custom_opsets())
            stack.callback(setattr, module, "export_mode", module.export_mode)
            module.export_mode = True
        yield custom_opsets


def export_recurrent_encodings(model, onnx_path, encodings_path):
    """Merge native encodings and run operator-owned ONNX normalization."""
    modules = list(named_native_recurrent(model))
    if not modules:
        return
    with open(encodings_path, encoding="utf-8") as stream:
        encodings = json.load(stream)
    registrations = {}
    excluded_layers = list(encodings.get("excluded_layers") or [])
    for name, module, op in modules:
        if module.use_quantization and not module.is_calibrated():
            raise RuntimeError(f"{op.module_type.__name__} is not calibrated: {name}")
        if module.use_quantization:
            module.export_quant_params_to_aimet_format(encodings, module_name=name)
        elif name not in excluded_layers:
            excluded_layers.append(name)
        registrations[op.package] = op
    encodings["excluded_layers"] = excluded_layers
    with open(encodings_path, "w", encoding="utf-8") as stream:
        json.dump(encodings, stream, indent=2, ensure_ascii=False)
    for op in registrations.values():
        op.normalize_onnx(onnx_path)
        if op.prune_encodings:
            op.prune_encodings(encodings_path)
    # The ONNX and encoding files must describe the same registered modules.
    import onnx
    from collections import Counter
    graph = onnx.load(onnx_path)
    counts = Counter((node.op_type, node.name) for node in graph.graph.node)
    for name, module, op in modules:
        if module.is_calibrated() and counts[(op.onnx_type, name)] != 1:
            raise RuntimeError(f"Expected one ONNX {op.onnx_type} for {name!r}")
