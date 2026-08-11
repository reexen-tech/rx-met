#!/usr/bin/env python3
"""Generate deterministic input/golden TSV for ADA300 LUT operators."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


OPS = ("exp", "sigmoid", "sin", "cos", "reciprocal", "rsqrt", "sqrt", "log", "silu")
SEGMENT_SEEDS = {
    16: 0xADA30016,
    31: 0xADA3001F,
    63: 0xADA3003F,
}
SUPPORTED_SEGMENTS = tuple(SEGMENT_SEEDS)
SUPPORTED_PRECISIONS = ("mixed_fp16", "fp32")


def bits_to_f32(bits: int) -> np.float32:
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def f32_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float32).view(np.uint32)[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for_segments(segments: int) -> int:
    if segments not in SUPPORTED_SEGMENTS:
        raise ValueError(f"unsupported segments: {segments}")
    return SEGMENT_SEEDS[segments]


def load_runner(abc_root: Path, segments: int):
    lut_fp = abc_root / "lut_fp"
    sys.path.insert(0, str(lut_fp))
    sys.path.insert(0, str(abc_root))
    path = lut_fp / f"test_ada300_bxc_{segments}seg.py"
    spec = importlib.util.spec_from_file_location(f"ada300_bxc_{segments}seg_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def add(values: list[np.float32], value: float) -> None:
    values.append(np.float32(value))


def add_neighbors(values: list[np.float32], value: float) -> None:
    center = np.float32(value)
    values.extend([
        np.nextafter(center, np.float32(-np.inf), dtype=np.float32),
        center,
        np.nextafter(center, np.float32(np.inf), dtype=np.float32),
    ])


def unique(values: list[np.float32]) -> np.ndarray:
    seen = set()
    result = []
    for value in values:
        bits = f32_bits(value)
        if bits not in seen:
            seen.add(bits)
            result.append(bits_to_f32(bits))
    return np.asarray(result, dtype=np.float32)


def random_values(op: str, count: int, *, seed: int) -> list[np.float32]:
    rng = np.random.default_rng(seed ^ sum(ord(ch) << (i % 8) for i, ch in enumerate(op)))
    result: list[np.float32] = []
    while len(result) < count:
        candidates = rng.integers(0, 2**32, size=max(4096, count * 2), dtype=np.uint32).view(np.float32)
        finite = candidates[np.isfinite(candidates)]
        if op == "exp":
            finite = finite[(finite >= -150.0) & (finite <= 100.0)]
        elif op in ("sigmoid", "silu"):
            finite = finite[(finite >= -150.0) & (finite <= 150.0)]
        elif op in ("sin", "cos"):
            finite = finite[np.abs(finite) <= 1.0e7]
        elif op in ("reciprocal", "rsqrt", "sqrt", "log"):
            finite = finite[finite > 0.0]
        result.extend(finite[: count - len(result)].tolist())
    return result


def table_thresholds(bundle: dict) -> list[float]:
    key = next(iter(bundle["lut_fp32"]))
    return [segment["threshold_quantized"][1] for segment in bundle["lut_fp32"][key]["segments"][:-1]]


def make_inputs(op: str, runner, bundles: dict, random_count: int, *, seed: int) -> np.ndarray:
    values: list[np.float32] = []
    ranges = {
        "exp": (-20.0, 11.0),
        "sigmoid": (-6.0, 6.0),
        "silu": (-6.0, 6.0),
        "sin": (-runner.PI, runner.PI),
        "cos": (-runner.PI, runner.PI),
        "reciprocal": (0.01, 8.0),
        "rsqrt": (0.01, 8.0),
        "sqrt": (0.0, 8.0),
        "log": (0.1, 8.0),
    }
    values.extend(np.linspace(*ranges[op], 1000, dtype=np.float32).tolist())

    if op == "exp":
        for k in range(-160, 145, 8):
            add_neighbors(values, k * runner.LN2)
        for threshold in table_thresholds(bundles["exp2"]):
            for k in (-128, -20, -1, 0, 1, 20, 127):
                add_neighbors(values, (k + threshold) * runner.LN2)
        for value in (-150.0, -104.0, -103.97208, 0.0, 1.0, 11.0, 88.0, 88.72284, 100.0):
            add_neighbors(values, value)
    elif op in ("sin", "cos"):
        for n in range(-32, 33):
            add_neighbors(values, n * runner.HALF_PI)
        for threshold in table_thresholds(bundles["sin"]):
            for quadrant in range(4):
                add_neighbors(values, quadrant * runner.HALF_PI + threshold)
        for value in (-1.0e7, -65536.0, 65536.0, 1.0e7):
            add_neighbors(values, value)
    elif op in ("reciprocal", "rsqrt", "sqrt", "log"):
        for exponent in range(-149, 128, 8):
            add_neighbors(values, math.ldexp(1.0, exponent))
        for threshold in table_thresholds(bundles[op]):
            for exponent in (-126, -20, -1, 0, 1, 20, 126):
                add_neighbors(values, math.ldexp(threshold, exponent))
    else:
        for value in (-150.0, -100.0, -20.0, -6.0, -0.0, 0.0, 6.0, 20.0, 100.0, 150.0):
            add_neighbors(values, value)

    values.extend(random_values(op, random_count, seed=seed))
    special_bits = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC12345]
    values.extend(bits_to_f32(bits) for bits in special_bits)
    return unique(values)


def eval_finite(op: str, x: np.ndarray, runner, bundles: dict, precision: str) -> np.ndarray:
    if op == "exp":
        return runner.exp_eval(bundles["exp2"], x, precision)
    if op == "sin":
        return runner.sin_like_eval(bundles["sin"], x, 0.0, precision)
    if op == "cos":
        return runner.sin_like_eval(bundles["sin"], x, runner.HALF_PI, precision)
    if op in ("reciprocal", "rsqrt", "sqrt", "log"):
        return runner.normalized_eval(bundles[op], op, x, precision)
    exp_neg = runner.exp_eval(bundles["exp2"], -x, precision)
    denominator = (np.float32(1.0) + exp_neg).astype(np.float32)
    sigmoid = runner.normalized_eval(bundles["reciprocal"], "reciprocal", denominator, precision)
    sigmoid[np.isposinf(denominator)] = np.float32(0.0)
    if op == "sigmoid":
        return sigmoid
    return (x * sigmoid).astype(np.float32)


def eval_op(op: str, inputs: np.ndarray, runner, bundles: dict, precision: str) -> np.ndarray:
    outputs = np.full(inputs.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(inputs)
    valid = finite.copy()
    if op in ("reciprocal", "rsqrt", "sqrt", "log"):
        valid &= inputs > 0.0
    if np.any(valid):
        outputs[valid] = eval_finite(op, inputs[valid], runner, bundles, precision)

    pos_inf = np.isposinf(inputs)
    neg_inf = np.isneginf(inputs)
    if op == "exp":
        outputs[pos_inf], outputs[neg_inf] = np.inf, np.float32(0.0)
    elif op == "sigmoid":
        outputs[pos_inf] = eval_finite(op, inputs[pos_inf], runner, bundles, precision)
        outputs[neg_inf] = np.float32(0.0)
    elif op == "silu":
        outputs[pos_inf], outputs[neg_inf] = np.inf, np.nan
    elif op == "reciprocal":
        outputs[pos_inf], outputs[neg_inf] = np.float32(0.0), np.float32(-0.0)
    elif op in ("rsqrt", "sqrt"):
        outputs[pos_inf] = np.float32(0.0) if op == "rsqrt" else np.inf
    elif op == "log":
        outputs[pos_inf] = np.inf

    zero_pos = inputs.view(np.uint32) == 0x00000000
    zero_neg = inputs.view(np.uint32) == 0x80000000
    if op == "reciprocal":
        outputs[zero_pos], outputs[zero_neg] = np.inf, -np.inf
    elif op == "rsqrt":
        outputs[zero_pos], outputs[zero_neg] = np.inf, np.inf
    elif op == "sqrt":
        outputs[zero_pos], outputs[zero_neg] = np.float32(0.0), np.float32(-0.0)
    elif op == "log":
        outputs[zero_pos], outputs[zero_neg] = -np.inf, -np.inf
    return outputs


def value_class(value: np.float32) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    if f32_bits(value) == 0x00000000:
        return "+zero"
    if f32_bits(value) == 0x80000000:
        return "-zero"
    return "finite"


def main() -> None:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--abc-lut-root", type=Path, default=script.parents[4] / "abc_lut")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--random-count", type=int, default=4096)
    parser.add_argument("--segments", type=int, choices=SUPPORTED_SEGMENTS, default=16)
    parser.add_argument("--precision", choices=SUPPORTED_PRECISIONS, default="mixed_fp16")
    args = parser.parse_args()

    seed = seed_for_segments(args.segments)
    runner, runner_path = load_runner(args.abc_lut_root.resolve(), args.segments)
    bundle_root = args.abc_lut_root.resolve() / f"lut_fp/output/ada300_bxc_{args.segments}segments"
    bundle_stems = {
        "exp2": "exponential_exp2",
        "sin": "sin_halfpi",
        "reciprocal": "reciprocal_normalized",
        "rsqrt": "rsqrt_normalized",
        "sqrt": "sqrt_normalized",
        "log": "log_normalized",
        "power_2": "power_2_normalized",
    }
    bundles = {
        name: runner.load_lut_bundle(stem, str(bundle_root))
        for name, stem in bundle_stems.items()
    }
    rows = []
    counts = {}
    for op in args.ops:
        inputs = make_inputs(op, runner, bundles, args.random_count, seed=seed)
        outputs = eval_op(op, inputs, runner, bundles, args.precision)
        counts[op] = int(inputs.size)
        for index, (input_value, output_value) in enumerate(zip(inputs, outputs)):
            rows.append((op, f"{index:06d}", f32_bits(input_value), f32_bits(output_value), value_class(output_value)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write(f"# reex-lut-bit-v1 segments={args.segments} precision={args.precision}\n")
        for op, case_id, input_bits, output_bits, cls in rows:
            out.write(f"{op}\t{case_id}\t{input_bits:08x}\t{output_bits:08x}\t{cls}\n")

    manifest = {
        "schema": "reex-lut-bit-v1",
        "segments": args.segments,
        "precision": args.precision,
        "seed": seed,
        "random_count_per_op": args.random_count,
        "ops": args.ops,
        "case_counts": counts,
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        "golden": {"path": str(args.output.resolve()), "sha256": sha256(args.output)},
        "sigmoid": "reciprocal(1 + exp(-x)); no clamp",
        "power_2": "excluded",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} rows={len(rows)} sha256={manifest['golden']['sha256']}")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
