#!/usr/bin/env python3
"""ONNX input PTQ -> clean ONNX + compiler-native encodings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "examples" / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
)
DEFAULT_BITWIDTH_CONFIG = (
    ROOT / "examples" / "config" / "quick_start_full_quant.json"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aimet_onnx.rx_ptq import run_onnx_ptq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接读取 ONNX，生成 compiler-compatible PTQ artifacts"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--bitwidth-config",
        type=Path,
        default=DEFAULT_BITWIDTH_CONFIG,
        help="与 quick_start.py 相同的混合精度 JSON；对不上的 Quantized* / GRU 会忽略",
    )
    parser.add_argument("--param-type", default="int8")
    parser.add_argument("--activation-type", default="int8")
    parser.add_argument(
        "--quant-scheme",
        default="percentile",
        choices=("min_max", "tf", "tf_enhanced", "percentile"),
    )
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument(
        "--po2-method",
        default="cover_range",
        choices=("cover_range", "round"),
    )
    parser.add_argument("--po2-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--no-align-bias-scale",
        action="store_true",
        help="跳过 Sb=Sx*Sw，仅保留独立校准后的 bias encodings",
    )
    parser.add_argument(
        "--bias-bitwidth",
        type=int,
        default=32,
        help="对齐后的 Conv/Gemm/MatMul bias 位宽，默认 32，与 RX mixed-precision 一致",
    )
    parser.add_argument("--calib-limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"模型不存在: {args.model}")
    if not args.calib_dir.is_dir():
        raise FileNotFoundError(f"校准目录不存在: {args.calib_dir}")
    if not args.config.is_file():
        raise FileNotFoundError(f"量化配置不存在: {args.config}")
    if not args.bitwidth_config.is_file():
        raise FileNotFoundError(f"混合精度配置不存在: {args.bitwidth_config}")

    if args.cpu:
        providers = ("CPUExecutionProvider",)
    else:
        try:
            import onnxruntime as ort

            providers = (
                ("CUDAExecutionProvider", "CPUExecutionProvider")
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ("CPUExecutionProvider",)
            )
        except ImportError:
            providers = ("CPUExecutionProvider",)

    artifacts = run_onnx_ptq(
        model_path=args.model,
        calibration_dir=args.calib_dir,
        output_dir=args.output_dir,
        filename_prefix=args.prefix,
        config_file=str(args.config),
        bitwidth_config_file=str(args.bitwidth_config),
        param_type=args.param_type,
        activation_type=args.activation_type,
        quant_scheme=args.quant_scheme,
        percentile_value=args.percentile,
        po2_method=args.po2_method,
        po2_tolerance=args.po2_tolerance,
        align_bias_scale=not args.no_align_bias_scale,
        bias_bitwidth=args.bias_bitwidth,
        providers=providers,
        calibration_limit=args.calib_limit,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
