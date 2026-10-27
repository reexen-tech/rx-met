"""
RX-MET ONNX PTQ 演示 — 标准用法

对应 examples/quick_start_kws.py 的 Torch QAT 入口。流程对齐，接口尽量复用：

    加载 ONNX + 真实校准样本
        → create_onnx_ptq_sim（JSON 1 + IO quantizer + percentile）
        → apply_mixed_precision_bitwidth（JSON 2，与 quick_start_kws.py 同一份）
        → sim.compute_encodings 校准 (PTQ)
        → apply_power_of_2_workflow（RX cover_range + Sb=Sx*Sw，算法在 aimet_common）
        → export_onnx_json（clean ONNX + compiler encodings）
        → 重建 QuantSim，恢复 encodings，对比保存前的量化输出

ONNX 没有 PyTorch 权重可训，因此没有 QAT 微调；其余步骤与 quick_start_kws.py 对应。
main() 只编排已有模块，不复制量化规则。

运行（先进入 examples/）：

    python quick_start_onnx_ptq.py

本入口只读取用户准备好的静态 ONNX 和 NPY/NPZ 校准数据，不执行模型专属预处理。
复现 MobileNetV2 + ImageNette 演示时，先单独运行 `prepare_onnx_ptq_data.py`，
再按其打印的命令执行本入口。模型、校准目录和输出目录可分别用
`RX_MET_ONNX_PTQ_MODEL`、`RX_MET_ONNX_PTQ_CALIB` 和
`RX_MET_ONNX_PTQ_OUTPUT_DIR` 覆盖。

默认模型: ./output/onnx_ptq_quick_start/models/mobilenetv2/mobilenetv2-12.onnx
默认校准: ./output/onnx_ptq_quick_start/datasets/imagenette2-320/calib
默认验证: 校准目录同级的 val/；RX_MET_ONNX_PTQ_VAL 可覆盖
默认输出: 当前运行目录下的 output/onnx_ptq_quick_start
"""

from __future__ import annotations

import contextlib
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aimet_onnx.rx_ptq import (
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    create_onnx_ptq_sim,
    disable_onnx_bias_quantizers,
    evaluate_onnx_dataset,
    export_onnx_json,
    force_model_io_quantizers,
    load_onnx_ptq_inputs,
    load_onnx_validation_inputs,
    verify_onnx_ptq_artifacts,
    write_onnx_ptq_metadata,
)
from aimet_common.run_context import execution


# ============================================================================
# 全局配置（命令行参数也可以覆盖模型、样本与设备）
# ============================================================================
EXAMPLE = os.environ.get("RX_MET_ONNX_PTQ_EXAMPLE", "mobilenetv2")

_PREPARED_ROOT = Path.cwd() / "output" / "onnx_ptq_quick_start"
_DATA = Path(os.environ.get("RX_MET_ONNX_PTQ_ROOT", str(_PREPARED_ROOT / "datasets")))
EXAMPLES = {
    "mobilenetv2": {
        "model": _PREPARED_ROOT / "models" / "mobilenetv2" / "mobilenetv2-12.onnx",
        "calib_dir": _DATA / "imagenette2-320" / "calib",
        "prefix": "mobilenetv2_ptq",
    },
}

QUANT_SCHEME = "percentile"
PERCENTILE_VALUE = 99.99
PARAM_TYPE = "int8"
ACTIVATION_TYPE = "int8"

# 与 Torch quick_start / train_qat 同一套 RX 契约。
PO2_METHOD = "cover_range"
PO2_TOLERANCE = 0.02
ALIGN_BIAS_SCALE = True
BIAS_BITWIDTH = 32

# 与 examples/quick_start_kws.py 同一对 JSON：QuantSim 基础配置 + 混合精度。
CONFIG_FILE = _HERE / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = _HERE / "config" / "quick_start_full_quant.json"
OUTPUT_DIR = Path(
    os.environ.get(
        "RX_MET_ONNX_PTQ_OUTPUT_DIR",
        str(_PREPARED_ROOT),
    )
)
DEVICE = os.environ.get("RX_MET_ONNX_PTQ_DEVICE", "cuda").strip().lower()
CALIB_LIMIT = None


# ============================================================================
# 辅助：provider / 阶段计时 / 结果打印
# ============================================================================
def choose_providers(device=None) -> tuple[str, ...]:
    device = DEVICE if device is None else device
    if device == "cpu":
        return ("CPUExecutionProvider",)
    if device != "cuda":
        raise ValueError("RX_MET_ONNX_PTQ_DEVICE 只能是 cuda 或 cpu")
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "当前 ONNX Runtime 不包含 CUDAExecutionProvider；"
            f"可用 providers: {available}"
        )
    return ("CUDAExecutionProvider", "CPUExecutionProvider")


def require_active_cuda_session(session, device=None) -> None:
    """拒绝 CUDA provider 初始化失败后整条 session 静默回退到 CPU。"""
    if (DEVICE if device is None else device) != "cuda":
        return
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "ONNX Runtime session 未启用 CUDAExecutionProvider；"
            f"实际 providers: {active}"
        )


@contextlib.contextmanager
def stage(name: str, timings: dict | None = None):
    """与 quick_start_kws.py 相同的阶段计时，不折叠业务调用。"""
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    t0 = time.time()
    try:
        yield
    finally:
        dur = time.time() - t0
        if timings is not None:
            timings[name] = dur
        if dur < 60:
            print(f"⏱️  耗时: {dur:.2f} 秒")
        else:
            print(f"⏱️  耗时: {dur:.2f} 秒 ({dur / 60:.2f} 分钟)")


def print_compare_report(title: str, report: list[dict]) -> None:
    print(f"{title}:")
    for item in report:
        print(
            f"  output[{item['index']}] shape={item['shape']}  "
            f"rmse={item['rmse']:.6f}  max_abs={item['max_abs']:.6f}"
        )


def print_accuracy_report(title: str, report: dict) -> None:
    if "top1_accuracy" in report:
        print(f"{title}: Top-1 {report['top1_accuracy'] * 100:.2f}% ({report['sample_count']} samples)")


def print_stage_timings(timings: dict) -> None:
    print("\n" + "=" * 70)
    print("耗时汇总")
    print("=" * 70)
    total = sum(timings.values())
    for name, dur in timings.items():
        pct = (dur / total * 100) if total > 0 else 0
        if dur < 60:
            print(f"  {name:40s} {dur:8.2f} 秒 ({pct:5.1f}%)")
        else:
            print(
                f"  {name:40s} {dur:8.2f} 秒 ({dur / 60:6.2f} 分钟, {pct:5.1f}%)"
            )
    print("-" * 70)
    print(f"  {'总计':40s} {total:8.2f} 秒")
    print("=" * 70)


# ============================================================================
# 主流程：与 quick_start_kws.py 对齐的步骤编号
# ============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对静态 ONNX 执行 CPU/CUDA PTQ、Po2、导出和恢复。")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calib-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prefix")
    parser.add_argument("--config-file", type=Path, default=CONFIG_FILE)
    parser.add_argument("--quant-config", type=Path, default=BITWIDTH_CONFIG_FILE)
    parser.add_argument("--calib-limit", type=int, default=CALIB_LIMIT)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args(argv)
    for name in ("calib_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    with execution(args.output_dir, 'onnx_ptq') as run:
        _run_onnx_ptq(run, args)


def _run_onnx_ptq(run, args=None) -> None:
    args = parse_args([]) if args is None else args
    if EXAMPLE not in EXAMPLES:
        raise KeyError(f"未知 EXAMPLE={EXAMPLE}，当前支持: {sorted(EXAMPLES)}")

    spec = EXAMPLES[EXAMPLE]
    model_path = args.model or Path(os.environ.get("RX_MET_ONNX_PTQ_MODEL", spec["model"]))
    calib_dir = args.calib_dir or Path(os.environ.get("RX_MET_ONNX_PTQ_CALIB", spec["calib_dir"]))
    filename_prefix = args.prefix or spec["prefix"]
    run.note_inputs(model=model_path, calibration=calib_dir)
    providers = choose_providers(args.device)
    config_file = args.config_file
    bitwidth_config = args.quant_config
    timings: dict = {}

    print("=" * 70)
    print("RX-MET ONNX PTQ 演示 — 标准用法（复用 RX 量化接口）")
    print("=" * 70)
    print(f"example:    {EXAMPLE}")
    print(f"model:      {model_path}")
    print(f"calib:      {calib_dir}")
    print(f"config:     {config_file}")
    print(f"bitwidth:   {bitwidth_config}")
    print(f"providers:  {providers}")
    print(
        f"scheme:     {QUANT_SCHEME}={PERCENTILE_VALUE}  "
        f"W={PARAM_TYPE} A={ACTIVATION_TYPE}  "
        f"po2={PO2_METHOD}  bias={BIAS_BITWIDTH}"
    )

    # ------------------------------------------------------------------
    # 步骤 1: 加载数据和模型
    #   对应 quick_start 的 build_dataloaders + 构建模型。
    # ------------------------------------------------------------------
    with stage("步骤 1: 加载数据和模型", timings):
        if not config_file.is_file():
            raise FileNotFoundError(f"量化配置不存在: {config_file}")
        original_model, input_shapes, feeds = load_onnx_ptq_inputs(
            model_path, calib_dir, args.calib_limit
        )
        validation_dir = args.validation_dir or Path(os.environ.get("RX_MET_ONNX_PTQ_VAL", str(calib_dir.parent / "val")))
        if validation_dir.is_dir() or args.validation_dir is not None or "RX_MET_ONNX_PTQ_VAL" in os.environ:
            evaluation_feeds, validation_labels = load_onnx_validation_inputs(original_model, validation_dir)
            evaluation_feeds = evaluation_feeds[:args.validation_limit]
            if validation_labels is not None:
                validation_labels = validation_labels[:args.validation_limit]
        else:
            evaluation_feeds, validation_labels = feeds, None
            validation_dir = None
        print(f"输入 shape: {input_shapes}")
        print(f"校准样本:   {len(feeds)}")
        print(f"评估数据:   {validation_dir or calib_dir} ({len(evaluation_feeds)} feeds)")

    # ------------------------------------------------------------------
    # 步骤 2: 浮点评估
    #   ONNX 已是训练好的浮点图，这里只跑 FP32 参考输出，供后续对比。
    # ------------------------------------------------------------------
    with stage("步骤 2: 浮点评估", timings):
        fp32_batches, fp32_evaluation = evaluate_onnx_dataset(
            model_path, evaluation_feeds, labels=validation_labels, providers=providers
        )
        fp32_outputs = fp32_batches[0]
        print(f"FP32 输出数量: {len(fp32_outputs)}")
        for index, value in enumerate(fp32_outputs):
            print(f"  output[{index}] shape={list(value.shape)}")
        print_accuracy_report("FP32 验证", fp32_evaluation)

    # ------------------------------------------------------------------
    # 步骤 3: 创建 QuantizationSimModel
    #   对应 prepare_model + QuantizationSimModel + set_percentile_value
    #   + apply_mixed_precision_bitwidth。两份 JSON 与 quick_start_kws.py 相同。
    # ------------------------------------------------------------------
    with stage("步骤 3: 创建 sim", timings):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not bitwidth_config.is_file():
            raise FileNotFoundError(f"混合精度配置不存在: {bitwidth_config}")
        sim = create_onnx_ptq_sim(
            original_model,
            feeds[0],
            config_file=str(config_file),
            param_type=PARAM_TYPE,
            activation_type=ACTIVATION_TYPE,
            quant_scheme=QUANT_SCHEME,
            percentile_value=PERCENTILE_VALUE,
            providers=providers,
            work_dir=run.work / "aimet",
        )
        require_active_cuda_session(sim.session, args.device)
        apply_mixed_precision_bitwidth(
            sim, config_file=str(bitwidth_config), verbose=True
        )
        force_model_io_quantizers(sim, original_model)
        disabled_biases = disable_onnx_bias_quantizers(
            sim, preserve_shared_percentile=True) if ALIGN_BIAS_SCALE else []
        print(f"quantizer 数量: {len(sim.qc_quantize_op_dict)}")
        print(
            f"延后至 Po2 对齐的 bias quantizer: {len(disabled_biases)}"
        )

    # ------------------------------------------------------------------
    # 步骤 4: 校准 (PTQ)
    #   对应 aimet.nn.compute_encodings。
    #   Percentile 按配置校准 bias；旧原生统计路径的 bias 延后到 Po2 对齐。
    # ------------------------------------------------------------------
    with stage("步骤 4: 校准 (PTQ)", timings):
        sim.compute_encodings(feeds)
        ptq_batches, ptq_evaluation = evaluate_onnx_dataset(
            sim, evaluation_feeds, reference=fp32_batches, labels=validation_labels, providers=providers
        )
        ptq_report = ptq_evaluation["comparison"]
        print_compare_report("PTQ vs FP32", ptq_report)
        print_accuracy_report("PTQ 验证", ptq_evaluation)

    # ------------------------------------------------------------------
    # 步骤 5: Power-of-2 量化
    #   算法是 aimet_common.power_of_2 里的 RX 原文；这里走 ONNX 入口，
    #   不经过 aimet_torch。
    # ------------------------------------------------------------------
    with stage("步骤 5: Power-of-2 量化", timings):
        po2_report = apply_power_of_2_workflow(
            sim,
            method=PO2_METHOD,
            tolerance=PO2_TOLERANCE,
            align_bias_scale=ALIGN_BIAS_SCALE,
            verbose=True,
            bias_bitwidth=BIAS_BITWIDTH,
        )
        quantized_batches, po2_evaluation = evaluate_onnx_dataset(
            sim, evaluation_feeds, reference=fp32_batches, labels=validation_labels, providers=providers
        )
        po2_eval = po2_evaluation["comparison"]
        print_compare_report("Po2 vs FP32", po2_eval)
        print_accuracy_report("Po2 验证", po2_evaluation)

    # ------------------------------------------------------------------
    # 步骤 6: QAT 微调
    #   直量化 ONNX 没有可训练的 PyTorch 权重，这一步跳过。
    #   若后续要微调，应走 quick_start_kws.py 的 Torch QAT 路径。
    # ------------------------------------------------------------------
    with stage("步骤 6: QAT 微调", timings):
        print("ONNX 直量化路径没有 QAT，跳过 freeze_quantizer_parameters / finetune")

    # ------------------------------------------------------------------
    # 步骤 7: 保存量化产物
    #   与 Torch 共用导出入口名：clean ONNX + compiler encodings + RX-MET raw。
    # ------------------------------------------------------------------
    with stage("步骤 7: 保存量化产物", timings):
        artifacts = export_onnx_json(
            sim, original_model, run.exports, filename_prefix
        )
        print("最终产物请配套使用 .onnx 模型和 .json 量化参数：")
        print(f"ONNX 模型 (.onnx): {artifacts['onnx']}")
        print(f"量化参数 (.json):  {artifacts['encodings']}")
        print(f"原始 AIMET 编码 (调试参考): {artifacts['aimet_encodings']}")
        print(
            "coverage: "
            f"{artifacts['coverage']['mapped_quantizers']}/"
            f"{artifacts['coverage']['enabled_quantizers']}"
        )

    # ------------------------------------------------------------------
    # 步骤 8: 重新加载并验证
    #   同时检查浮点图、恢复后的量化输出和 compiler encodings。
    # ------------------------------------------------------------------
    with stage("步骤 8: 重新加载并验证", timings):
        _, clean_evaluation = evaluate_onnx_dataset(
            artifacts["onnx"], evaluation_feeds, reference=fp32_batches, providers=providers
        )
        clean_report = clean_evaluation["comparison"]
        print_compare_report("clean ONNX vs FP32", clean_report)
        if any(item["max_abs"] > 0 for item in clean_report):
            raise RuntimeError("clean ONNX 与 FP32 不一致，请检查 export 是否改图")
        reload_report = verify_onnx_ptq_artifacts(
            artifacts, evaluation_feeds[0], quantized_batches[0], config_file=str(config_file),
            param_type=PARAM_TYPE, activation_type=ACTIVATION_TYPE,
            quant_scheme=QUANT_SCHEME, percentile_value=PERCENTILE_VALUE,
            providers=providers,
            validation_feeds=evaluation_feeds, expected_validation_outputs=quantized_batches,
            validation_labels=validation_labels,
        )
        print_compare_report("恢复后的量化输出 vs 保存前", reload_report)
        print_accuracy_report("恢复验证", reload_report[0])
        print("✅ ONNX 维度、量化输出和 compiler encodings 恢复验证通过")

    archived_outputs = {}
    for stage_name, batches in (("fp32", fp32_batches), ("ptq", ptq_batches), ("po2", quantized_batches)):
        count = len(batches[0])
        if any(len(batch) != count for batch in batches):
            raise ValueError(f"{stage_name}: output count changed between batches")
        for index in range(count):
            values = [np.asarray(batch[index]) for batch in batches]
            rank, nonbatch_shape = values[0].ndim, values[0].shape[1:]
            if any(value.ndim != rank or value.shape[1:] != nonbatch_shape for value in values):
                raise ValueError(f"{stage_name}: output {index} rank or nonbatch shape changed between batches")
            archived_outputs[f"{stage_name}_output{index}"] = (
                np.stack(values) if rank == 0 else np.concatenate(values, axis=0))
    np.savez(run.exports / "outputs.npz", **archived_outputs)

    metadata_path = write_onnx_ptq_metadata(
        run.exports,
        filename_prefix,
        {
            "example": EXAMPLE,
            "device": args.device,
            "model": str(model_path),
            "input_shapes": input_shapes,
            "calibration_dir": str(calib_dir.resolve()),
            "calibration_count": len(feeds),
            "calibration_sha256": hashlib.sha256(b"".join(
                feed[name].tobytes() for feed in feeds for name in sorted(feed))).hexdigest(),
            "validation_sha256": hashlib.sha256(b"".join(
                feed[name].tobytes() for feed in evaluation_feeds for name in sorted(feed))).hexdigest(),
            "evaluation_source": "validation" if validation_dir is not None else "calibration",
            "validation_dir": str(validation_dir.resolve()) if validation_dir is not None else None,
            "evaluation": {"fp32": fp32_evaluation, "ptq": ptq_evaluation, "po2": po2_evaluation},
            "quant_scheme": QUANT_SCHEME,
            "percentile_value": PERCENTILE_VALUE,
            "po2_method": PO2_METHOD,
            "align_bias_scale": ALIGN_BIAS_SCALE,
            "bias_bitwidth": BIAS_BITWIDTH,
            "coverage": artifacts["coverage"],
            "onnx_shapes": artifacts["onnx_shapes"],
            "ptq_vs_fp32": ptq_report,
            "po2_vs_fp32": po2_eval,
            "bias_align": po2_report["bias_align"],
            "all_power_of_2": po2_report.get("all_power_of_2"),
            "clean_onnx_vs_fp32": clean_report,
            "restored_quantized_vs_pre_export": reload_report,
            "onnx": str(artifacts["onnx"]),
            "compiler_encodings": str(artifacts["encodings"]),
            "aimet_encodings": str(artifacts["aimet_encodings"]),
        },
    )
    print_stage_timings(timings)
    print(f"\n量化流程完成。metadata: {metadata_path}\n")


if __name__ == "__main__":
    main()
