"""对已有静态 ONNX 模型做训练后量化（PTQ），导出配套的 ONNX 和编译器 JSON。

将 PTQ 接入自己的项目时，可参照下方示例从 aimet_onnx.rx_ptq 导入各阶段接口。
用模型自己的前处理准备校准样本。每份 feed 是按 ONNX 输入名组织的 NumPy 数组字典，
包含一次前向所需的全部输入。用首份 feed 创建 QuantSim，应用混合精度配置后将
校准 feeds 交给 sim.compute_encodings。校准后对齐 bias，比较浮点和量化输出。
示例还演示了后端要求 Power-of-2 时如何转换 scale、重新对齐 bias。
export_onnx_json 导出产物，verify_onnx_ptq_artifacts 验证恢复后的量化输出。
已有 ONNX 和校准目录的项目也可直接调用 aimet_onnx.rx_ptq.run_onnx_ptq。

在 KWS 示例的工作目录中自动读取最近一次成功的 GRU ONNX 导出：

    python3 examples/quick_start_onnx_ptq.py

对指定模型和预处理样本执行量化：

    python3 examples/quick_start_onnx_ptq.py --model /models/model.onnx --calib-dir /data/calib

--model 指定实际模型，--rnn-type lstm 可自动选择 LSTM 导出。
省略 --calib-dir 时，从 /datasets/speech_commands_v0.02
计算校准和验证用的 MFCC，适用于 input: float32 [1,49,20] 的语音模型。
--calib-dir 指定已预处理的样本目录。单输入可用 NPY，多输入可用按输入名保存的 NPZ。
脚本默认查找校准目录旁的 val/，也可用 --validation-dir 指定验证集。
默认验证目录缺失时，用校准输入比较数值；提供分类验证标签时才计算 Top-1。
"""

from __future__ import annotations

import contextlib
import argparse
import hashlib
import json
import random
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
    align_onnx_bias_scale,
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    create_onnx_ptq_sim,
    disable_onnx_bias_quantizers,
    evaluate_onnx_dataset,
    export_onnx_json,
    force_model_io_quantizers,
    load_onnx_ptq_inputs,
    load_onnx_validation_inputs,
    run_onnx_feed,
    verify_onnx_ptq_artifacts,
    write_onnx_ptq_metadata,
)
from aimet_common.run_context import execution


# ============================================================================
# 全局配置（命令行参数也可以覆盖模型、样本与设备）
# ============================================================================
QUANT_SCHEME = "percentile"
PERCENTILE_VALUE = 99.99
PARAM_TYPE = "int8"
ACTIVATION_TYPE = "int8"

# Power-of-2 转换参数和普通层 bias 的量化位宽。
PO2_METHOD = "cover_range"
PO2_TOLERANCE = 0.02
BIAS_BITWIDTH = 32

# 分别控制 QuantSim 的基础量化规则和各层的混合精度。
CONFIG_FILE = _HERE / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = _HERE / "config" / "quick_start_full_quant.json"
OUTPUT_DIR = Path.cwd() / "output" / "quick_start_onnx_ptq"
DEVICE = "cuda"
CALIB_LIMIT = None


# ============================================================================
# 辅助：provider / 阶段计时 / 结果打印
# ============================================================================
def choose_providers(device=None) -> tuple[str, ...]:
    device = DEVICE if device is None else device
    if device == "cpu":
        return ("CPUExecutionProvider",)
    if device != "cuda":
        raise ValueError("device 只能是 cuda 或 cpu")
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
    """打印本阶段的耗时，并在提供 timings 时记录结果。"""
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
# 命令行参数与模型、样本的准备
# ============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对静态 ONNX 执行 CPU/CUDA PTQ、可选 Po2、导出和恢复。")
    parser.add_argument("--model", type=Path, help="默认使用最近一次成功的 KWS 导出")
    parser.add_argument("--rnn-type", choices=("gru", "lstm"), default="gru",
                        help="自动选择的 KWS ONNX 循环层类型，默认 gru")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/datasets/speech_commands_v0.02"),
                        help="未提供校准目录时，自行从 Speech Commands 计算 MFCC")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--calib-dir", type=Path, help="已预处理的 NPY/NPZ，提供后跳过音频预处理")
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=DEVICE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prefix")
    parser.add_argument("--config-file", type=Path, default=CONFIG_FILE)
    parser.add_argument("--quant-config", type=Path, default=BITWIDTH_CONFIG_FILE)
    parser.add_argument("--calib-limit", type=int, default=CALIB_LIMIT)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args(argv)
    if args.validation_dir is not None and args.calib_dir is None:
        parser.error("--validation-dir 需要 --calib-dir")
    for name in ("samples_per_class", "calib_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def resolve_model_path(args) -> Path:
    if args.model is not None:
        return args.model
    root = Path.cwd() / "output" / "quick_start_kws" / args.rnn_type
    candidates = list(root.glob("*/exports/att_mh_rnn_kws.onnx"))
    candidates += list(root.glob("exports/att_mh_rnn_kws.onnx"))
    successful = []
    for model in candidates:
        if not model.is_file():
            continue
        record_path = model.parent.parent / "run.json"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("status") != "success":
            continue
        started_at = record.get("started_at", "")
        if not isinstance(started_at, str):
            continue
        outputs = record.get("outputs")
        if isinstance(outputs, dict) and "exports/att_mh_rnn_kws.onnx" not in outputs:
            continue
        successful.append((started_at, model.stat().st_mtime_ns, model))
    if successful:
        return max(successful, key=lambda item: item[:2])[2]
    raise FileNotFoundError(
        f"未找到 {root} 中成功运行的 KWS ONNX；请用 --model 指定已有 ONNX，"
        "或先在当前工作目录运行 KWS 示例生成模型。"
    )


def prepare_kws_inputs(model_path, data_dir, output_dir, samples_per_class):
    """Prepare deterministic MFCC feeds from the supplied audio dataset."""
    import onnx
    from aimet_onnx.rx_ptq import inspect_model_inputs, inspect_model_input_dtypes

    model = onnx.load(str(model_path))
    shapes = inspect_model_inputs(model)
    dtypes = inspect_model_input_dtypes(model)
    if shapes != {"input": [1, 49, 20]} or dtypes != {"input": np.dtype("float32")}:
        raise ValueError("内置 Speech Commands 前处理只适用于 input: float32 [1,49,20]；其他模型请提供 --calib-dir")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from examples.common import kws_data
    import torch

    frontend = kws_data.make_speech_features("cpu").eval()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"labels": list(kws_data.LABELS), "seed": kws_data.SEED, "batch_size": 1,
                "augmentation": False, "input_shape": [1, *kws_data.FEATURE_SHAPE],
                "preprocessing": "kws_streaming mfcc_tf; fixed floating-point frontend"}
    for split, folder in (("train", "calib"), ("val", "val")):
        dataset = kws_data.SpeechCommandsKWS(str(data_dir), split=split)
        dataset.train_mode = False
        rng = random.Random(kws_data.SEED)
        selected = []
        for label in kws_data.LABELS:
            indices = [index for index, (_, name) in enumerate(dataset.items) if name == label]
            if len(indices) < samples_per_class:
                raise ValueError(f"{split}/{label} 样本少于 {samples_per_class}")
            selected.extend(rng.sample(indices, samples_per_class))
        rng.shuffle(selected)
        directory = output_dir / folder
        directory.mkdir()
        files, labels, sources = [], [], []
        for start in range(0, len(selected), 32):
            indices = selected[start:start + 32]
            samples = [dataset[index] for index in indices]
            with torch.no_grad():
                features = frontend(torch.stack([waveform for waveform, _ in samples])).numpy()
            for offset, (feature, (_, label), index) in enumerate(zip(features, samples, indices)):
                filename = f"{start + offset:05d}.npz"
                np.savez(directory / filename, input=feature[None].copy())
                files.append(filename)
                labels.append(int(label))
                sources.append(str(dataset.items[index][0]))
        manifest.update({f"{folder}_files": files, f"{folder}_labels": labels,
                         f"{folder}_sources": sources, f"{folder}_split": split})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir / "calib", output_dir / "val"


def main(argv=None) -> None:
    args = parse_args(argv)
    with execution(args.output_dir, 'onnx_ptq') as run:
        _run_onnx_ptq(run, args)


def _run_onnx_ptq(run, args) -> None:
    model_path = resolve_model_path(args)
    calib_dir = args.calib_dir
    filename_prefix = args.prefix or f"{model_path.stem}_ptq"
    run.note_inputs(model=model_path)
    providers = choose_providers(args.device)
    config_file = args.config_file
    bitwidth_config = args.quant_config
    timings: dict = {}

    print("=" * 70)
    print("RX-MET ONNX PTQ 演示 — 标准用法（复用 RX 量化接口）")
    print("=" * 70)
    print(f"model:      {model_path}")
    print(f"calib:      {calib_dir or args.data_dir}")
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
    #   每份校准或验证样本提供模型执行一次前向所需的全部输入。
    # ------------------------------------------------------------------
    with stage("步骤 1: 加载数据和模型", timings):
        if not config_file.is_file():
            raise FileNotFoundError(f"量化配置不存在: {config_file}")
        if calib_dir is None:
            run.note_inputs(dataset=args.data_dir)
            calib_dir, _ = prepare_kws_inputs(
                model_path, args.data_dir, run.root / "mfcc_feeds", args.samples_per_class)
        run.note_inputs(calibration=calib_dir)
        original_model, input_shapes, feeds = load_onnx_ptq_inputs(
            model_path, calib_dir, args.calib_limit
        )
        validation_dir = args.validation_dir or calib_dir.parent / "val"
        if validation_dir.is_dir() or args.validation_dir is not None:
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
    #   先按基础配置创建 QuantSim，再按混合精度配置设置各层位宽。
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
            sim, preserve_shared_percentile=True)
        print(f"quantizer 数量: {len(sim.qc_quantize_op_dict)}")
        print(
            f"延后至校准后对齐的 bias quantizer: {len(disabled_biases)}"
        )

    # ------------------------------------------------------------------
    # 步骤 4: 校准 (PTQ)
    #   所有普通层 bias 在评估前按实际输入、权重 scale 对齐；不依赖 Po2。
    # ------------------------------------------------------------------
    with stage("步骤 4: 校准 (PTQ)", timings):
        sim.compute_encodings(feeds)
        bias_report = align_onnx_bias_scale(sim, bias_bitwidth=BIAS_BITWIDTH)
        _, ptq_evaluation = evaluate_onnx_dataset(
            sim, evaluation_feeds, reference=fp32_batches, labels=validation_labels, providers=providers
        )
        ptq_report = ptq_evaluation["comparison"]
        print_compare_report("PTQ vs FP32", ptq_report)
        print_accuracy_report("PTQ 验证", ptq_evaluation)

    po2_report = None
    po2_metadata = {}
    evaluations = {"fp32": fp32_evaluation, "ptq": ptq_evaluation}

    # ------------------------------------------------------------------
    # 步骤 5: Power-of-2 量化（可选）
    #   仅在后端要求时采用该转换。跳过整块将保留步骤 4 的 PTQ 编码。
    # ------------------------------------------------------------------
    with stage("步骤 5: Power-of-2 量化（可选）", timings):
        print("按目标后端要求选用 Power-of-2；本示例执行转换演示。")
        po2_report = apply_power_of_2_workflow(
            sim,
            method=PO2_METHOD,
            tolerance=PO2_TOLERANCE,
            align_bias_scale=False,
            verbose=True,
        )
        po2_bias_report = align_onnx_bias_scale(sim, bias_bitwidth=BIAS_BITWIDTH)
        _, po2_evaluation = evaluate_onnx_dataset(
            sim, evaluation_feeds, reference=fp32_batches, labels=validation_labels, providers=providers
        )
        po2_eval = po2_evaluation["comparison"]
        print_compare_report("Po2 vs FP32", po2_eval)
        print_accuracy_report("Po2 验证", po2_evaluation)
        evaluations["po2"] = po2_evaluation
        po2_metadata = {
            "po2_method": PO2_METHOD,
            "po2_tolerance": PO2_TOLERANCE,
            "po2_vs_fp32": po2_eval,
            "po2_bias_align": po2_bias_report,
            "po2_changes": po2_report["po2_changes"],
            "all_power_of_2": po2_report["all_power_of_2"],
        }

    # ------------------------------------------------------------------
    # 步骤 6: QAT 微调
    #   ONNX PTQ 直接使用已训练的权重，此阶段仅打印提示。
    # ------------------------------------------------------------------
    with stage("步骤 6: QAT 微调", timings):
        print("ONNX 直量化路径没有 QAT，跳过 freeze_quantizer_parameters / finetune")

    # ------------------------------------------------------------------
    # 步骤 7: 保存量化产物
    #   导出 clean ONNX 和部署所需的编译器 JSON，同时保存可重载的 AIMET 原始编码。
    # ------------------------------------------------------------------
    with stage("步骤 7: 保存量化产物", timings):
        quantized_batches = [run_onnx_feed(sim, feed) for feed in evaluation_feeds]
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

    metadata_path = write_onnx_ptq_metadata(
        run.exports,
        filename_prefix,
        {
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
            "evaluation": evaluations,
            "quant_scheme": QUANT_SCHEME,
            "percentile_value": PERCENTILE_VALUE,
            "align_bias_scale": True,
            "bias_bitwidth": BIAS_BITWIDTH,
            "coverage": artifacts["coverage"],
            "onnx_shapes": artifacts["onnx_shapes"],
            "ptq_vs_fp32": ptq_report,
            "bias_align": bias_report,
            **po2_metadata,
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
