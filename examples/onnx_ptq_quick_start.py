"""
RX-MET ONNX PTQ 演示 — 标准用法

对应 examples/quick_start.py 的 Torch QAT 入口。流程对齐，接口尽量复用：

    加载 ONNX + 真实校准样本
        → create_onnx_ptq_sim（JSON 1 + IO quantizer + percentile）
        → apply_mixed_precision_bitwidth（JSON 2，与 quick_start.py 同一份）
        → sim.compute_encodings 校准 (PTQ)
        → apply_power_of_2_workflow（RX cover_range + Sb=Sx*Sw，算法在 aimet_common）
        → export_onnx_compiler_artifacts（clean ONNX + compiler encodings）
        → 重建 clean ONNX session，对比 FP32

ONNX 没有 PyTorch 权重可训，因此没有 QAT 微调；其余步骤与 quick_start.py 对应。
main() 只编排已有模块，不复制量化规则。

运行（先进入 examples/）：

    python onnx_ptq_quick_start.py

跑 MobileNetV2 示例。先用 `prepare_onnx_ptq_data.py` 从 ImageNette
parquet 生成 calib/val npy。模型和校准目录可用环境变量覆盖。
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aimet_onnx.rx_ptq import (
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    create_onnx_ptq_sim,
    disable_onnx_bias_quantizers,
    evaluate_onnx_outputs,
    export_onnx_compiler_artifacts,
    force_model_io_quantizers,
    load_onnx_ptq_inputs,
    run_onnx_feed,
    write_onnx_ptq_metadata,
)


# ============================================================================
# 全局配置（复制后改这里即可，无 CLI）
# ============================================================================
EXAMPLE = os.environ.get("RX_MET_ONNX_PTQ_EXAMPLE", "mobilenetv2")

_DATA = Path(os.environ.get("RX_MET_ONNX_PTQ_ROOT", "/datasets"))
EXAMPLES = {
    "mobilenetv2": {
        "model": _DATA / "mobilenetv2" / "mobilenetv2-12.onnx",
        "calib_dir": _DATA / "mobilenetv2" / "calib",
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

# 与 examples/quick_start.py 同一对 JSON：QuantSim 基础配置 + 混合精度。
CONFIG_FILE = _HERE / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = _HERE / "config" / "quick_start_full_quant.json"
OUTPUT_DIR = _HERE / "output" / "onnx_ptq_quick_start"
USE_CPU = True
CALIB_LIMIT = None


# ============================================================================
# 辅助：provider / 阶段计时 / 结果打印
# ============================================================================
def choose_providers() -> tuple[str, ...]:
    if USE_CPU:
        return ("CPUExecutionProvider",)
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


@contextlib.contextmanager
def stage(name: str, timings: dict | None = None):
    """与 quick_start.py 相同的阶段计时，不折叠业务调用。"""
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
# 主流程：与 quick_start.py 对齐的步骤编号
# ============================================================================
def main() -> None:
    if EXAMPLE not in EXAMPLES:
        raise KeyError(f"未知 EXAMPLE={EXAMPLE}，当前支持: {sorted(EXAMPLES)}")

    spec = EXAMPLES[EXAMPLE]
    model_path = Path(os.environ.get("RX_MET_ONNX_PTQ_MODEL", spec["model"]))
    calib_dir = Path(os.environ.get("RX_MET_ONNX_PTQ_CALIB", spec["calib_dir"]))
    filename_prefix = spec["prefix"]
    providers = choose_providers()
    timings: dict = {}

    print("=" * 70)
    print("RX-MET ONNX PTQ 演示 — 标准用法（复用 RX 量化接口）")
    print("=" * 70)
    print(f"example:    {EXAMPLE}")
    print(f"model:      {model_path}")
    print(f"calib:      {calib_dir}")
    print(f"config:     {CONFIG_FILE}")
    print(f"bitwidth:   {BITWIDTH_CONFIG_FILE}")
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
        if not CONFIG_FILE.is_file():
            raise FileNotFoundError(f"量化配置不存在: {CONFIG_FILE}")
        original_model, input_shapes, feeds = load_onnx_ptq_inputs(
            model_path, calib_dir, CALIB_LIMIT
        )
        print(f"输入 shape: {input_shapes}")
        print(f"校准样本:   {len(feeds)}")

    # ------------------------------------------------------------------
    # 步骤 2: 浮点评估
    #   ONNX 已是训练好的浮点图，这里只跑 FP32 参考输出，供后续对比。
    # ------------------------------------------------------------------
    with stage("步骤 2: 浮点评估", timings):
        fp32_outputs = run_onnx_feed(model_path, feeds[0], providers)
        print(f"FP32 输出数量: {len(fp32_outputs)}")
        for index, value in enumerate(fp32_outputs):
            print(f"  output[{index}] shape={list(value.shape)}")

    # ------------------------------------------------------------------
    # 步骤 3: 创建 QuantizationSimModel
    #   对应 prepare_model + QuantizationSimModel + set_percentile_value
    #   + apply_mixed_precision_bitwidth。两份 JSON 与 quick_start.py 相同。
    # ------------------------------------------------------------------
    with stage("步骤 3: 创建 sim", timings):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not BITWIDTH_CONFIG_FILE.is_file():
            raise FileNotFoundError(f"混合精度配置不存在: {BITWIDTH_CONFIG_FILE}")
        sim = create_onnx_ptq_sim(
            original_model,
            feeds[0],
            config_file=str(CONFIG_FILE),
            param_type=PARAM_TYPE,
            activation_type=ACTIVATION_TYPE,
            quant_scheme=QUANT_SCHEME,
            percentile_value=PERCENTILE_VALUE,
            providers=providers,
            work_dir=OUTPUT_DIR / "_aimet_tmp",
        )
        apply_mixed_precision_bitwidth(
            sim, config_file=str(BITWIDTH_CONFIG_FILE), verbose=True
        )
        force_model_io_quantizers(sim, original_model)
        disabled_biases = disable_onnx_bias_quantizers(sim)
        print(f"quantizer 数量: {len(sim.qc_quantize_op_dict)}")
        print(
            f"校准前关闭 {len(disabled_biases)} 个 bias quantizer；"
            "Sb=Sx*Sw 在步骤 5 写入"
        )

    # ------------------------------------------------------------------
    # 步骤 4: 校准 (PTQ)
    #   对应 aimet.nn.compute_encodings。
    #   bias 不参与统计校准：近零 per-channel scale=0 会向后续图注入 Inf。
    # ------------------------------------------------------------------
    with stage("步骤 4: 校准 (PTQ)", timings):
        sim.compute_encodings(feeds)
        ptq_report = evaluate_onnx_outputs(
            sim, feeds[0], fp32_outputs, providers
        )
        print_compare_report("PTQ vs FP32", ptq_report)

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
        po2_eval = evaluate_onnx_outputs(sim, feeds[0], fp32_outputs, providers)
        print_compare_report("Po2 vs FP32", po2_eval)

    # ------------------------------------------------------------------
    # 步骤 6: QAT 微调
    #   直量化 ONNX 没有可训练的 PyTorch 权重，这一步跳过。
    #   若后续要微调，应走 quick_start.py 的 Torch QAT 路径。
    # ------------------------------------------------------------------
    with stage("步骤 6: QAT 微调", timings):
        print("ONNX 直量化路径没有 QAT，跳过 freeze_quantizer_parameters / finetune")

    # ------------------------------------------------------------------
    # 步骤 7: 保存量化产物
    #   对应 export_onnx_json：clean ONNX + compiler encodings + RX-MET raw。
    # ------------------------------------------------------------------
    with stage("步骤 7: 保存量化产物", timings):
        artifacts = export_onnx_compiler_artifacts(
            sim, original_model, OUTPUT_DIR, filename_prefix
        )
        print(f"ONNX:              {artifacts['onnx']}")
        print(f"compiler encodings:{artifacts['encodings']}")
        print(f"RX-MET encodings:   {artifacts['rxmet_encodings']}")
        print(
            "coverage: "
            f"{artifacts['coverage']['mapped_quantizers']}/"
            f"{artifacts['coverage']['enabled_quantizers']}"
        )

    # ------------------------------------------------------------------
    # 步骤 8: 重新加载并验证
    #   对应重建 sim + load encodings。这里加载 clean ONNX，确认图契约未变。
    # ------------------------------------------------------------------
    with stage("步骤 8: 重新加载并验证", timings):
        reload_report = evaluate_onnx_outputs(
            artifacts["onnx"], feeds[0], fp32_outputs, providers
        )
        print_compare_report("clean ONNX vs FP32", reload_report)
        if any(item["rmse"] > 0 for item in reload_report):
            print("⚠️  clean ONNX 与 FP32 不一致，请检查 export 是否改图")
        else:
            print("✅ clean ONNX 与 FP32 输出一致，加载验证通过")

    metadata_path = write_onnx_ptq_metadata(
        OUTPUT_DIR,
        filename_prefix,
        {
            "example": EXAMPLE,
            "model": str(model_path),
            "input_shapes": input_shapes,
            "calibration_dir": str(calib_dir.resolve()),
            "calibration_count": len(feeds),
            "quant_scheme": QUANT_SCHEME,
            "percentile_value": PERCENTILE_VALUE,
            "po2_method": PO2_METHOD,
            "align_bias_scale": ALIGN_BIAS_SCALE,
            "bias_bitwidth": BIAS_BITWIDTH,
            "coverage": artifacts["coverage"],
            "ptq_vs_fp32": ptq_report,
            "po2_vs_fp32": po2_eval,
            "bias_align": po2_report["bias_align"],
            "all_power_of_2": po2_report.get("all_power_of_2"),
            "clean_onnx_vs_fp32": reload_report,
            "onnx": str(artifacts["onnx"]),
            "compiler_encodings": str(artifacts["encodings"]),
            "rxmet_encodings": str(artifacts["rxmet_encodings"]),
        },
    )
    print_stage_timings(timings)
    print(f"\n量化流程完成。metadata: {metadata_path}\n")


if __name__ == "__main__":
    main()
