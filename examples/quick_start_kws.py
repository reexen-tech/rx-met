"""
PyTorch 量化脚本参考实现：用 RX-MET 量化 kws_streaming 的 att_mh_rnn 关键词分类器。

编写自己的脚本时，先读 _run_kws()。它从浮点训练开始，创建 QuantizationSimModel，
设置位宽，在 compute_encodings() 中校准，再对齐 INT32 bias（Sb = Sx * Sw）。
示例随后演示按部署要求选用的 Power-of-2 转换和 QAT 微调，保存主权重、导出
ONNX 和配套编码 JSON，并重建 QuantSim 比较同一输入的输出。已有浮点检查点、
只做 PTQ 时，读 _run_kws_ptq()。源码仓库中的 docs/PyTorch_quantization.md 提供自有模型的接入模板。

接入自己的模型时，对照以下实现替换项目代码：
    * 模型与权重：将 _new_model() 和检查点读写替换为自己的浮点模型与权重格式。
      examples/common/kws_model.py 的 prepare_model() 和 configure_attention_quantizers() 为 KWS 注意力层
      保留显式运算并共享编码；按自己的图结构设置。
    * 数据与输入：build_dataloaders() 提供训练、校准和评估数据。FeatureLoader 用
      外置的浮点 SpeechFeatures 将波形 [B, 16000] 转为分类器输入 [B, 49, 20] MFCC。
      为自己的模型提供实际推理前处理后的代表性样本，dummy_input、校准前向与
      ONNX 导出应采用相同的输入格式。此例校准取自关闭随机增强的训练集。
    * 训练与验证：qat_finetune() 在 sim.model 上计算分类损失，evaluate() 比较分类
      准确率。按任务替换训练循环、损失和指标，再比较各量化阶段的输出与精度。

本例按 google-research/kws_streaming/models/att_mh_rnn.py 搭建卷积、两层双向
QuantGRU 或 QuantLSTM（每方向 128 单元）、四头注意力和分类层。Speech Commands
v0.02 使用 12 类（10 个关键词、_silence_、_unknown_），按文件名 hash 划分样本，
验证集和测试集各占 10%。训练增强包括时间偏移、背景噪声和时间拉伸。

复现 KWS 示例时，在宿主机将 Speech Commands v0.02 解压到挂载的数据目录，使其在容器内
对应 /datasets/speech_commands_v0.02/。在 RX-MET CUDA 容器的工作目录运行：

    cd /workspace
    python3 examples/quick_start_kws.py --rnn-type gru --data-dir /datasets/speech_commands_v0.02

将 --rnn-type 改为 lstm 即可运行 LSTM 示例。两种模式从 examples/config/quick_start_full_quant.json
分别读取 GRU_config / LSTM_config，默认浮点训练与 QAT 各 1 epoch。输出写入当前目录下的
output/quick_start_kws/gru/ 或 lstm/。运行条件与产物说明见 examples/README.md。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SRC = _ROOT / "src"
for path in (_ROOT, _SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aimet_torch.v2 as aimet
from aimet_torch.rx_export.export_onnx_json import export_onnx_json
from aimet_torch.staged_quantization_utils import load_quantizer_encodings
from aimet_torch.utils_rx import (
    align_bias_scale,
    apply_mixed_precision_bitwidth,
    apply_power_of_2_workflow,
    freeze_quantizer_parameters,
    set_train_mode_freeze_bn,
)
from aimet_torch.v2 import quantsim
from aimet_common.run_context import execution
from examples.common.kws_data import (
    FEATURE_SHAPE,
    LABELS,
    N_FRAMES,
    N_MFCC,
    SAMPLE_RATE,
    SEED,
    SpeechCommandsKWS,
    SpeechFeatures,
    TARGET_LEN,
    WANTED_WORDS,
    extract_mfcc_feeds,
    load_ptq_feeds,
    make_speech_features,
    set_seed,
    which_set,
)
from examples.common.kws_model import (
    AttentionHead,
    AttMHRNN,
    ConvActBN,
    LinearAsConv2d,
    configure_attention_quantizers,
    load_checkpoint,
    make_model,
    prepare_model,
    save_checkpoint,
)
from quant_gru import QuantGRU
from quant_lstm import QuantLSTM


DEVICE = torch.device("cuda")
DATA_ROOT = "/datasets/speech_commands_v0.02"

# 训练：quick_start 用 1 epoch 跑通流程；完整 kws 是分阶段 step（如 10k+10k+10k）
BATCH_SIZE = 64
NUM_WORKERS = 4
FP_EPOCHS = 1
QAT_EPOCHS = 1
FP_LR = 5e-4          # kws_streaming --learning_rate 第一段
QAT_LR = 1e-4

QUANT_SCHEME = "percentile"
PERCENTILE_VALUE = 99.99
DEFAULT_BW = 8
MAX_CALIB_BATCHES = 100

CONFIG_FILE = _HERE / "config" / "mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = _HERE / "config" / "quick_start_full_quant.json"
OUTPUT_DIR = Path.cwd() / "output" / "quick_start_kws"


# ============================================================================
# 数据加载
# ============================================================================
def worker_init_fn(worker_id: int) -> None:
    seed = SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def build_dataloaders(root: str):
    train_ds = SpeechCommandsKWS(root, split="train")
    test_ds = SpeechCommandsKWS(root, split="test")
    val_ds = SpeechCommandsKWS(root, split="val")
    # 校准只使用训练集，关闭随机数据增强；测试集仅用于阶段评估。
    calib_ds = SpeechCommandsKWS(root, split="train")
    calib_ds.train_mode = False

    g_train = torch.Generator()
    g_train.manual_seed(SEED)
    g_calib = torch.Generator()
    g_calib.manual_seed(SEED)

    def _make(ds, *, shuffle, generator=None, drop_last=False):
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=collate,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

    print(
        f"kws_streaming 12 类划分  train={len(train_ds)}  val={len(val_ds)}  "
        f"test={len(test_ds)}  labels={list(LABELS)}"
    )
    return {
        "train": _make(train_ds, shuffle=True, generator=g_train, drop_last=True),
        "test": _make(test_ds, shuffle=False),
        "val": _make(val_ds, shuffle=False),
        "calib": _make(calib_ds, shuffle=True, generator=g_calib),
    }


class FeatureLoader:
    """Adapt waveform batches while keeping the frontend outside the core."""

    def __init__(self, loader, speech_features):
        self.loader = loader
        self.speech_features = speech_features

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        device = next(self.speech_features.buffers()).device
        for waveform, labels in self.loader:
            with torch.no_grad():
                features = self.speech_features(waveform.to(device)).detach()
            yield features, labels


# ============================================================================
# 通用工具
# ============================================================================
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).max(1).indices
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return correct / total if total else 0.0


@contextlib.contextmanager
def stage(name: str, timings: dict | None = None):
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


# ============================================================================
# 训练循环
# ============================================================================
def train_floating_point(model, train_loader, test_loader, device,
                         save_path, rnn_type: str,
                         epochs: int = FP_EPOCHS, lr: float = FP_LR,
                         speech_features=None) -> float:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, batch_idx = 0.0, 0
        pbar = tqdm(train_loader, desc=f"FP Epoch {epoch}/{epochs}")
        for batch_idx, (x, y) in enumerate(pbar, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
        print(f"[FP] Epoch {epoch}/{epochs} - Loss: {running_loss / max(batch_idx, 1):.4f}")
        save_checkpoint(model, save_path, rnn_type, speech_features=speech_features)

    load_checkpoint(model, save_path, rnn_type, device, speech_features=speech_features)
    return evaluate(model, test_loader, device)


def qat_finetune(sim, train_loader, device,
                 epochs: int = QAT_EPOCHS, lr: float = QAT_LR) -> None:
    optim = torch.optim.Adam(
        [p for p in sim.model.parameters() if p.requires_grad], lr=lr, eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        set_train_mode_freeze_bn(sim.model)
        running_loss, valid = 0.0, 0
        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(sim.model(x), y)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            valid += 1
            pbar.set_postfix(
                loss=f"{running_loss / valid:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        print(
            f"[QAT] Epoch {epoch}/{epochs} - Loss: {running_loss / max(valid, 1):.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        scheduler.step()


def print_accuracy_summary(fp_acc, ptq_acc, po2_acc, qat_acc, reload_acc):
    print("\n" + "=" * 70)
    print("精度汇总")
    print("=" * 70)
    print(f"  浮点模型精度:           {fp_acc * 100:.2f}%")
    print(f"  PTQ 量化精度:           {ptq_acc * 100:.2f}%")
    if po2_acc is not None:
        print(f"  Power-of-2 量化精度:    {po2_acc * 100:.2f}%")
    print(f"  QAT 微调后精度:         {qat_acc * 100:.2f}%")
    print(
        f"  重新加载后精度:         {reload_acc * 100:.2f}%  "
        f"(差距 {abs(qat_acc - reload_acc) * 100:.3f}%)"
    )
    print("=" * 70)


def print_stage_timings(timings: dict):
    print("\n" + "=" * 70)
    print("耗时汇总")
    print("=" * 70)
    total = sum(timings.values())
    for name, dur in timings.items():
        pct = (dur / total * 100) if total > 0 else 0
        if dur < 60:
            print(f"  {name:35s} {dur:8.2f} 秒 ({pct:5.1f}%)")
        else:
            print(f"  {name:35s} {dur:8.2f} 秒 ({dur / 60:6.2f} 分钟, {pct:5.1f}%)")
    print("-" * 70)
    if total < 3600:
        print(f"  {'总计':35s} {total:8.2f} 秒 ({total / 60:6.2f} 分钟)")
    else:
        print(f"  {'总计':35s} {total:8.2f} 秒 ({total / 3600:.2f} 小时)")
    print("=" * 70)


def _new_model(rnn_type: str = "gru") -> AttMHRNN:
    return make_model(rnn_type, DEVICE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="使用 Speech Commands 演示 GRU / LSTM 的浮点训练、PTQ、Po2、QAT、导出和恢复。"
    )
    parser.add_argument("--rnn_type", "--rnn-type", choices=("gru", "lstm"), default="gru",
                        help="循环算子类型（默认 gru）；其余量化步骤相同")
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_ROOT),
                        help="Speech Commands v0.02 目录（默认 /datasets/speech_commands_v0.02）")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="输出根目录；默认当前目录下的 output/quick_start_kws，自动追加 gru/ 或 lstm/")
    parser.add_argument("--quant-config", type=Path, default=BITWIDTH_CONFIG_FILE,
                        help="完整的量化配置 JSON；编辑其中的 GRU_config 或 LSTM_config")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ptq-only", action="store_true",
                        help="加载浮点检查点，执行校准、可选 Po2、导出和恢复；不训练")
    parser.add_argument("--fp-checkpoint", type=Path,
                        help="PTQ 输入的浮点检查点；完整训练时指定浮点检查点的保存路径")
    parser.add_argument("--calib-dir", type=Path, help="复用 batch-1 的 input NPY/NPZ 样本")
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--export-feeds-dir", type=Path,
                        help="保存固定的 calib/、val/ 和 manifest.json，供 CPU/GPU 和 ONNX 复用")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--calib-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args(argv)
    if args.device == "cpu" and not args.ptq_only:
        parser.error("CPU 仅支持 --ptq-only；浮点训练和 QAT 请使用 CUDA")
    if args.ptq_only and args.fp_checkpoint is None:
        parser.error("--ptq-only 需要 --fp-checkpoint")
    if args.validation_dir is not None and args.calib_dir is None:
        parser.error("--validation-dir 需要 --calib-dir")
    for name in ("samples_per_class", "calib_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    return args


def prepare_ptq_feeds(data_dir: Path, output_dir: Path, samples_per_class: int):
    """Select reproducible unaugmented samples and persist the ONNX input feeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"labels": list(LABELS), "seed": SEED, "batch_size": 1, "augmentation": False}
    for split, folder in (("train", "calib"), ("val", "val")):
        dataset = SpeechCommandsKWS(str(data_dir), split=split)
        dataset.train_mode = False
        rng = random.Random(SEED)
        selected = []
        for label in LABELS:
            indices = [index for index, (_, name) in enumerate(dataset.items) if name == label]
            if len(indices) < samples_per_class:
                raise ValueError(f"{split}/{label} 样本少于 {samples_per_class}")
            selected.extend(rng.sample(indices, samples_per_class))
        rng.shuffle(selected)
        directory = output_dir / folder
        directory.mkdir()
        files, labels, sources = [], [], []
        for number, index in enumerate(selected):
            waveform, label = dataset[index]
            filename = f"{number:05d}.npz"
            np.savez(directory / filename, input=waveform.unsqueeze(0).numpy())
            files.append(filename)
            labels.append(int(label))
            sources.append(str(dataset.items[index][0]))
        manifest.update({f"{folder}_files": files, f"{folder}_labels": labels,
                         f"{folder}_sources": sources})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir / "calib", output_dir / "val"


def _run_kws_ptq(args, run):
    from aimet_common.evaluation import classification_accuracy

    if not args.fp_checkpoint.is_file():
        raise FileNotFoundError(args.fp_checkpoint)
    run.note_inputs(checkpoint=args.fp_checkpoint)
    if args.calib_dir is None:
        calib_dir, validation_dir = prepare_ptq_feeds(
            args.data_dir, args.export_feeds_dir or run.root / "data", args.samples_per_class)
    else:
        calib_dir = args.calib_dir
        validation_dir = args.validation_dir or calib_dir.parent / "val"
    waveform_feeds, _ = load_ptq_feeds(calib_dir, args.calib_limit)
    waveform_validation, labels = load_ptq_feeds(validation_dir, args.validation_limit)
    if not waveform_feeds or not waveform_validation:
        raise ValueError("校准和验证样本均不能为空")
    run.note_inputs(calibration=calib_dir, validation=validation_dir)
    set_seed(SEED)
    model = _new_model(args.rnn_type).eval()
    frontend = make_speech_features(DEVICE)
    load_checkpoint(model, args.fp_checkpoint, args.rnn_type, DEVICE, speech_features=frontend)
    feeds = extract_mfcc_feeds(waveform_feeds, frontend, DEVICE)
    validation = extract_mfcc_feeds(waveform_validation, frontend, DEVICE)

    def outputs(module):
        module.eval()
        with torch.no_grad():
            result = [module(torch.from_numpy(feed["input"]).to(DEVICE)).detach().cpu().numpy()
                      for feed in validation]
        if not all(np.isfinite(value).all() for value in result):
            raise RuntimeError("KWS 输出含非有限值")
        return np.concatenate(result)

    stage_outputs = {"fp32": outputs(model)}
    prepared = prepare_model(model.cpu()).to(DEVICE).eval()
    dummy = torch.from_numpy(feeds[0]["input"]).to(DEVICE)
    sim_options = dict(dummy_input=dummy, quant_scheme=QUANT_SCHEME,
                       config_file=str(CONFIG_FILE), default_output_bw=DEFAULT_BW,
                       default_param_bw=DEFAULT_BW)
    sim = quantsim.QuantizationSimModel(prepared, **sim_options)
    sim.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(sim.model, config_file=str(args.quant_config), verbose=True)
    configure_attention_quantizers(sim.model)
    sim.model.eval()
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        for feed in tqdm(feeds, desc="PTQ calibration"):
            sim.model(torch.from_numpy(feed["input"]).to(DEVICE))
    align_bias_scale(sim.model, bias_bitwidth=32, verbose=True)
    stage_outputs["ptq"] = outputs(sim.model)
    po2_report = None
    with stage("Power-of-2 量化（可选）", {}):
        po2_report = apply_power_of_2_workflow(sim.model, align_bias_scale=False, verbose=True)
        align_bias_scale(sim.model, bias_bitwidth=32, verbose=True)
        stage_outputs["po2"] = outputs(sim.model)
    expected_output = outputs(sim.model)
    checkpoint = run.checkpoints / "att_mh_rnn_kws_ptq.pth"
    save_checkpoint(sim.get_original_model(sim.model, qdq_weights=False), checkpoint, args.rnn_type,
                    speech_features=frontend)
    sim.model.cpu().eval()
    onnx_path, enc_path = export_onnx_json(
        sim, export_dir=str(run.exports), filename_prefix="att_mh_rnn_kws_ptq",
        dummy_input_shape=(1, *FEATURE_SHAPE), opset=18)
    fresh = prepare_model(_new_model(args.rnn_type).eval())
    load_checkpoint(fresh, checkpoint, args.rnn_type, DEVICE, speech_features=frontend)
    restored = quantsim.QuantizationSimModel(fresh, **sim_options)
    restored.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(restored.model, config_file=str(args.quant_config), verbose=True)
    configure_attention_quantizers(restored.model)
    load_quantizer_encodings(restored.model, load_path=str(enc_path), skip_if_not_found=False,
                             allow_overwrite=False, verbose=True)
    stage_outputs["restored"] = outputs(restored.model)
    restore_delta = float(np.abs(stage_outputs["restored"] - expected_output).max())
    metadata = {
        "mode": "ptq-only", "device": str(DEVICE), "rnn_type": args.rnn_type,
        "fp_checkpoint": str(args.fp_checkpoint.resolve()),
        "fp_checkpoint_sha256": hashlib.sha256(args.fp_checkpoint.read_bytes()).hexdigest(),
        "calibration_dir": str(calib_dir.resolve()), "calibration_count": len(feeds),
        "validation_dir": str(validation_dir.resolve()), "validation_count": len(validation),
        "calibration_sha256": hashlib.sha256(b"".join(feed["input"].tobytes() for feed in waveform_feeds)).hexdigest(),
        "validation_sha256": hashlib.sha256(b"".join(feed["input"].tobytes() for feed in waveform_validation)).hexdigest(),
        "restore_max_abs": restore_delta, "restore_atol": 1e-5, "restore_rtol": 1e-4,
        "onnx": str(onnx_path), "encodings": str(enc_path),
    }
    if po2_report is not None:
        metadata["all_power_of_2"] = po2_report["all_power_of_2"]
    if labels is not None:
        metadata["accuracy"] = {name: classification_accuracy([[array]], labels)
                                for name, array in stage_outputs.items()}
    (run.exports / "att_mh_rnn_kws_ptq.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.testing.assert_allclose(stage_outputs["restored"], expected_output, atol=1e-5, rtol=1e-4)
    print(f"PTQ-only 完成；恢复误差: {restore_delta:.6g}")
    print("最终产物请配套使用 .onnx 模型和 .json 量化参数：")
    print(f"ONNX 模型 (.onnx): {onnx_path}")
    print(f"量化参数 (.json):  {enc_path}")


# ============================================================================
# 主流程：标准 RX-MET 量化 + 加载验证
# ============================================================================
def main(argv=None):
    args = parse_args(argv)
    with execution(args.output_dir / args.rnn_type, f'kws_{args.rnn_type}') as run:
        run.note_inputs(dataset=args.data_dir)
        _run_kws(args, run)


def _run_kws(args, run):
    global DEVICE
    DEVICE = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("此示例需要 CUDA GPU，请在 rx-met GPU 镜像中使用 --gpus 启动容器")
    if args.ptq_only:
        return _run_kws_ptq(args, run)
    output_dir = run.exports
    fp_model_path = args.fp_checkpoint or run.checkpoints / "model_fp_kws.pth"
    if not args.quant_config.is_file():
        raise FileNotFoundError(f"量化配置不存在: {args.quant_config}")
    set_seed(SEED)
    timings: dict = {}

    print("=" * 70)
    operator_name = "QuantGRU" if args.rnn_type == "gru" else "QuantLSTM"
    print(f"RX-MET 量化演示 — att_mh_rnn + 双向 {operator_name} + Speech Commands 12 类")
    print("=" * 70)
    print(f"使用设备: {DEVICE}")
    print(f"数据目录: {args.data_dir}")
    print(f"量化配置: {args.quant_config}（循环层查看 {args.rnn_type.upper()}_config）")
    print(f"输出目录: {output_dir}")
    print("默认浮点训练和 QAT 各 1 epoch，用于学习完整流程；精度需充分训练后评估。")
    print(f"特征: 外置完整 MFCC，分类网络输入 {FEATURE_SHAPE}")

    with stage("步骤 1: 加载数据和模型", timings):
        frontend = make_speech_features(DEVICE)
        loaders = {name: FeatureLoader(loader, frontend)
                   for name, loader in build_dataloaders(str(args.data_dir)).items()}
        model = _new_model(args.rnn_type)
        n_params = sum(p.numel() for p in model.parameters())
        n_feat_buf = sum(b.numel() for b in frontend.buffers())
        n_rnn = sum(1 for m in model.modules() if isinstance(m, (QuantGRU, QuantLSTM)))
        print(
            f"att_mh_rnn 参数量: {n_params / 1e3:.1f}K  {operator_name} 层: {n_rnn}（均为 bidirectional）"
        )
        print(f"外置 SpeechFeatures 常量 buffer: {n_feat_buf / 1e3:.1f}K（不进入量化或 ONNX）")

    with stage("步骤 2: 浮点训练 + 评估", timings):
        fp_accuracy = train_floating_point(
            model, loaders["train"], loaders["test"], DEVICE, fp_model_path, args.rnn_type,
            speech_features=frontend,
        )
        print(f"浮点精度: {fp_accuracy * 100:.2f}%")

    with stage("步骤 3: prepare_model + 创建 sim", timings):
        prepared_model = prepare_model(model)
        sample_input, _ = next(iter(loaders["train"]))
        dummy_input = sample_input.to(DEVICE)

        sim = quantsim.QuantizationSimModel(
            prepared_model,
            dummy_input=dummy_input,
            quant_scheme=QUANT_SCHEME,
            config_file=str(CONFIG_FILE),
            default_output_bw=DEFAULT_BW,
            default_param_bw=DEFAULT_BW,
        )
        sim.set_percentile_value(PERCENTILE_VALUE)
        apply_mixed_precision_bitwidth(
            sim.model, config_file=str(args.quant_config), verbose=True,
        )
        configure_attention_quantizers(sim.model)

    with stage("步骤 4: 校准 (PTQ)", timings):
        sim.model.to(DEVICE).eval()
        with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
            for idx, (x, _) in enumerate(tqdm(loaders["calib"], desc="校准", total=MAX_CALIB_BATCHES)):
                if idx >= MAX_CALIB_BATCHES:
                    break
                sim.model(x.to(DEVICE))

        align_bias_scale(sim.model, bias_bitwidth=32, verbose=True)
        ptq_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"PTQ 量化精度: {ptq_accuracy * 100:.2f}%")

    po2_accuracy = None
    with stage("步骤 5: Power-of-2 量化（可选）", timings):
        print("按目标后端要求选用 Power-of-2；本示例执行转换演示。")
        apply_power_of_2_workflow(
            sim.model,
            align_bias_scale=False,
            verbose=True,
        )
        align_bias_scale(sim.model, bias_bitwidth=32, verbose=True)
        po2_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"Power-of-2 量化精度: {po2_accuracy * 100:.2f}%")

    with stage("步骤 6: QAT 微调", timings):
        freeze_quantizer_parameters(sim.model, verbose=True, freeze_bn_affine=True)
        qat_finetune(sim, loaders["train"], DEVICE)
        qat_accuracy = evaluate(sim.model, loaders["test"], DEVICE)
        print(f"QAT 微调后精度: {qat_accuracy * 100:.2f}%")

    with stage("步骤 7: 保存量化产物", timings):
        output_dir.mkdir(parents=True, exist_ok=True)
        qat_state_pth = run.checkpoints / "att_mh_rnn_kws_qat.pth"
        # 主权重与量化参数分开保存；避免检查点依赖 QuantSim 的内部 Python 状态。
        original_model = sim.get_original_model(sim.model, qdq_weights=False)
        save_checkpoint(original_model, qat_state_pth, args.rnn_type, speech_features=frontend)
        # 在导出前保留测试样本的输出，恢复后比较数值而不只比较准确率。
        reload_input = next(iter(loaders["test"]))[0].to(DEVICE)
        sim.model.eval()
        with torch.no_grad():
            expected_output = sim.model(reload_input).detach().clone()

        sim.model.cpu().eval()
        onnx_path, enc_path = export_onnx_json(
            sim,
            export_dir=str(output_dir),
            filename_prefix="att_mh_rnn_kws",
            dummy_input_shape=(1, *FEATURE_SHAPE),
            opset=18,
        )
        sim.model.to(DEVICE)
        print(f"训练态权重: {qat_state_pth}")
        print("最终产物请配套使用 .onnx 模型和 .json 量化参数：")
        print(f"ONNX 模型 (.onnx): {onnx_path}")
        print(f"量化参数 (.json):  {enc_path}")

    with stage("步骤 8: 重新加载并验证", timings):
        fresh_model = _new_model(args.rnn_type).eval()
        fresh_prepared = prepare_model(fresh_model)
        load_checkpoint(fresh_prepared, qat_state_pth, args.rnn_type, DEVICE, speech_features=frontend)
        fresh_sim = quantsim.QuantizationSimModel(
            fresh_prepared,
            dummy_input=dummy_input,
            quant_scheme=QUANT_SCHEME,
            config_file=str(CONFIG_FILE),
            default_output_bw=DEFAULT_BW,
            default_param_bw=DEFAULT_BW,
        )
        fresh_sim.set_percentile_value(PERCENTILE_VALUE)
        apply_mixed_precision_bitwidth(
            fresh_sim.model, config_file=str(args.quant_config), verbose=True,
        )
        configure_attention_quantizers(fresh_sim.model)

        load_quantizer_encodings(
            fresh_sim.model,
            load_path=str(enc_path),
            skip_if_not_found=False,
            allow_overwrite=False,
            verbose=True,
        )

        fresh_sim.model.eval()
        with torch.no_grad():
            restored_output = fresh_sim.model(reload_input)
        torch.testing.assert_close(restored_output, expected_output, atol=1e-5, rtol=1e-4)
        print(f"恢复输出最大绝对误差: {(restored_output - expected_output).abs().max().item():.6g}")
        reload_accuracy = evaluate(fresh_sim.model, loaders["test"], DEVICE)
        gap = abs(qat_accuracy - reload_accuracy) * 100
        print(f"QAT 保存前精度:    {qat_accuracy * 100:.2f}%")
        print(f"重新加载后精度:    {reload_accuracy * 100:.2f}%")
        if gap < 0.5:
            print(f"✅ 精度差异 {gap:.3f}%，加载验证通过")
        else:
            raise RuntimeError(f"恢复前后精度差异 {gap:.3f} 个百分点，检查点验证失败")

    print_accuracy_summary(fp_accuracy, ptq_accuracy, po2_accuracy, qat_accuracy, reload_accuracy)
    print_stage_timings(timings)
    print("\n量化流程完成（已通过加载验证）\n")


if __name__ == "__main__":
    main()
