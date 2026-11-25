"""Run one leg of the CPU/CUDA comparison and capture its stage outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pkgutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "examples/config/mrnn_quantsim_config_custom_mixed_precision_v2.json"
BITWIDTH_CONFIG_FILE = ROOT / "examples/config/quick_start_full_quant.json"
FILENAME_PREFIX = "att_mh_rnn_kws_ptq"
QUANT_SCHEME = "percentile"
PERCENTILE_VALUE = 99.99


def evaluate_torch(model, feeds, device):
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for feed in feeds:
            value = model(torch.from_numpy(feed["input"]).to(device)).detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise RuntimeError("Nonfinite KWS output")
            outputs.append([value])
    return outputs


def make_torch_sim(model, sample, device, config_file, bitwidth_config):
    import torch
    from aimet_torch.utils_rx import apply_mixed_precision_bitwidth
    from aimet_torch.v2 import quantsim

    from examples.common import kws_model

    dummy = torch.from_numpy(sample["input"]).to(device)
    sim = quantsim.QuantizationSimModel(
        model, dummy_input=dummy, quant_scheme=QUANT_SCHEME,
        config_file=str(config_file), default_output_bw=8, default_param_bw=8,
    )
    sim.set_percentile_value(PERCENTILE_VALUE)
    apply_mixed_precision_bitwidth(sim.model, config_file=str(bitwidth_config), verbose=False)
    kws_model.configure_attention_quantizers(sim.model)
    sim.model.eval()
    return sim


def calibrate_torch(model, feeds, device, config_file, bitwidth_config):
    import torch
    import aimet_torch.v2 as aimet

    sim = make_torch_sim(model, feeds[0], device, config_file, bitwidth_config)
    with torch.no_grad(), aimet.nn.compute_encodings(sim.model):
        for feed in feeds:
            sim.model(torch.from_numpy(feed["input"]).to(device))
    return sim


def load_kws_inputs(args, device):
    from examples.common import kws_data, kws_model

    kws_data.set_seed(kws_data.SEED)
    model = kws_model.make_model(args.rnn_type, device).eval()
    frontend = kws_data.make_speech_features(device).eval()
    kws_model.load_checkpoint(model, args.fp_checkpoint, args.rnn_type, device, speech_features=frontend)
    calibration, _ = kws_data.load_ptq_feeds(args.calib_dir, args.calib_limit)
    validation, labels = kws_data.load_ptq_feeds(args.validation_dir, args.validation_limit)
    if not calibration or not validation:
        raise ValueError("Calibration and validation feeds must both be nonempty")
    return (model, frontend, kws_data.extract_mfcc_feeds(calibration, frontend, device),
            kws_data.extract_mfcc_feeds(validation, frontend, device), labels)


def prepare_mfcc(args):
    from examples.common import kws_data

    _, _, calibration, validation, labels = load_kws_inputs(args, "cpu")
    for folder, feeds in (("calib", calibration), ("val", validation)):
        directory = args.output_dir / folder
        directory.mkdir(parents=True)
        for index, feed in enumerate(feeds):
            np.savez(directory / f"{index:05d}.npz", **feed)
    if labels is not None:
        manifest = {
            "labels": list(kws_data.LABELS),
            "batch_size": 1,
            "augmentation": False,
            "val_files": [f"{index:05d}.npz" for index in range(len(validation))],
            "val_labels": labels,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_torch(args):
    import torch
    from aimet_common.run_context import execution
    from aimet_torch.rx_export.export_onnx_json import export_onnx_json
    from aimet_torch.staged_quantization_utils import load_quantizer_encodings
    from aimet_torch.utils_rx import align_bias_scale, apply_power_of_2_workflow

    from examples.common import kws_data, kws_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA Torch verification leg requires an active CUDA device")
    device = torch.device(args.device)
    model, frontend, calibration, validation, _ = load_kws_inputs(args, device)
    stages = {"fp32": evaluate_torch(model, validation, device)}
    with execution(args.output_dir, "torch_ptq_validation") as run:
        run.note_inputs(checkpoint=args.fp_checkpoint, calibration=args.calib_dir,
                        validation=args.validation_dir)
        prepared = kws_model.prepare_model(model.cpu()).to(device).eval()
        sim = calibrate_torch(prepared, calibration, device, CONFIG_FILE, BITWIDTH_CONFIG_FILE)
        align_bias_scale(sim.model, bias_bitwidth=32, verbose=False)
        stages["ptq"] = evaluate_torch(sim.model, validation, device)
        apply_power_of_2_workflow(sim.model, align_bias_scale=False, verbose=False)
        align_bias_scale(sim.model, bias_bitwidth=32, verbose=False)
        stages["po2"] = evaluate_torch(sim.model, validation, device)
        checkpoint = run.checkpoints / f"{FILENAME_PREFIX}.pth"
        kws_model.save_checkpoint(sim.get_original_model(sim.model, qdq_weights=False),
                                  checkpoint, args.rnn_type, speech_features=frontend)
        sim.model.cpu().eval()
        onnx_path, encodings_path = export_onnx_json(
            sim, export_dir=str(run.exports), filename_prefix=FILENAME_PREFIX,
            dummy_input_shape=(1, *kws_data.FEATURE_SHAPE), opset=18,
        )
        fresh = kws_model.prepare_model(kws_model.make_model(args.rnn_type, device).eval())
        kws_model.load_checkpoint(fresh, checkpoint, args.rnn_type, device, speech_features=frontend)
        restored = make_torch_sim(fresh, calibration[0], device, CONFIG_FILE, BITWIDTH_CONFIG_FILE)
        load_quantizer_encodings(restored.model, load_path=str(encodings_path), skip_if_not_found=False,
                                 allow_overwrite=False, verbose=False)
        np.testing.assert_allclose(evaluate_torch(restored.model, validation, device), stages["po2"],
                                   atol=1e-5, rtol=1e-4)
    return onnx_path, encodings_path, stages


def run_onnx(args):
    from aimet_onnx.rx_ptq import run_onnx_ptq

    providers = ["CPUExecutionProvider"]
    if args.device == "cuda":
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("The CUDA ONNX verification leg requires CUDAExecutionProvider")
        providers.insert(0, "CUDAExecutionProvider")
    stages = {}
    artifacts = run_onnx_ptq(
        model_path=args.model, calibration_dir=args.calib_dir, validation_dir=args.validation_dir,
        output_dir=args.output_dir, filename_prefix=FILENAME_PREFIX, config_file=str(CONFIG_FILE),
        bitwidth_config_file=str(BITWIDTH_CONFIG_FILE), providers=providers,
        on_evaluation=stages.__setitem__,
    )
    return artifacts["onnx"], artifacts["encodings"], stages


def archive_outputs(stages, path):
    expected_stages = {"fp32", "ptq", "po2"}
    if stages.keys() != expected_stages:
        raise RuntimeError(f"Expected stages {sorted(expected_stages)}, got {sorted(stages)}")
    arrays = {}
    for stage, batches in stages.items():
        if not batches or not batches[0]:
            raise ValueError(f"No outputs for {stage} stage")
        output_count = len(batches[0])
        if any(len(batch) != output_count for batch in batches):
            raise ValueError(f"Output count changed between {stage} batches")
        for index in range(output_count):
            values = [batch[index] for batch in batches]
            first = values[0]
            for value in values[1:]:
                if value.ndim != first.ndim or value.shape[1:] != first.shape[1:]:
                    raise ValueError(f"Output {index} shape changed between {stage} batches")
                if value.dtype != first.dtype:
                    raise ValueError(f"Output {index} dtype changed between {stage} batches")
            arrays[f"{stage}_output{index}"] = (
                np.stack(values) if first.ndim == 0 else np.concatenate(values)
            )
    np.savez(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    input_options = argparse.ArgumentParser(add_help=False)
    input_options.add_argument("--calib-dir", type=Path, required=True)
    input_options.add_argument("--validation-dir", type=Path, required=True)
    input_options.add_argument("--output-dir", type=Path, required=True)
    waveform_options = argparse.ArgumentParser(add_help=False)
    waveform_options.add_argument("--rnn-type", choices=("gru", "lstm"), required=True)
    waveform_options.add_argument("--fp-checkpoint", type=Path, required=True)
    waveform_options.add_argument("--calib-limit", type=int)
    waveform_options.add_argument("--validation-limit", type=int)
    commands.add_parser("mfcc", parents=[input_options, waveform_options])
    torch_parser = commands.add_parser("torch", parents=[input_options, waveform_options])
    torch_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    onnx_parser = commands.add_parser("onnx", parents=[input_options])
    onnx_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    onnx_parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    # Source checkouts can reuse native extensions from the installed package.
    import aimet_common
    aimet_common.__path__ = pkgutil.extend_path(aimet_common.__path__, aimet_common.__name__)
    if args.command == "mfcc":
        prepare_mfcc(args)
        return
    run = run_torch if args.command == "torch" else run_onnx
    onnx_path, encodings_path, stages = run(args)
    archive = args.output_dir / "outputs.npz"
    archive_outputs(stages, archive)
    artifacts = {"onnx": str(onnx_path), "encodings": str(encodings_path), "outputs": str(archive)}
    (args.output_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")


if __name__ == "__main__":
    main()
