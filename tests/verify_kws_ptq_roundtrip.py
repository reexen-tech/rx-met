"""KWS float checkpoint -> Torch PTQ/Po2 -> ONNX PTQ/Po2, without QAT.

Requires CUDA and the updated RX-MET/native recurrent runtime. Example:
    python3 tests/verify_kws_ptq_roundtrip.py --fp-checkpoint /models/model_fp_kws.pth

Both passes consume the same saved batch-1 feeds and the same JSON snapshots.
Differences are recorded before returning 1; --report-only returns 0 instead.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pkgutil
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# Source checkouts can reuse the installed package's native extensions.
import aimet_common
aimet_common.__path__ = pkgutil.extend_path(aimet_common.__path__, aimet_common.__name__)

import numpy as np
import onnx
import torch

from examples import quick_start_kws as kws
from aimet_common.evaluation import classification_accuracy, compare_output_batches
from aimet_common.compiler_artifacts import validate_compiler_artifacts
from aimet_onnx.rx_ptq import run_onnx_ptq, create_onnx_ptq_sim
from aimet_onnx.rx_ptq import pipeline as ptq_pipeline
from aimet_onnx.rx_ptq.calibration import iter_calibration_feeds
from aimet_onnx.rx_ptq.evaluation import evaluate_onnx_dataset
from aimet_onnx.rx_ptq.mixed_precision import apply_onnx_mixed_precision_bitwidth
from aimet_onnx.quantsim import load_encodings_to_sim

ENCODING_FIELDS = ("bitwidth", "dtype", "is_symmetric", "enc_type", "scale", "n",
                   "zero_point", "qmin", "qmax", "real_min", "real_max")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_feeds(root, output, samples_per_class, seed):
    manifest = {"labels": list(kws.LABELS), "seed": seed, "batch_size": 1, "augmentation": False}
    digest = hashlib.sha256()
    for split, folder in (("train", "calib"), ("val", "val")):
        dataset = kws.SpeechCommandsKWS(str(root), split=split)
        dataset.train_mode = False
        by_label = defaultdict(list)
        for index, (_, label) in enumerate(dataset.items):
            by_label[label].append(index)
        rng = random.Random(seed)
        selected = []
        for label in kws.LABELS:
            if len(by_label[label]) < samples_per_class:
                raise ValueError(f"Not enough samples for {split}/{label}")
            selected.extend(rng.sample(by_label[label], samples_per_class))
        rng.shuffle(selected)
        directory = output / folder
        directory.mkdir(parents=True)
        files, labels, sources = [], [], []
        for number, index in enumerate(selected):
            waveform, label = dataset[index]
            array = waveform.unsqueeze(0).numpy()
            filename = f"{number:05d}.npz"
            np.savez(directory / filename, input=array)
            digest.update(folder.encode() + filename.encode() + array.tobytes())
            files.append(filename)
            labels.append(int(label))
            sources.append(str(dataset.items[index][0]))
        manifest.update({f"{folder}_files": files, f"{folder}_labels": labels,
                         f"{folder}_sources": sources})
    manifest["feed_sha256"] = digest.hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_feeds(directory):
    return list(iter_calibration_feeds(directory, {"input": [1, kws.TARGET_LEN]},
                input_dtypes={"input": np.dtype("float32")}))


def evaluate(model, feeds, labels, *, return_outputs=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for feed in feeds:
            value = model(torch.from_numpy(feed["input"]).to(kws.DEVICE)).detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise RuntimeError("Nonfinite KWS output")
            outputs.append([value])
    metrics = classification_accuracy(outputs, labels)
    return (metrics, outputs) if return_outputs else metrics


def encoding_records(value, path=()):
    if isinstance(value, dict):
        if "scale" in value:
            yield path, value
        else:
            for name, child in value.items():
                yield from encoding_records(child, path + (name,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from encoding_records(child, path + (str(index),))


def equal_field(left, right):
    if left is None or right is None:
        return left is right
    if isinstance(left, str) or isinstance(right, str):
        return str(left).lower() == str(right).lower()
    try:
        a, b = np.broadcast_arrays(np.asarray(left).reshape(-1), np.asarray(right).reshape(-1))
    except ValueError:
        return False
    return bool(np.array_equal(a, b))


def compare_encodings(left, right):
    result = {"json_equal": left == right}
    for section in ("activation_encodings", "param_encodings"):
        a, b = left[section], right[section]
        ar, br = dict(encoding_records(a)), dict(encoding_records(b))
        changes = []
        for path in sorted(ar.keys() & br.keys()):
            fields = [field for field in ENCODING_FIELDS
                      if not equal_field(ar[path].get(field), br[path].get(field))]
            if fields:
                changes.append({"path": list(path), "fields": fields,
                                "torch": ar[path], "onnx": br[path]})
        result[section] = {
            "entry_count": {"torch": len(a), "onnx": len(b)},
            "added_entries": sorted(b.keys() - a.keys()),
            "removed_entries": sorted(a.keys() - b.keys()),
            "common_records": len(ar.keys() & br.keys()),
            "changed_records": changes,
            "added_records": [list(path) for path in sorted(br.keys() - ar.keys())],
            "removed_records": [list(path) for path in sorted(ar.keys() - br.keys())],
        }
    owners = {name: kind for name, entry in left["activation_encodings"].items()
              for kind in ("GRU", "LSTM") if entry.get(f'is_{kind}')}
    result["recurrent_equal"] = {
        name: {"activation": left["activation_encodings"][name] == right["activation_encodings"].get(name),
               "kind": owners[name],
               "parameters": {key: value == right["param_encodings"].get(key)
                              for key, value in left["param_encodings"].items() if key.startswith(name + ".")}}
        for name in sorted(owners)
    }
    for kind in ("GRU", "LSTM"):
        result[f'{kind.lower()}_equal'] = {name: entry for name, entry in result['recurrent_equal'].items()
                                         if entry['kind'] == kind}
    return result


def compare_models(left_path, right_path):
    left, right = onnx.load(str(left_path)), onnx.load(str(right_path))
    a, b = ({item.name: item for item in model.graph.initializer} for model in (left, right))
    return {
        "file_equal": sha256(left_path) == sha256(right_path),
        "sha256": {"torch": sha256(left_path), "onnx": sha256(right_path)},
        "node_count": [len(left.graph.node), len(right.graph.node)],
        "nodes_equal": list(left.graph.node) == list(right.graph.node),
        "initializer_names_equal": a.keys() == b.keys(),
        "changed_weights": [name for name in sorted(a.keys() & b.keys())
                            if not np.array_equal(onnx.numpy_helper.to_array(a[name]), onnx.numpy_helper.to_array(b[name]))],
        "inputs_equal": list(left.graph.input) == list(right.graph.input),
        "outputs_equal": list(left.graph.output) == list(right.graph.output),
        "value_info_equal": {v.name: v for v in left.graph.value_info} == {v.name: v for v in right.graph.value_info},
    }


def compare_operands(model, left, right):
    from aimet_common.quantsim_config.compute_ops import is_compute_op

    def input_encoding(payload, node, index):
        slots = payload["activation_encodings"].get(node.name, {}).get("input", {})
        record = slots[index] if isinstance(slots, list) and index < len(slots) else slots.get(str(index)) if isinstance(slots, dict) else None
        if record is not None:
            return record, "activation.input"
        record = payload["param_encodings"].get(node.input[index])
        if isinstance(record, list):
            if len(record) != 1:
                raise ValueError("Expected compact compiler parameter encodings")
            record = record[0]
        return record, "param" if record is not None else None

    changes = []
    for node in model.graph.node:
        if not is_compute_op(node.op_type):
            continue
        for index, name in enumerate(node.input):
            if not name:
                continue
            a, source_a = input_encoding(left, node, index)
            b, source_b = input_encoding(right, node, index)
            if a is not None and b is not None:
                fields = [field for field in ENCODING_FIELDS if not equal_field(a.get(field), b.get(field))]
            else:
                fields = ["presence"] if (a is None) != (b is None) else []
            if fields or source_a != source_b:
                changes.append({"node": node.name, "op_type": node.op_type, "input_index": index,
                    "tensor": name, "fields": fields, "torch_location": source_a, "onnx_location": source_b,
                    "torch": a, "onnx": b})
    return changes


def write_summary(output, report):
    rows = ["# KWS PTQ Roundtrip", "", "QAT steps: 0. Both passes use the same JSON snapshots and batch-1 feeds.", "",
            "| Check | Result |", "| --- | --- |",
            f"| ONNX file identical | {report['onnx']['file_equal']} |",
            f"| Encoding JSON identical | {report['encodings']['json_equal']} |",
            f"| Torch/ONNX PTQ outputs identical | {all(item['max_abs'] == 0 for item in report['torch_vs_onnx_ptq']['comparison'])} |",
            f"| Torch/ONNX Po2 outputs identical | {report['functional_match']} |",
            f"| PTQ prediction differences | {report['torch_vs_onnx_ptq']['prediction_differences']} |",
            f"| Floating weights unchanged by PTQ | {report['weights_unchanged_by_ptq']} |",
            f"| Calibration / validation samples | {report['calibration_count']} / {report['validation_count']} |"]
    for section in ("activation_encodings", "param_encodings"):
        part = report["encodings"][section]
        rows.append(f"| {section}: Torch / ONNX entries | {part['entry_count']['torch']} / {part['entry_count']['onnx']} |")
        rows.append(f"| {section}: changed common records | {len(part['changed_records'])} |")
    for name, part in report["encodings"]["recurrent_equal"].items():
        rows.append(f"| {name}: full native encodings identical | {part['activation'] and all(part['parameters'].values())} |")
    metrics = report["evaluation"]
    rows.extend(["", "## Top-1", "", "| Stage | Torch | ONNX |", "| --- | --- | --- |",
                 f"| PTQ | {metrics['torch_ptq']['top1_accuracy']:.2%} | {metrics['onnx']['ptq']['top1_accuracy']:.2%} |",
                 f"| Po2 | {metrics['torch_po2']['top1_accuracy']:.2%} | {metrics['onnx']['po2']['top1_accuracy']:.2%} |"])
    rows.extend(["", "## Operand Encoding Differences", "",
                 "| Input | Changed fields |", "| --- | --- |"])
    for change in report["operand_differences"]:
        if change["fields"]:
            rows.append(f"| {change['node']}.input[{change['input_index']}] | "
                        f"{', '.join(change['fields'])} |")
    rows.extend(["", "See comparison.json for every changed, added and removed encoding record."])
    (output / "report.md").write_text("\n".join(rows) + "\n")


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA and the native recurrent runtime")
    for path in (args.fp_checkpoint, args.config_file, args.bitwidth_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_dir.is_dir():
        raise FileNotFoundError(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    kws.set_seed(args.seed)
    config_dir = args.output_dir / "config"
    config_dir.mkdir()
    configs = []
    fingerprints = {str(args.fp_checkpoint.resolve()): sha256(args.fp_checkpoint)}
    for name, source in (("quantsim.json", args.config_file), ("mixed_precision.json", args.bitwidth_config)):
        # One byte-for-byte snapshot is passed to both frameworks.
        target = config_dir / name
        target.write_bytes(source.read_bytes())
        fingerprints[str(source.resolve())] = sha256(source)
        configs.append(target)

    print("1. Prepare shared batch-1 calibration and validation feeds", flush=True)
    data_dir = args.output_dir / "data"
    manifest = prepare_feeds(args.data_dir, data_dir, args.samples_per_class, args.seed)
    feeds, validation = load_feeds(data_dir / "calib"), load_feeds(data_dir / "val")
    labels = manifest["val_labels"]

    print("2. Load floating checkpoint; KWS PTQ -> Po2 -> export (no QAT)", flush=True)
    model = kws._new_model(args.rnn_type).eval()
    kws.load_checkpoint(model, args.fp_checkpoint, args.rnn_type, kws.DEVICE)
    prepared = kws.model_preparer.prepare_model(model)
    before = state_fingerprint(prepared)
    sim = kws.quantsim.QuantizationSimModel(prepared,
        dummy_input=torch.from_numpy(feeds[0]["input"]).to(kws.DEVICE),
        quant_scheme=kws.QUANT_SCHEME, config_file=str(configs[0]),
        default_output_bw=kws.DEFAULT_BW, default_param_bw=kws.DEFAULT_BW)
    sim.set_percentile_value(kws.PERCENTILE_VALUE)
    kws.apply_mixed_precision_bitwidth(sim.model, config_file=str(configs[1]), verbose=False)
    sim.model.eval()
    with torch.no_grad(), kws.aimet.nn.compute_encodings(sim.model):
        for index, feed in enumerate(feeds, 1):
            sim.model(torch.from_numpy(feed["input"]).to(kws.DEVICE))
            if index % 64 == 0:
                print(f"  calibrated {index}/{len(feeds)}", flush=True)
    ptq, torch_ptq_outputs = evaluate(sim.model, validation, labels, return_outputs=True)
    kws.apply_power_of_2_workflow(sim.model, align_bias_scale=True, verbose=False)
    po2, torch_po2_outputs = evaluate(sim.model, validation, labels, return_outputs=True)
    after = state_fingerprint(sim.get_original_model(sim.model, qdq_weights=False))
    if before != after:
        raise RuntimeError("PTQ unexpectedly changed the floating weights or buffers")
    sim.model.cpu().eval()
    first_onnx, first_enc = kws.export_onnx_json(sim,
        export_dir=str(args.output_dir / "torch_ptq"), filename_prefix="att_mh_rnn_kws",
        dummy_input_shape=(1, kws.TARGET_LEN), opset=18)

    print("3. Feed the first exported ONNX into ONNX PTQ with the same JSON/data", flush=True)
    stage_outputs = []
    evaluator = ptq_pipeline.evaluate_onnx_dataset
    def capture_evaluation(source, *values, **options):
        outputs, metrics = evaluator(source, *values, **options)
        if hasattr(source, "qc_quantize_op_dict"):
            stage_outputs.append(outputs)
        return outputs, metrics
    with patch.object(ptq_pipeline, "evaluate_onnx_dataset", side_effect=capture_evaluation):
        second = run_onnx_ptq(model_path=first_onnx, calibration_dir=data_dir / "calib",
            validation_dir=data_dir / "val", output_dir=args.output_dir / "onnx_ptq",
            filename_prefix="att_mh_rnn_kws", config_file=str(configs[0]),
            bitwidth_config_file=str(configs[1]), quant_scheme=kws.QUANT_SCHEME,
            percentile_value=kws.PERCENTILE_VALUE, providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    if len(stage_outputs) < 2:
        raise RuntimeError("Missing ONNX PTQ/Po2 evaluation stages")
    ptq_comparison = compare_output_batches(torch_ptq_outputs, stage_outputs[0])
    ptq_prediction_differences = sum(
        int(np.argmax(left[0]) != np.argmax(right[0]))
        for left, right in zip(torch_ptq_outputs, stage_outputs[0]))
    np.savez_compressed(args.output_dir / "quantized_outputs.npz",
        torch_ptq=np.concatenate([batch[0] for batch in torch_ptq_outputs]),
        onnx_ptq=np.concatenate([batch[0] for batch in stage_outputs[0]]),
        torch_po2=np.concatenate([batch[0] for batch in torch_po2_outputs]),
        onnx_po2=np.concatenate([batch[0] for batch in stage_outputs[1]]), labels=labels)
    restored = create_onnx_ptq_sim(onnx.load(first_onnx), feeds[0], config_file=str(configs[0]),
        quant_scheme=kws.QUANT_SCHEME, percentile_value=kws.PERCENTILE_VALUE,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"))
    apply_onnx_mixed_precision_bitwidth(restored, str(configs[1]), verbose=False)
    load_encodings_to_sim(restored, str(second["aimet_encodings"]), strict=False,
        allow_overwrite=False, disable_missing_quantizers=True)
    restored_outputs, cross_evaluation = evaluate_onnx_dataset(restored, validation,
        reference=torch_po2_outputs, labels=labels)
    cross_evaluation["prediction_differences"] = sum(
        int(np.argmax(left[0]) != np.argmax(right[0]))
        for left, right in zip(torch_po2_outputs, restored_outputs))
    functional_match = all(item["max_abs"] == 0 for item in cross_evaluation["comparison"])
    first = json.loads(Path(first_enc).read_text())
    second_enc = json.loads(second["encodings"].read_text())
    strict_acceptance = {
        "torch": validate_compiler_artifacts(first_onnx, first_enc),
        "onnx": validate_compiler_artifacts(second["onnx"], second["encodings"]),
    }
    metadata = json.loads(second["metadata"].read_text())
    onnx_comparison = compare_models(first_onnx, second["onnx"])
    enc_comparison = compare_encodings(first, second_enc)
    identical = onnx_comparison["file_equal"] and enc_comparison["json_equal"] and functional_match
    report = {
        "status": "identical" if identical else "different", "qat_steps": 0,
        "rnn_type": args.rnn_type,
        "strict_acceptance": strict_acceptance,
        "functional_match": functional_match, "torch_vs_onnx_po2": cross_evaluation,
        "torch_vs_onnx_ptq": {"comparison": ptq_comparison,
                              "prediction_differences": ptq_prediction_differences},
        "fp_checkpoint": str(args.fp_checkpoint.resolve()), "input_fingerprints": fingerprints,
        "float_state_before_ptq": before, "float_state_after_ptq": after,
        "weights_unchanged_by_ptq": before == after, "seed": args.seed,
        "calibration_batch_size": 1, "calibration_count": len(feeds), "validation_count": len(validation),
        "feed_sha256": manifest["feed_sha256"],
        "config_sha256": {p.name: sha256(p) for p in configs},
        "onnx": onnx_comparison, "encodings": enc_comparison,
        "operand_differences": compare_operands(onnx.load(first_onnx), first, second_enc),
        "encoding_files_equal": sha256(first_enc) == sha256(second["encodings"]),
        "evaluation": {"torch_ptq": ptq, "torch_po2": po2, "onnx": metadata["evaluation"]},
        "clean_onnx_vs_fp32": metadata["clean_onnx_vs_fp32"],
        "restored_quantized_vs_pre_export": metadata["restored_quantized_vs_pre_export"],
        "artifacts": {"torch_onnx": str(first_onnx), "torch_encodings": str(first_enc),
                      "onnx_onnx": str(second["onnx"]), "onnx_encodings": str(second["encodings"]),
                      "onnx_metadata": str(second["metadata"]),
                      "onnx_aimet_encodings": str(second["aimet_encodings"])},
    }
    for path, digest in fingerprints.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Input changed during verification: {path}")
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_summary(args.output_dir, report)
    print(f"ONNX identical: {onnx_comparison['file_equal']}", flush=True)
    print(f"Encoding JSON identical: {enc_comparison['json_equal']}", flush=True)
    print(f"Torch/ONNX Po2 outputs identical: {functional_match}", flush=True)
    print(f"Comparison: {report['status'].upper()}", flush=True)
    print(f"Report: {args.output_dir.resolve() / 'comparison.json'}", flush=True)
    return 0 if identical or args.report_only else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnn-type", "--rnn_type", choices=("gru", "lstm"), default="gru")
    parser.add_argument("--fp-checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path(kws.DATA_ROOT))
    parser.add_argument("--config-file", type=Path, default=kws.CONFIG_FILE)
    parser.add_argument("--bitwidth-config", type=Path, default=kws.BITWIDTH_CONFIG_FILE)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=kws.SEED)
    parser.add_argument("--output-dir", type=Path,
        default=Path("output/kws_ptq_roundtrip") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--report-only", action="store_true", help="Return 0 even when artifacts differ")
    args = parser.parse_args()
    if args.fp_checkpoint is None:
        args.fp_checkpoint = Path('output/quick_start_kws') / args.rnn_type / 'model_fp_kws.pth'
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
