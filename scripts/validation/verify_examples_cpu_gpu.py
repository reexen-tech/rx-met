"""Run both public PTQ examples on fixed CPU/GPU inputs and compare their artifacts.

Each command prefix must select a Python environment with the corresponding
rx-met wheels. Container prefixes must mount this repository and input paths
at their host absolute paths. No shell is used to execute command prefixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from compare_ptq_artifacts import sha256


ROOT = Path(__file__).resolve().parents[2]


def run(command, log):
    print("+ " + shlex.join(map(str, command)), flush=True)
    with log.open("w") as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def exported(directory, filename):
    records = list(directory.rglob("run.json"))
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one example run under {directory}: {records}")
    record = json.loads(records[0].read_text())
    if record["status"] != "success":
        raise RuntimeError(f"Example did not succeed: {records[0]}")
    result = records[0].parent / "exports" / filename
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnn-type", choices=("gru", "lstm"), required=True)
    parser.add_argument("--fp-checkpoint", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-command", default=sys.executable,
                        help="Python command prefix, e.g. '/cpu-env/bin/python'")
    parser.add_argument("--cuda-command", default=sys.executable)
    parser.add_argument("--calib-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    common = ["--calib-dir", args.calib_dir.resolve(),
              "--validation-dir", args.validation_dir.resolve()]
    for name in ("calib_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
            common += ["--" + name.replace("_", "-"), str(value)]
    checkpoint_sha = sha256(args.fp_checkpoint)
    products, commands = {}, []
    for device in ("cpu", "cuda"):
        prefix = shlex.split(getattr(args, device + "_command"))
        command = prefix + [str(ROOT / "examples/quick_start_kws.py"),
            "--rnn-type", args.rnn_type, "--device", device, "--ptq-only",
            "--fp-checkpoint", str(args.fp_checkpoint.resolve()),
            "--output-dir", str(args.output_dir / ("kws_" + device)), *map(str, common)]
        commands.append(command)
        run(command, args.output_dir / f"kws_{device}.log")
        products["kws_" + device] = exported(args.output_dir / ("kws_" + device), "att_mh_rnn_kws_ptq.onnx")
    # Both ONNX passes start from one identical clean ONNX, preventing graph
    # differences in the first pair from changing the second pair's inputs.
    shared_onnx = products["kws_cuda"]
    # KWS accepts waveform feeds; its deployment ONNX accepts external MFCC.
    feature_root = args.output_dir / "mfcc_feeds"
    prepare_features = """
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from examples import quick_start_kws as kws
checkpoint, rnn_type, output, calib, validation = sys.argv[2:7]
frontend = kws.make_speech_features().eval()
kws.load_checkpoint(kws.AttMHRNN(rnn_type=rnn_type), Path(checkpoint),
                    rnn_type, 'cpu', speech_features=frontend)
root = Path(output)
root.mkdir()
manifest = {'labels': list(kws.LABELS), 'batch_size': 1, 'augmentation': False}
for folder, source, raw_limit in (('calib', calib, sys.argv[7]),
                                  ('val', validation, sys.argv[8])):
    feeds, labels = kws.load_ptq_feeds(Path(source), int(raw_limit) if raw_limit else None)
    features = kws.extract_mfcc_feeds(feeds, frontend)
    directory = root / folder
    directory.mkdir()
    files = []
    for index, feed in enumerate(features):
        filename = f'{index:05d}.npz'
        np.savez(directory / filename, **feed)
        files.append(filename)
    manifest[folder + '_files'] = files
    if labels is not None:
        manifest[folder + '_labels'] = labels
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
"""
    command = shlex.split(args.cpu_command) + ["-c", prepare_features, str(ROOT),
        str(args.fp_checkpoint.resolve()), args.rnn_type, str(feature_root),
        str(args.calib_dir.resolve()), str(args.validation_dir.resolve()),
        str(args.calib_limit or ""), str(args.validation_limit or "")]
    commands.append(command)
    run(command, args.output_dir / "mfcc_feeds.log")
    onnx_common = ["--calib-dir", feature_root / "calib",
                   "--validation-dir", feature_root / "val"]
    for device in ("cpu", "cuda"):
        prefix = shlex.split(getattr(args, device + "_command"))
        command = prefix + [str(ROOT / "examples/quick_start_onnx_ptq.py"),
            "--model", str(shared_onnx), "--device", device,
            "--prefix", "att_mh_rnn_kws_ptq", "--output-dir",
            str(args.output_dir / ("onnx_" + device)), *map(str, onnx_common)]
        commands.append(command)
        run(command, args.output_dir / f"onnx_{device}.log")
        products["onnx_" + device] = exported(args.output_dir / ("onnx_" + device), "att_mh_rnn_kws_ptq.onnx")
    comparisons = {}
    pairs = {"kws_cpu_cuda": ("kws_cpu", "kws_cuda"),
             "onnx_cpu_cuda": ("onnx_cpu", "onnx_cuda"),
             "torch_onnx_cpu": ("kws_cpu", "onnx_cpu"),
             "torch_onnx_cuda": ("kws_cuda", "onnx_cuda")}
    for name, (left_name, right_name) in pairs.items():
        left, right = products[left_name], products[right_name]
        report = args.output_dir / f"{name}_comparison.json"
        command = [sys.executable, str(ROOT / "scripts/validation/compare_ptq_artifacts.py"),
            "--left-onnx", str(left), "--right-onnx", str(right),
            "--left-encodings", str(left.with_suffix(".json")),
            "--right-encodings", str(right.with_suffix(".json")),
            "--left-outputs", str(left.parent / "outputs.npz"),
            "--right-outputs", str(right.parent / "outputs.npz"), "--report", str(report)]
        result = subprocess.run(command)
        if result.returncode not in (0, 1) or not report.is_file():
            raise RuntimeError(f"Artifact comparison failed: {report}")
        comparisons[name] = json.loads(report.read_text())
    if sha256(args.fp_checkpoint) != checkpoint_sha:
        raise RuntimeError("The shared floating checkpoint changed during validation")
    summary = {"comparison": "exact", "rnn_type": args.rnn_type,
               "fp_checkpoint_sha256": checkpoint_sha, "shared_onnx_sha256": sha256(shared_onnx),
               "commands": commands, "equal": all(value["equal"] for value in comparisons.values()),
               "comparisons": {name: {"equal": value["equal"], "report": f"{name}_comparison.json"}
                               for name, value in comparisons.items()}}
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Example validation {'matches' if summary['equal'] else 'differs'}: {args.output_dir / 'comparison.json'}")
    return 0 if summary["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
