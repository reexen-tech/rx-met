"""Run Torch and ONNX PTQ on fixed CPU/GPU inputs and compare their artifacts.

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
WORKER = ROOT / "scripts/validation/kws_ptq_worker.py"


def run(command, log):
    print("+ " + shlex.join(map(str, command)), flush=True)
    with log.open("w") as stream:
        result = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def read_artifacts(directory):
    record = json.loads((directory / "artifacts.json").read_text())
    artifacts = {name: Path(record[name]) for name in ("onnx", "encodings", "outputs")}
    for path in artifacts.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return artifacts


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
    waveform_options = ["--rnn-type", args.rnn_type, "--fp-checkpoint", args.fp_checkpoint.resolve(),
                        "--calib-dir", args.calib_dir.resolve(),
                        "--validation-dir", args.validation_dir.resolve()]
    for name in ("calib_limit", "validation_limit"):
        value = getattr(args, name)
        if value is not None:
            if value < 1:
                parser.error(f"--{name.replace('_', '-')} must be positive")
            waveform_options += ["--" + name.replace("_", "-"), value]
    waveform_options = list(map(str, waveform_options))
    prefixes = {device: shlex.split(getattr(args, device + "_command"))
                for device in ("cpu", "cuda")}
    checkpoint_sha = sha256(args.fp_checkpoint)
    products, commands = {}, []
    for device in ("cpu", "cuda"):
        directory = args.output_dir / f"kws_{device}"
        command = prefixes[device] + [str(WORKER), "torch", "--device", device,
                                     "--output-dir", str(directory), *waveform_options]
        commands.append(command)
        run(command, args.output_dir / f"kws_{device}.log")
        products[f"kws_{device}"] = read_artifacts(directory)
    # Both ONNX passes start from one identical clean ONNX, preventing graph
    # differences in the first pair from changing the second pair's inputs.
    shared_onnx = products["kws_cuda"]["onnx"]
    # KWS accepts waveform feeds; its deployment ONNX accepts external MFCC.
    feature_root = args.output_dir / "mfcc_feeds"
    command = prefixes["cpu"] + [str(WORKER), "mfcc", "--output-dir", str(feature_root),
                                 *waveform_options]
    commands.append(command)
    run(command, args.output_dir / "mfcc_feeds.log")
    onnx_options = ["--calib-dir", str(feature_root / "calib"),
                    "--validation-dir", str(feature_root / "val")]
    for device in ("cpu", "cuda"):
        directory = args.output_dir / f"onnx_{device}"
        command = prefixes[device] + [str(WORKER), "onnx", "--device", device,
                                     "--model", str(shared_onnx), "--output-dir", str(directory),
                                     *onnx_options]
        commands.append(command)
        run(command, args.output_dir / f"onnx_{device}.log")
        products[f"onnx_{device}"] = read_artifacts(directory)
    comparisons = {}
    pairs = {"kws_cpu_cuda": ("kws_cpu", "kws_cuda"),
             "onnx_cpu_cuda": ("onnx_cpu", "onnx_cuda"),
             "torch_onnx_cpu": ("kws_cpu", "onnx_cpu"),
             "torch_onnx_cuda": ("kws_cuda", "onnx_cuda")}
    for name, (left_name, right_name) in pairs.items():
        left, right = products[left_name], products[right_name]
        report = args.output_dir / f"{name}_comparison.json"
        command = [sys.executable, str(ROOT / "scripts/validation/compare_ptq_artifacts.py"),
            "--left-onnx", str(left["onnx"]), "--right-onnx", str(right["onnx"]),
            "--left-encodings", str(left["encodings"]), "--right-encodings", str(right["encodings"]),
            "--left-outputs", str(left["outputs"]), "--right-outputs", str(right["outputs"]),
            "--report", str(report)]
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
    print(f"PTQ validation {'matches' if summary['equal'] else 'differs'}: {args.output_dir / 'comparison.json'}")
    return 0 if summary["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
