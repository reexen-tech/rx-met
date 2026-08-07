#!/usr/bin/env python3
"""Run the fixed ADA300 PWNL Qwen3.5 end-to-end experiment matrix.

This is a host-side orchestrator. Model evaluation runs inside rx-met-zcx,
while GPU occupancy is sampled from the host so unrelated processes cannot be
hidden by the container PID namespace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/chengxing.zou.srv/projects/aimet_rx/llama.cpp")
RESULT_ROOT = REPO / "tests/test-reex/results/ada300_pwnl_lut_qwen35"
GGUF_MANIFEST = RESULT_ROOT / "artifacts/gguf-manifest.json"
CONTAINER = "rx-met-zcx"
BUILDS = {
    "nolut": REPO / "build_cuda_q64_nolut",
    "lut": REPO / "build_cuda_q64_lut",
}
DATASETS = {
    "wikitext103": {
        "path": Path("/mnt/data1/share/datasets/model_evaluation/wikitext-103-v1/test-00000-of-00001.raw"),
        "size_bytes": 1_279_610,
        "sha256": "5c3f54ac993e8e6bbf53c5f531f08c50c28c98c295250883468196169e3532cc",
    },
    "cci2": {
        "path": Path("/mnt/data1/share/datasets/model_evaluation/CCI2-Data/data/cci2-00000-of-00178.raw"),
        "size_bytes": 2_884_374_398,
        "sha256": "1921f4638832ff06a9e509c3532159b9f5fa94a30339e16a5e2436f029646988",
    },
}
MODELS = {
    "qwen35-9b": {
        "display": "Qwen3.5-9B",
        "gguf_dir": Path("/mnt/data1/share/models/Qwen/Qwen3.5-9B/GGUF"),
        "files": {
            "f16": "Qwen3.5-9B-F16.gguf",
            "q8": "Qwen3.5-9B-Q8_0_64.gguf",
            "q4": "Qwen3.5-9B-Q4_0_64.gguf",
        },
    },
    "qwen35-35b-a3b": {
        "display": "Qwen3.5-35B-A3B",
        "gguf_dir": Path("/mnt/data1/share/models/Qwen/Qwen3.5-35B-A3B/GGUF"),
        "files": {
            "f16": "Qwen3.5-35B-A3B-F16.gguf",
            "q8": "Qwen3.5-35B-A3B-Q8_0_64.gguf",
            "q4": "Qwen3.5-35B-A3B-Q4_0_64.gguf",
        },
    },
}

CTX = 4096
CHUNKS = 32
BATCH = 512
NGL = 99
GPU_INDEX = 0
MIN_BASELINE_HEADROOM = 40 * 1024**3
CUDA_ENV = {
    "HOME": "/root",
    "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
    "CUDA_VISIBLE_DEVICES": str(GPU_INDEX),
}


@dataclass(frozen=True)
class RunSpec:
    number: int
    model_key: str
    dataset_key: str
    name: str
    build: str
    weight: str
    save_baseline: str | None = None
    compare_baseline: str | None = None
    comparison: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.model_key}__{self.dataset_key}__{self.number:02d}-{self.name}"

    @property
    def model_path(self) -> Path:
        model = MODELS[self.model_key]
        return model["gguf_dir"] / model["files"][self.weight]

    @property
    def large_dir(self) -> Path:
        return MODELS[self.model_key]["gguf_dir"] / "ada300_pwnl_eval" / self.dataset_key

    def baseline_path(self, baseline_name: str) -> Path:
        return self.large_dir / f"{baseline_name}.kld"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_checked(args: list[str], *, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(args, check=True, capture_output=True, text=text)


def docker_command(inner: list[str]) -> list[str]:
    env_args = [item for key, value in CUDA_ENV.items() for item in ("-e", f"{key}={value}")]
    clean_env = ["env", "-i", *(f"{key}={value}" for key, value in CUDA_ENV.items())]
    return ["docker", "exec", "-w", str(REPO), *env_args, CONTAINER, *clean_env, *inner]


def matrix() -> list[RunSpec]:
    template = [
        (1, "f16-nolut-baseline", "nolut", "f16", "f16_nolut", None, None),
        (2, "q8-nolut-baseline", "nolut", "q8", "q8_nolut", None, None),
        (3, "q8-nolut-vs-f16", "nolut", "q8", None, "f16_nolut", "vs_f16_nolut"),
        (4, "q4-nolut-baseline", "nolut", "q4", "q4_nolut", None, None),
        (5, "q4-nolut-vs-f16", "nolut", "q4", None, "f16_nolut", "vs_f16_nolut"),
        (6, "f16-lut-vs-f16", "lut", "f16", None, "f16_nolut", "vs_f16_nolut"),
        (7, "q8-lut-vs-f16", "lut", "q8", None, "f16_nolut", "vs_f16_nolut"),
        (8, "q8-lut-vs-q8", "lut", "q8", None, "q8_nolut", "vs_same_weight_nolut"),
        (9, "q4-lut-vs-f16", "lut", "q4", None, "f16_nolut", "vs_f16_nolut"),
        (10, "q4-lut-vs-q4", "lut", "q4", None, "q4_nolut", "vs_same_weight_nolut"),
    ]
    runs: list[RunSpec] = []
    # Small/fast combinations come first, but every invocation keeps the same
    # fixed 10-run order within a model/dataset pair.
    for model_key in MODELS:
        for dataset_key in DATASETS:
            for number, name, build, weight, save, compare, comparison in template:
                runs.append(RunSpec(number, model_key, dataset_key, name, build, weight, save, compare, comparison))
    return runs


def load_gguf_manifest() -> dict[str, dict[str, Any]]:
    payload = json.loads(GGUF_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema") != "ada300-pwnl-gguf-manifest-v1":
        raise RuntimeError(f"unexpected GGUF manifest schema: {GGUF_MANIFEST}")
    return {entry["path"]: entry for entry in payload["artifacts"]}


def parse_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        values[key] = value
    return values


def compute_processes() -> list[dict[str, str]]:
    query = run_checked([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
        "-i",
        str(GPU_INDEX),
    ]).stdout.strip()
    processes = []
    for line in query.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            processes.append({"pid": parts[0], "name": parts[1], "used_mib": parts[2]})
    return processes


def gpu_snapshot() -> dict[str, Any]:
    row = run_checked([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        str(GPU_INDEX),
    ]).stdout.strip().split(",")
    if len(row) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {row}")
    return {
        "timestamp_utc": utc_now(),
        "name": row[0].strip(),
        "driver_version": row[1].strip(),
        "total_mib": int(row[2]),
        "used_mib": int(row[3]),
        "free_mib": int(row[4]),
        "utilization_percent": int(row[5]),
        "processes": compute_processes(),
    }


def validate_kld(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "valid": False}
    if not path.is_file():
        result["error"] = "missing"
        return result
    size = path.stat().st_size
    result["size_bytes"] = size
    if size < 20:
        result["error"] = "short header"
        return result
    with path.open("rb") as stream:
        header = stream.read(20)
    magic = header[:8]
    n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", header[8:20])
    result.update({"magic": magic.decode("ascii", "replace"), "n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk})
    if magic != b"_logits_":
        result["error"] = "bad magic"
        return result
    if (n_ctx, n_chunk) != (CTX, CHUNKS):
        result["error"] = f"unexpected dimensions: ctx={n_ctx}, chunks={n_chunk}"
        return result
    if n_vocab <= 0:
        result["error"] = f"invalid vocabulary: {n_vocab}"
        return result
    scored_tokens = n_ctx - 1 - n_ctx // 2
    packed_vocab = 2 * ((n_vocab + 1) // 2) + 4
    expected = 20 + n_ctx * n_chunk * 4 + n_chunk * scored_tokens * packed_vocab * 2
    result["expected_size_bytes"] = expected
    if size != expected:
        result["error"] = f"size mismatch: got {size}, expected {expected}"
        return result
    result["valid"] = True
    return result


FLOAT = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def last_match(pattern: str, content: str) -> tuple[float, ...] | None:
    matches = list(re.finditer(pattern, content, flags=re.MULTILINE))
    if not matches:
        return None
    values = tuple(float(value) for value in matches[-1].groups())
    return values if all(math.isfinite(value) for value in values) else None


def parse_and_validate_log(path: Path, is_kld: bool) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8", errors="replace")
    errors = []
    fatal_patterns = {
        "decode failure": r"failed to decode|llama_decode\(\) failed",
        "CUDA failure": r"CUDA error|CUDA failure|cudaError|GGML_ASSERT|CUDA out of memory",
        "model load failure": r"unable to load model|failed to load model|failed to create context",
        "allocation failure": r"out of memory|bad_alloc|failed to allocate",
        "corrupt input": r"does not look like a file|failed reading|inconsistent vocabulary",
    }
    for label, pattern in fatal_patterns.items():
        if re.search(pattern, content, flags=re.IGNORECASE):
            errors.append(label)

    metrics: dict[str, Any] = {}
    if is_kld:
        required = [
            "====== Perplexity statistics ======",
            "====== KL divergence statistics ======",
            "====== Token probability statistics ======",
        ]
        for marker in required:
            if marker not in content:
                errors.append(f"missing terminal marker: {marker}")
        patterns = {
            "ppl": rf"Mean PPL\(Q\)\s*:\s*{FLOAT}\s*(?:±|\+/-)\s*{FLOAT}",
            "ppl_base": rf"Mean PPL\(base\)\s*:\s*{FLOAT}\s*(?:±|\+/-)\s*{FLOAT}",
            "ppl_ratio": rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*{FLOAT}\s*(?:±|\+/-)\s*{FLOAT}",
            "mean_kld": rf"Mean\s+KLD:\s*{FLOAT}\s*(?:±|\+/-)\s*{FLOAT}",
            "same_top1_percent": rf"Same top p:\s*{FLOAT}\s*(?:±|\+/-)\s*{FLOAT}\s*%",
        }
        for name, pattern in patterns.items():
            match = last_match(pattern, content)
            if match is None:
                errors.append(f"missing or non-finite metric: {name}")
            else:
                metrics[name] = {"value": match[0], "uncertainty": match[1]}
    else:
        ppl = last_match(rf"Final estimate: PPL =\s*{FLOAT}\s*\+/-\s*{FLOAT}", content)
        if ppl is None:
            errors.append("missing or non-finite final PPL")
        else:
            metrics["ppl"] = {"value": ppl[0], "uncertainty": ppl[1]}
        if re.search(r"\[32\]" + FLOAT, content) is None:
            errors.append("missing final chunk marker [32]")

    setup = re.search(
        rf"(?:calculating perplexity|computing) over {CHUNKS} chunks, n_ctx={CTX}, batch_size={BATCH}, n_seq=1",
        content,
    )
    if setup is None:
        errors.append("fixed runtime parameters were not confirmed by llama-perplexity")

    throughputs = [float(value) for value in re.findall(rf"{FLOAT}\s+tokens per second", content)]
    throughputs = [value for value in throughputs if math.isfinite(value)]
    if throughputs:
        metrics["reported_tokens_per_second"] = throughputs
    else:
        errors.append("missing throughput metric")
    return {"valid": not errors, "errors": errors, "metrics": metrics}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight(*, require_idle: bool, hash_datasets: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    try:
        running = run_checked(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER]).stdout.strip()
        if running != "true":
            errors.append(f"container {CONTAINER} is not running")
    except subprocess.CalledProcessError as exc:
        errors.append(f"cannot inspect container {CONTAINER}: {exc}")

    expected_cache = {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_CUDA": "ON",
        "GGML_USE_REEX_Q64": "ON",
        "GGML_REEX_FP16_PIPELINE": "OFF",
        "GGML_REEX_GEMM": "OFF",
        "GGML_LUT_NUM_SEGMENTS_REEX": "16",
    }
    cache_summary: dict[str, dict[str, str]] = {}
    for build_name, build_dir in BUILDS.items():
        binary = build_dir / "bin/llama-perplexity"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            errors.append(f"missing executable: {binary}")
        cache_path = build_dir / "CMakeCache.txt"
        if not cache_path.is_file():
            errors.append(f"missing CMake cache: {cache_path}")
            continue
        cache = parse_cache(cache_path)
        selected = {key: cache.get(key, "<missing>") for key in (*expected_cache, "GGML_USE_REEX")}
        cache_summary[build_name] = selected
        for key, expected in expected_cache.items():
            if selected[key] != expected:
                errors.append(f"{build_name} {key}={selected[key]}, expected {expected}")
        expected_reex = "ON" if build_name == "lut" else "OFF"
        if selected["GGML_USE_REEX"] != expected_reex:
            errors.append(f"{build_name} GGML_USE_REEX={selected['GGML_USE_REEX']}, expected {expected_reex}")

    gguf_entries = load_gguf_manifest()
    artifact_summary = []
    for model in MODELS.values():
        for filename in model["files"].values():
            path = model["gguf_dir"] / filename
            entry = gguf_entries.get(str(path))
            if entry is None:
                errors.append(f"GGUF absent from manifest: {path}")
                continue
            actual_size = path.stat().st_size if path.is_file() else -1
            if actual_size != entry["size_bytes"]:
                errors.append(f"GGUF size mismatch: {path}: {actual_size} != {entry['size_bytes']}")
            artifact_summary.append({
                "path": str(path),
                "size_bytes": actual_size,
                "sha256": entry["sha256"],
                "sha256_source": str(GGUF_MANIFEST),
            })

    dataset_summary = []
    for key, dataset in DATASETS.items():
        path = dataset["path"]
        size = path.stat().st_size if path.is_file() else -1
        digest = sha256_file(path) if hash_datasets and path.is_file() else None
        if size != dataset["size_bytes"]:
            errors.append(f"dataset size mismatch: {path}: {size} != {dataset['size_bytes']}")
        if digest is not None and digest != dataset["sha256"]:
            errors.append(f"dataset SHA-256 mismatch: {path}: {digest} != {dataset['sha256']}")
        dataset_summary.append({"id": key, "path": str(path), "size_bytes": size, "sha256": digest or dataset["sha256"]})

    gpu = gpu_snapshot()
    if require_idle and gpu["processes"]:
        description = ", ".join(f"PID {p['pid']} {p['name']} {p['used_mib']} MiB" for p in gpu["processes"])
        errors.append(f"GPU {GPU_INDEX} is not exclusive: {description}")

    remaining_baselines = 0
    for spec in matrix():
        if not spec.save_baseline:
            continue
        path = spec.baseline_path(spec.save_baseline)
        if not validate_kld(path)["valid"]:
            remaining_baselines += 1
    expected_one = 20 + CTX * CHUNKS * 4 + CHUNKS * (CTX - 1 - CTX // 2) * (2 * ((248_320 + 1) // 2) + 4) * 2
    free_bytes = shutil.disk_usage(next(iter(MODELS.values()))["gguf_dir"]).free
    required_bytes = remaining_baselines * expected_one + MIN_BASELINE_HEADROOM
    if free_bytes < required_bytes:
        errors.append(f"insufficient disk: {free_bytes} bytes free, {required_bytes} required")

    return {
        "schema": "ada300-pwnl-phase2-preflight-v1",
        "timestamp_utc": utc_now(),
        "valid": not errors,
        "errors": errors,
        "fixed_parameters": {"ctx": CTX, "chunks": CHUNKS, "batch": BATCH, "ngl": NGL, "gpu_index": GPU_INDEX},
        "container": CONTAINER,
        "gpu": gpu,
        "build_cache": cache_summary,
        "datasets": dataset_summary,
        "gguf_artifacts": artifact_summary,
        "gguf_manifest_sha256": sha256_file(GGUF_MANIFEST),
        "baseline_storage": {
            "remaining_files": remaining_baselines,
            "expected_bytes_each": expected_one,
            "free_bytes": free_bytes,
            "required_bytes_including_headroom": required_bytes,
        },
    }


def done_path(spec: RunSpec) -> Path:
    return RESULT_ROOT / "status" / f"{spec.run_id}.done.json"


def is_complete(spec: RunSpec) -> bool:
    marker = done_path(spec)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not payload.get("valid"):
        return False
    if spec.save_baseline:
        return validate_kld(spec.baseline_path(spec.save_baseline))["valid"]
    return True


def experiment_args(spec: RunSpec, partial_baseline: Path | None) -> list[str]:
    binary = BUILDS[spec.build] / "bin/llama-perplexity"
    args = [
        str(binary),
        "-m", str(spec.model_path),
        "-f", str(DATASETS[spec.dataset_key]["path"]),
        "-c", str(CTX),
        "--chunks", str(CHUNKS),
        "-b", str(BATCH),
        "-ngl", str(NGL),
    ]
    if spec.save_baseline:
        assert partial_baseline is not None
        args.extend(["--save-all-logits", str(partial_baseline)])
    else:
        assert spec.compare_baseline is not None
        args.extend([
            "--kl-divergence",
            "--kl-divergence-base", str(spec.baseline_path(spec.compare_baseline)),
        ])
    return args


def sample_gpu(csv_writer: csv.writer) -> tuple[int, list[dict[str, str]]]:
    snapshot = gpu_snapshot()
    processes = snapshot["processes"]
    csv_writer.writerow([
        snapshot["timestamp_utc"], snapshot["used_mib"], snapshot["free_mib"],
        snapshot["utilization_percent"], json.dumps(processes, separators=(",", ":")),
    ])
    return snapshot["used_mib"], processes


def run_one(spec: RunSpec, sample_seconds: float) -> bool:
    if is_complete(spec):
        print(f"SKIP {spec.run_id}: validated done marker exists", flush=True)
        return True

    # Baselines are dependencies and are never silently regenerated over an
    # existing canonical file.
    if spec.compare_baseline:
        baseline_check = validate_kld(spec.baseline_path(spec.compare_baseline))
        if not baseline_check["valid"]:
            raise RuntimeError(f"invalid prerequisite for {spec.run_id}: {baseline_check}")
    if spec.save_baseline:
        canonical = spec.baseline_path(spec.save_baseline)
        if canonical.exists():
            check = validate_kld(canonical)
            raise RuntimeError(f"canonical baseline exists without a valid done marker; preserve and audit it: {check}")

    occupied = compute_processes()
    if occupied:
        raise RuntimeError(f"GPU became non-exclusive before {spec.run_id}: {occupied}")

    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = RESULT_ROOT / "logs" / "runs" / spec.model_key / spec.dataset_key
    status_dir = RESULT_ROOT / "status" / "attempts"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    spec.large_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.number:02d}-{spec.name}.{attempt}.log"
    gpu_path = log_dir / f"{spec.number:02d}-{spec.name}.{attempt}.gpu.csv"
    status_path = status_dir / f"{spec.run_id}.{attempt}.json"
    partial_baseline = spec.large_dir / f"{spec.save_baseline}.{attempt}.partial.kld" if spec.save_baseline else None
    command = docker_command(experiment_args(spec, partial_baseline))

    start = time.monotonic()
    start_utc = utc_now()
    peak_used_mib = 0
    observed_processes: dict[tuple[str, str], dict[str, str]] = {}
    unexpected_processes: dict[tuple[str, str], dict[str, str]] = {}
    print(f"START {spec.run_id}", flush=True)
    print(f"  log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_stream, gpu_path.open("w", newline="", encoding="utf-8") as gpu_stream:
        log_stream.write(f"# start_utc={start_utc}\n# command={shlex.join(command)}\n")
        log_stream.flush()
        writer = csv.writer(gpu_stream)
        writer.writerow(["timestamp_utc", "used_mib", "free_mib", "utilization_percent", "compute_processes_json"])
        process = subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            try:
                used_mib, processes = sample_gpu(writer)
                gpu_stream.flush()
                peak_used_mib = max(peak_used_mib, used_mib)
                for item in processes:
                    key = (item["pid"], item["name"])
                    observed_processes[key] = item
                    if "llama-perplex" not in item["name"].lower():
                        unexpected_processes[key] = item
            except Exception as exc:  # Sampling failure invalidates the run but must not kill it.
                log_stream.write(f"\n# gpu_sample_error={exc!r}\n")
                unexpected_processes[("sampler", repr(exc))] = {"pid": "sampler", "name": repr(exc), "used_mib": "unknown"}
            time.sleep(sample_seconds)
        exit_code = process.wait()
        try:
            used_mib, processes = sample_gpu(writer)
            peak_used_mib = max(peak_used_mib, used_mib)
            for item in processes:
                observed_processes[(item["pid"], item["name"])] = item
        except Exception as exc:
            log_stream.write(f"\n# final_gpu_sample_error={exc!r}\n")
            unexpected_processes[("sampler", repr(exc))] = {"pid": "sampler", "name": repr(exc), "used_mib": "unknown"}

    end_utc = utc_now()
    wall_seconds = time.monotonic() - start
    log_check = parse_and_validate_log(log_path, is_kld=not bool(spec.save_baseline))
    baseline_check = validate_kld(partial_baseline) if partial_baseline else None
    errors = list(log_check["errors"])
    if exit_code != 0:
        errors.append(f"nonzero exit code: {exit_code}")
    if unexpected_processes:
        errors.append(f"unexpected GPU process or sampling failure: {list(unexpected_processes.values())}")
    if baseline_check is not None and not baseline_check["valid"]:
        errors.append(f"invalid generated KLD baseline: {baseline_check.get('error')}")

    status: dict[str, Any] = {
        "schema": "ada300-pwnl-phase2-run-v1",
        "run": asdict(spec),
        "run_id": spec.run_id,
        "attempt": attempt,
        "valid": not errors,
        "errors": errors,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "wall_seconds": wall_seconds,
        "exit_code": exit_code,
        "command": command,
        "log_path": str(log_path),
        "gpu_samples_path": str(gpu_path),
        "peak_gpu_used_mib": peak_used_mib,
        "observed_gpu_processes": list(observed_processes.values()),
        "log_validation": log_check,
        "generated_baseline": baseline_check,
    }

    if not errors and partial_baseline is not None:
        canonical = spec.baseline_path(spec.save_baseline or "")
        partial_baseline.rename(canonical)
        status["generated_baseline"]["path"] = str(canonical)
        status["generated_baseline"]["sha256"] = sha256_file(canonical)
    write_json(status_path, status)
    if not errors:
        write_json(done_path(spec), status)
        print(f"DONE  {spec.run_id}: {wall_seconds:.1f}s, peak GPU {peak_used_mib} MiB", flush=True)
        return True

    print(f"FAIL  {spec.run_id}: {'; '.join(errors)}", file=sys.stderr, flush=True)
    print(f"  retained status: {status_path}", file=sys.stderr, flush=True)
    return False


def collect_metadata(preflight_result: dict[str, Any]) -> Path:
    metadata_dir = RESULT_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    git_head = run_checked(["git", "-C", str(REPO), "rev-parse", "HEAD"]).stdout.strip()
    git_status = run_checked(["git", "-C", str(REPO), "status", "--short"]).stdout.splitlines()
    abc_root = Path("/home/chengxing.zou.srv/projects/abc_lut")
    abc_head = run_checked(["git", "-C", str(abc_root), "rev-parse", "HEAD"]).stdout.strip()
    abc_status = run_checked(["git", "-C", str(abc_root), "status", "--short"]).stdout.splitlines()
    versions = {
        "host_uname": run_checked(["uname", "-a"]).stdout.strip(),
        "docker_image": run_checked(["docker", "inspect", "-f", "{{.Config.Image}}", CONTAINER]).stdout.strip(),
        "container_id": run_checked(["docker", "inspect", "-f", "{{.Id}}", CONTAINER]).stdout.strip(),
        "container_cuda": run_checked(docker_command(["nvcc", "--version"])).stdout.strip(),
        "container_cmake": run_checked(docker_command(["cmake", "--version"])).stdout.splitlines()[0],
        "container_compiler": run_checked(docker_command(["c++", "--version"])).stdout.splitlines()[0],
    }
    payload = {
        "schema": "ada300-pwnl-phase2-reproducibility-v1",
        "created_utc": utc_now(),
        "preflight": preflight_result,
        "repositories": {
            "aimet_rx_llama_cpp": {"path": str(REPO), "head": git_head, "status_short": git_status},
            "abc_lut_read_only": {"path": str(abc_root), "head": abc_head, "status_short": abc_status},
        },
        "versions": versions,
        "matrix": [asdict(spec) | {"run_id": spec.run_id, "model_path": str(spec.model_path)} for spec in matrix()],
    }
    path = metadata_dir / f"reproducibility.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(path, payload)
    return path


def print_status() -> int:
    completed = 0
    for spec in matrix():
        state = "DONE" if is_complete(spec) else "PENDING"
        completed += state == "DONE"
        print(f"{state:7} {spec.run_id}")
    print(f"\n{completed}/40 validated runs complete")
    return 0 if completed == 40 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("matrix", help="print the fixed 40-run matrix")
    subparsers.add_parser("status", help="show resumable run status")
    preflight_parser = subparsers.add_parser("preflight", help="validate artifacts, builds, disk, datasets, and exclusive GPU")
    preflight_parser.add_argument("--allow-busy-gpu", action="store_true", help="diagnostic only; never accepted by the run action")
    run_parser = subparsers.add_parser("run", help="run all pending experiments sequentially")
    run_parser.add_argument("--sample-seconds", type=float, default=2.0)
    run_parser.add_argument("--only", help="run exactly one full run_id (dependencies must already exist)")
    args = parser.parse_args()

    if args.action == "matrix":
        for spec in matrix():
            print(json.dumps(asdict(spec) | {"run_id": spec.run_id, "model_path": str(spec.model_path)}, sort_keys=True))
        return 0
    if args.action == "status":
        return print_status()

    require_idle = not (args.action == "preflight" and args.allow_busy_gpu)
    check = preflight(require_idle=require_idle)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    preflight_path = RESULT_ROOT / "metadata" / f"preflight.{stamp}.json"
    write_json(preflight_path, check)
    print(json.dumps(check, indent=2, sort_keys=True))
    print(f"preflight record: {preflight_path}")
    if not check["valid"]:
        return 2
    metadata_path = collect_metadata(check)
    print(f"reproducibility record: {metadata_path}")
    if args.action == "preflight":
        return 0

    if args.sample_seconds <= 0:
        parser.error("--sample-seconds must be positive")
    selected = matrix()
    if args.only:
        selected = [spec for spec in selected if spec.run_id == args.only]
        if not selected:
            parser.error(f"unknown --only run_id: {args.only}")
    for spec in selected:
        if not run_one(spec, args.sample_seconds):
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
