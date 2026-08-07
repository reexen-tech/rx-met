#!/usr/bin/env python3
"""Compare original and SmoothQuant Qwen3.5 BF16 logits on a fixed holdout."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Qwen3_5MoeForCausalLM

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from llm_quant.smoothquant.adapters import qwen35_calibration_targets  # noqa: E402
from llm_quant.smoothquant.loading import (  # noqa: E402
    keep_qwen35_moe_norms_in_fp32,
    load_qwen35_text_model,
    validate_loading_info,
    validate_qwen35_smoothed_parameter_dtypes,
    validate_qwen35_text_structure,
)
from llm_quant.smoothquant.runner import (  # noqa: E402
    _file_fingerprint,
    _source_model_fingerprint,
)


def _load_holdout(path: Path) -> list[dict[str, str]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record.get("category"), str) or not isinstance(
            record.get("text"), str
        ):
            raise ValueError(f"invalid holdout record at {path}:{line_number}")
        records.append({"category": record["category"], "text": record["text"]})
    categories = {record["category"] for record in records}
    expected = {"general", "code", "math", "dialogue"}
    if categories != expected:
        raise ValueError(f"holdout categories {categories} != {expected}")
    return records


def _assert_no_calibration_overlap(
    calibration_path: Path, holdout: list[dict[str, str]]
) -> None:
    holdout_texts = {record["text"] for record in holdout}
    with calibration_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("text") in holdout_texts:
                raise ValueError(
                    f"holdout/calibration overlap at calibration line {line_number}"
                )


def _render_holdout(tokenizer, holdout: list[dict[str, str]]) -> list[dict[str, str]]:
    rendered = []
    for record in holdout:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered.append({**record, "rendered": text})
    return rendered


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _capture_router_indices(model, input_ids: torch.Tensor):
    targets = qwen35_calibration_targets(model)
    if targets is None:
        raise TypeError("router capture requires Qwen3.5 MoE")
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for target in targets.routers:
        hooks.append(
            target.module.register_forward_hook(
                lambda _module, _inputs, output, name=target.layer_name: captured.__setitem__(
                    name, output[2].detach().cpu()
                )
            )
        )
    try:
        output = model(input_ids=input_ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    expected_names = tuple(target.layer_name for target in targets.routers)
    if tuple(captured) != expected_names:
        raise RuntimeError(
            f"router capture order mismatch: {tuple(captured)} != {expected_names}"
        )
    return output.logits, captured


def _model_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device


def _collect_reference(
    model,
    tokenizer,
    rendered_holdout: list[dict[str, str]],
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    references = []
    all_finite = True
    token_counts = []
    with torch.inference_mode():
        for record in rendered_holdout:
            input_ids = tokenizer(
                record["rendered"],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).input_ids.to(_model_input_device(model))
            logits, routers = _capture_router_indices(model, input_ids)
            logits = logits.squeeze(0).float().cpu()
            all_finite = all_finite and bool(torch.isfinite(logits).all().item())
            token_counts.append(input_ids.numel())
            references.append(
                {
                    "category": record["category"],
                    "input_ids": input_ids.cpu(),
                    "logits": logits,
                    "routers": routers,
                }
            )
    return references, {
        "all_finite": all_finite,
        "token_counts": token_counts,
        "total_tokens": sum(token_counts),
    }


def _compare_target(model, references: list[dict[str, Any]]) -> dict[str, Any]:
    absolute_sum = 0.0
    absolute_max = 0.0
    logit_count = 0
    relative_samples = []
    absolute_samples = []
    kld_values = []
    same_top = 0
    token_count = 0
    router_same = 0
    router_count = 0
    all_finite = True

    with torch.inference_mode():
        for reference in references:
            input_ids = reference["input_ids"].to(_model_input_device(model))
            target_logits, target_routers = _capture_router_indices(model, input_ids)
            target_logits = target_logits.squeeze(0).float()
            base_logits = reference["logits"].to(target_logits.device)
            all_finite = all_finite and bool(torch.isfinite(target_logits).all().item())
            if target_logits.shape != base_logits.shape:
                raise ValueError(
                    f"logit shape mismatch: {target_logits.shape} != {base_logits.shape}"
                )

            absolute = (target_logits - base_logits).abs()
            absolute_sum += float(absolute.sum().item())
            absolute_max = max(absolute_max, float(absolute.max().item()))
            logit_count += absolute.numel()
            symmetric_denom = torch.maximum(
                target_logits.abs(), base_logits.abs()
            ).clamp_min(1e-6)
            relative = absolute / symmetric_denom
            stride = max(1, absolute.numel() // 250_000)
            absolute_samples.append(absolute.reshape(-1)[::stride].cpu())
            relative_samples.append(relative.reshape(-1)[::stride].cpu())

            base_logp = torch.log_softmax(base_logits, dim=-1)
            target_logp = torch.log_softmax(target_logits, dim=-1)
            kld = (base_logp.exp() * (base_logp - target_logp)).sum(dim=-1)
            kld_values.append(kld.cpu())
            same_top += int(
                (base_logits.argmax(dim=-1) == target_logits.argmax(dim=-1)).sum().item()
            )
            token_count += target_logits.shape[0]

            for layer_name, base_indices in reference["routers"].items():
                target_indices = target_routers[layer_name]
                if target_indices.shape != base_indices.shape:
                    raise ValueError(
                        f"router shape mismatch at {layer_name}: "
                        f"{target_indices.shape} != {base_indices.shape}"
                    )
                router_same += int((target_indices == base_indices).sum().item())
                router_count += base_indices.numel()

            del base_logits, target_logits, absolute, relative, base_logp, target_logp

    absolute_sample = torch.cat(absolute_samples)
    relative_sample = torch.cat(relative_samples)
    kld = torch.cat(kld_values)
    quantiles = torch.tensor([0.50, 0.95, 0.99])
    absolute_q = torch.quantile(absolute_sample, quantiles)
    relative_q = torch.quantile(relative_sample, quantiles)
    kld_q = torch.quantile(kld, quantiles)
    return {
        "logits_absolute": {
            "max": absolute_max,
            "mean": absolute_sum / logit_count,
            "p50_sampled": float(absolute_q[0].item()),
            "p95_sampled": float(absolute_q[1].item()),
            "p99_sampled": float(absolute_q[2].item()),
            "sample_count": absolute_sample.numel(),
        },
        "logits_symmetric_relative": {
            "definition": "abs(target-base) / max(abs(target), abs(base), 1e-6)",
            "mean_sampled": float(relative_sample.mean().item()),
            "p50_sampled": float(relative_q[0].item()),
            "p95_sampled": float(relative_q[1].item()),
            "p99_sampled": float(relative_q[2].item()),
            "max_sampled": float(relative_sample.max().item()),
            "sample_count": relative_sample.numel(),
        },
        "kld_base_to_target": {
            "mean": float(kld.mean().item()),
            "max": float(kld.max().item()),
            "p50": float(kld_q[0].item()),
            "p95": float(kld_q[1].item()),
            "p99": float(kld_q[2].item()),
            "token_count": kld.numel(),
        },
        "same_top_p": {
            "definition": "same highest-probability token as the original model",
            "fraction": same_top / token_count,
            "percent": 100.0 * same_top / token_count,
            "same_tokens": same_top,
            "token_count": token_count,
        },
        "router_topk": {
            "fraction": router_same / router_count,
            "percent": 100.0 * router_same / router_count,
            "same_indices": router_same,
            "total_indices": router_count,
        },
        "all_target_logits_finite": all_finite,
    }


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_smoothed(path: Path, *, device: str):
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "qwen3_5_moe_text":
        raise ValueError(f"smoothed checkpoint is not text-only Qwen3.5: {config.model_type}")
    with keep_qwen35_moe_norms_in_fp32():
        model, loading_info = Qwen3_5MoeForCausalLM.from_pretrained(
            path,
            config=config,
            dtype=torch.bfloat16,
            device_map=None if device == "cpu" else device,
            local_files_only=True,
            output_loading_info=True,
        )
    if device == "cpu":
        model.to("cpu")
    validate_loading_info(loading_info)
    validate_qwen35_text_structure(model, config, require_production_shape=True)
    validate_qwen35_smoothed_parameter_dtypes(model)
    model.eval()
    model.config.use_cache = False
    return model


def _parse_target(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("smoothed model must be LABEL=/path/to/model")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("smoothed model must be LABEL=/path/to/model")
    return label, Path(path).expanduser()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument(
        "--smoothed-model",
        action="append",
        type=_parse_target,
        required=True,
        help="Repeatable LABEL=/path/to/text-only-smoothed-HF",
    )
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    source_path = Path(args.source_model).expanduser().resolve()
    holdout_path = Path(args.holdout).expanduser().resolve()
    calibration_path = Path(args.calibration_data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    holdout = _load_holdout(holdout_path)
    _assert_no_calibration_overlap(calibration_path, holdout)

    tokenizer = AutoTokenizer.from_pretrained(source_path, local_files_only=True)
    rendered = _render_holdout(tokenizer, holdout)
    rendered_bytes = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in rendered
    ).encode()

    outer_config = AutoConfig.from_pretrained(source_path, local_files_only=True)
    model, source_loading = load_qwen35_text_model(
        source_path,
        outer_config.text_config,
        dtype=torch.bfloat16,
        device=args.device,
        require_production_shape=True,
    )
    model.eval()
    model.config.use_cache = False
    references, reference_summary = _collect_reference(
        model, tokenizer, rendered, args.max_length
    )
    del model
    _release_cuda_memory()

    comparisons = {}
    target_fingerprints = {}
    for label, path in args.smoothed_model:
        resolved = path.resolve()
        target = _load_smoothed(resolved, device=args.device)
        comparisons[label] = _compare_target(target, references)
        target_fingerprints[label] = _source_model_fingerprint(str(resolved))
        del target
        _release_cuda_memory()

    git_commit = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = {
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "bfloat16",
            "git_commit": git_commit,
        },
        "inputs": {
            "source_model": _source_model_fingerprint(str(source_path)),
            "smoothed_models": target_fingerprints,
            "holdout": _file_fingerprint(holdout_path),
            "rendered_holdout_sha256": _sha256_bytes(rendered_bytes),
            "calibration_data": _file_fingerprint(calibration_path),
            "holdout_calibration_exact_text_overlap": False,
            "categories": [record["category"] for record in holdout],
        },
        "reference": {
            **reference_summary,
            "loading": source_loading,
        },
        "comparisons": comparisons,
        "thresholds": None,
        "note": "BF16 equivalence thresholds intentionally unset pending audit point 1.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
