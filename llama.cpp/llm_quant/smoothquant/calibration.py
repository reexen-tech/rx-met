"""Activation-scale calibration for SmoothQuant.

Adapted from https://github.com/mit-han-lab/smoothquant (MIT License),
file ``smoothquant/calibration.py``.

Unlike ``llm_quant.awq.utils.calib_data``, samples are tokenized and truncated
individually instead of being concatenated into fixed-size blocks; this keeps
the collected scales numerically identical to the reference implementation.
"""

# === REEX_SMOOTHQUANT BEGIN: per-input-channel activation scale collection ===

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset

_CALIB_PATH_ENV_VARS = ("SMOOTHQUANT_PILEVAL_PATH", "AWQ_PILEVAL_PATH")


def resolve_calib_dataset(data: str = "pileval") -> Path:
    """Resolve ``--calib_data`` to a local .jsonl file.

    ``"pileval"`` falls back to ``SMOOTHQUANT_PILEVAL_PATH`` and then to the
    ``AWQ_PILEVAL_PATH`` already used by the AWQ exporter.
    """
    if data == "pileval":
        value = next(
            (os.environ[var] for var in _CALIB_PATH_ENV_VARS if os.environ.get(var)),
            None,
        )
        if not value:
            raise ValueError(
                "Set SMOOTHQUANT_PILEVAL_PATH (or AWQ_PILEVAL_PATH) to a local "
                "PileVal val.jsonl file, or pass the local JSONL path through "
                "--calib_data."
            )
    else:
        value = data

    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Local calibration dataset not found: {path}.")
    if path.suffix != ".jsonl":
        raise ValueError(f"Expected a .jsonl calibration dataset, got: {path}")
    return path


@torch.no_grad()
def get_act_scales(
    model: nn.Module,
    tokenizer,
    dataset_path: str | Path,
    num_samples: int = 512,
    seq_len: int = 512,
    *,
    progress_every: int = 64,
) -> dict[str, torch.Tensor]:
    """Collect the per-input-channel absolute maximum seen by every Linear."""
    model.eval()
    device = next(model.parameters()).device
    act_scales: dict[str, torch.Tensor] = {}

    def stat_tensor(name: str, tensor: torch.Tensor) -> None:
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = [
        module.register_forward_hook(
            lambda m, x, y, name=name: stat_input_hook(m, x, y, name)
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    try:
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        dataset = dataset.shuffle(seed=42)
        for i in range(min(num_samples, len(dataset))):
            input_ids = tokenizer(
                dataset[i]["text"],
                return_tensors="pt",
                max_length=seq_len,
                truncation=True,
            ).input_ids.to(device)
            model(input_ids)
            if progress_every and (i + 1) % progress_every == 0:
                print(f" * calibrated {i + 1}/{num_samples} samples", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    return act_scales

# === REEX_SMOOTHQUANT END ===
