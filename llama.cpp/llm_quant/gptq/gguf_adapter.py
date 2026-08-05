"""Isolated GPTQ sidecar integration for the upstream HF-to-GGUF converter."""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
GGUF_PY = str(ROOT / "gguf-py")
if GGUF_PY not in sys.path:
    sys.path.insert(1, GGUF_PY)

import gguf  # noqa: E402

from .formats import Q64Format
from .sidecar import SidecarShardReader


logger = logging.getLogger("hf-to-gguf")


def layout_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    """Hyper-parameters used for Qwen3.5 linear-attention GGUF layout."""

    if hparams.get("model_type") == "qwen3_5_moe":
        text = hparams.get("text_config")
        if isinstance(text, dict):
            return text
    return hparams


class GPTQGGUFAdapter:
    """Load and emit packed GPTQ tensors without leaking details to converter."""

    def __init__(
        self,
        tensors: Mapping[str, dict[str, Any]],
        block_format: Q64Format,
        hparams: dict[str, Any],
    ) -> None:
        self.tensors = tensors
        self.block_format = block_format
        self.hparams = hparams
        self._remaining = set(tensors)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        hparams: dict[str, Any],
    ) -> GPTQGGUFAdapter | None:
        """Load the unique supported GPTQ sidecar, or return ``None``."""

        directory = Path(model_dir)
        paths = [
            path
            for path in (
                directory / "gptq_q4_0_64.pt",
                directory / "gptq_q4_1_64.pt",
                directory / "gptq_q8_0_64.pt",
            )
            if path.is_file()
        ]
        if not paths:
            return None
        if len(paths) > 1:
            raise ValueError(
                "multiple GPTQ sidecars found; keep exactly one manifest"
            )

        model_type = hparams.get("model_type")
        if model_type not in {"qwen2", "qwen3"} and not str(model_type).startswith(
            "qwen3_5"
        ):
            raise ValueError(
                "GPTQ Q64 sidecars support only Qwen2, Qwen3, and Qwen3.5"
            )
        reader = SidecarShardReader(paths[0])
        logger.info(
            "Loaded %d pre-quantized %s tensors",
            len(reader),
            reader.block_format.name,
        )
        return cls(reader, reader.block_format, layout_hparams(hparams))

    @property
    def file_type(self) -> gguf.LlamaFileType:
        return {
            "Q4_0_64": gguf.LlamaFileType.MOSTLY_Q4_0_64,
            "Q4_1_64": gguf.LlamaFileType.MOSTLY_Q4_1_64,
            "Q8_0_64": gguf.LlamaFileType.MOSTLY_Q8_0_64,
        }[self.block_format.name]

    @property
    def tensor_type(self) -> gguf.GGMLQuantizationType:
        return {
            "Q4_0_64": gguf.GGMLQuantizationType.Q4_0_64,
            "Q4_1_64": gguf.GGMLQuantizationType.Q4_1_64,
            "Q8_0_64": gguf.GGMLQuantizationType.Q8_0_64,
        }[self.block_format.name]

    def _lookup(
        self,
        source_name: str,
        gguf_name: str,
    ) -> tuple[str, dict[str, Any], bool] | None:
        for key, transform in ((gguf_name, False), (source_name, True)):
            if key in self._remaining:
                entry = self.tensors[key]
                if not isinstance(entry, dict):
                    raise ValueError(f"invalid GPTQ sidecar entry for {key}")
                entry_gguf_name = entry.get("gguf_name")
                entry_source_name = entry.get("source_name")
                if entry_gguf_name in (None, gguf_name) and entry_source_name in (
                    None,
                    source_name,
                ):
                    return key, entry, transform
        if isinstance(self.tensors, SidecarShardReader):
            return None

        matches: list[tuple[str, dict[str, Any], bool]] = []
        for key in self._remaining:
            entry = self.tensors[key]
            if not isinstance(entry, dict):
                raise ValueError(f"invalid GPTQ sidecar entry for {key}")
            entry_gguf_name = entry.get("gguf_name")
            entry_source_name = entry.get("source_name")
            if key == gguf_name:
                matches.append((key, entry, False))
            elif entry_gguf_name == gguf_name and entry_source_name in (
                None,
                source_name,
            ):
                matches.append(
                    (key, entry, entry_source_name == source_name)
                )
            elif key == source_name and entry_gguf_name in (None, gguf_name):
                matches.append((key, entry, True))
            elif entry_source_name == source_name and entry_gguf_name is None:
                matches.append((key, entry, True))

        if not matches:
            return None
        if len(matches) > 1:
            keys = ", ".join(sorted(match[0] for match in matches))
            raise ValueError(
                f"ambiguous GPTQ sidecar entries for {gguf_name}: {keys}"
            )
        return matches[0]

    @staticmethod
    def _reorder_v_heads(
        tensor: torch.Tensor,
        dim: int,
        num_k_heads: int,
        num_v_per_k: int,
        head_dim: int,
    ) -> torch.Tensor:
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        new_shape = (
            shape[:dim]
            + [num_k_heads, num_v_per_k, head_dim]
            + shape[dim + 1 :]
        )
        tensor = tensor.reshape(*new_shape)
        permutation = list(range(len(new_shape)))
        permutation[dim], permutation[dim + 1] = (
            permutation[dim + 1],
            permutation[dim],
        )
        return tensor.permute(*permutation).contiguous().reshape(*shape)

    def transform(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
        source_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "linear_attn." not in source_name:
            return codes, scales

        num_k_heads = self.hparams.get("linear_num_key_heads", 0)
        num_v_heads = self.hparams.get("linear_num_value_heads", 0)
        if num_k_heads <= 0 or num_v_heads <= 0 or num_k_heads == num_v_heads:
            return codes, scales

        head_k_dim = self.hparams["linear_key_head_dim"]
        head_v_dim = self.hparams["linear_value_head_dim"]
        num_v_per_k = num_v_heads // num_k_heads

        def reorder_rows(
            tensor: torch.Tensor, head_dim: int
        ) -> torch.Tensor:
            return self._reorder_v_heads(
                tensor, 0, num_k_heads, num_v_per_k, head_dim
            )

        if ".in_proj_qkv." in source_name:
            q_dim = head_k_dim * num_k_heads
            k_dim = head_k_dim * num_k_heads
            codes = torch.cat(
                (
                    codes[:q_dim],
                    codes[q_dim : q_dim + k_dim],
                    reorder_rows(codes[q_dim + k_dim :], head_v_dim),
                ),
                dim=0,
            )
            scales = torch.cat(
                (
                    scales[:q_dim],
                    scales[q_dim : q_dim + k_dim],
                    reorder_rows(scales[q_dim + k_dim :], head_v_dim),
                ),
                dim=0,
            )
        elif ".in_proj_z." in source_name:
            codes = reorder_rows(codes, head_v_dim)
            scales = reorder_rows(scales, head_v_dim)
        elif ".in_proj_b." in source_name or ".in_proj_a." in source_name:
            codes = reorder_rows(codes, 1)
            scales = reorder_rows(scales, 1)
        elif ".out_proj." in source_name:
            group_size = self.block_format.group_size
            if head_v_dim % group_size:
                raise ValueError(
                    f"{self.block_format.name} linear-attention V head "
                    f"dimension must be divisible by {group_size}"
                )
            codes = self._reorder_v_heads(
                codes, 1, num_k_heads, num_v_per_k, head_v_dim
            )
            scales = self._reorder_v_heads(
                scales,
                1,
                num_k_heads,
                num_v_per_k,
                head_v_dim // group_size,
            )
        return codes.contiguous(), scales.contiguous()

    def emit_if_present(
        self,
        *,
        writer: Any,
        source_name: str,
        gguf_name: str,
        weight: torch.Tensor,
        old_dtype: torch.dtype,
        max_name_len: int,
    ) -> bool:
        """Write a matching packed sidecar tensor and report whether consumed."""

        match = self._lookup(source_name, gguf_name)
        if match is None:
            return False
        entry_key, entry, transform_from_source = match
        packed = entry.get("packed")
        codes = entry.get("codes")
        scales = entry.get("scales")
        if not isinstance(packed, torch.Tensor):
            raise ValueError(
                f"invalid {self.block_format.name} sidecar entry for {entry_key}"
            )
        if not isinstance(codes, torch.Tensor) or not isinstance(
            scales, torch.Tensor
        ):
            logical_shape = entry.get("shape")
            if (
                not isinstance(logical_shape, list)
                or len(logical_shape) != packed.ndim
            ):
                raise ValueError(
                    f"packed-only {self.block_format.name} entry lacks shape "
                    f"for {entry_key}"
                )
            codes, scales = self.block_format.unpack(
                packed, logical_size=int(logical_shape[-1])
            )
            if tuple(codes.shape) != tuple(logical_shape):
                raise ValueError(
                    f"packed-only {self.block_format.name} shape mismatch "
                    f"for {entry_key}"
                )

        if transform_from_source:
            codes, scales = self.transform(codes, scales, source_name)
            packed = self.block_format.pack(codes, scales)
        if (
            codes.shape != weight.shape
            or codes.shape[-1] % self.block_format.group_size
        ):
            raise ValueError(
                f"{self.block_format.name} sidecar shape mismatch for "
                f"{entry_key}: codes={tuple(codes.shape)}, "
                f"weight={tuple(weight.shape)}"
            )

        n_blocks = codes.shape[-1] // self.block_format.group_size
        expected_scales = (*codes.shape[:-1], n_blocks)
        if self.block_format.parameter_count > 1:
            expected_scales += (self.block_format.parameter_count,)
        expected_packed = (
            *codes.shape[:-1],
            n_blocks * self.block_format.type_size,
        )
        if (
            tuple(scales.shape) != expected_scales
            or tuple(packed.shape) != expected_packed
        ):
            raise ValueError(
                f"invalid {self.block_format.name} scales/packed shape "
                f"for {entry_key}"
            )

        data = packed.to(torch.uint8).contiguous().numpy()
        shape = gguf.quant_shape_from_byte_shape(data.shape, self.tensor_type)
        shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"
        logger.info(
            f"{f'%-{max_name_len}s' % f'{gguf_name},'} "
            f"{old_dtype} --> {self.tensor_type.name} "
            f"(GPTQ sidecar), shape = {shape_str}"
        )
        writer.add_tensor(gguf_name, data, raw_dtype=self.tensor_type)
        self._remaining.remove(entry_key)
        return True

    def finish(self) -> None:
        if self._remaining:
            missing = ", ".join(sorted(self._remaining)[:8])
            raise ValueError(
                "GPTQ sidecar tensors were not found in the HF model: "
                f"{missing}"
            )
