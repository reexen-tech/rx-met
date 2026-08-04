from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from llm_quant.artifacts import build_source_signature
from llm_quant.gptq.q4_0_64 import quantize_q4_0_64
from llm_quant.gptq.q8_0_64 import quantize_q8_0_64
from llm_quant.gptq.sidecar import write_sidecar_v3
from validate_gptq_gguf import validate_gguf, validate_sidecar

import convert_hf_to_gguf as converter
import gguf

def _write_tiny_qwen2(
    path: Path,
    *,
    sidecar_version: int = 3,
    bits: int = 4,
) -> None:
    vocabulary = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
    for token in ByteLevel.alphabet():
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    config = Qwen2Config(
        vocab_size=len(vocabulary),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)

    backend = Tokenizer(BPE(vocabulary, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(path)

    tensors: dict[str, dict[str, torch.Tensor]] = {}
    quantize = quantize_q4_0_64 if bits == 4 else quantize_q8_0_64
    format_name = f"Q{bits}_0_64"
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.layers."):
            continue
        result = quantize(module.weight.detach())
        tensors[f"{name}.weight"] = {
            "codes": result.codes.cpu(),
            "scales": result.scales.cpu(),
            "packed": result.packed.cpu(),
        }
    if sidecar_version != 3:
        torch.save(
            {
                "format": format_name,
                "version": sidecar_version,
                "tensors": tensors,
            },
            path / f"gptq_{format_name.lower()}.pt",
        )
    else:
        write_sidecar_v3(
            path,
            tensors,
            source_model=str(path),
            source_signature=build_source_signature(path),
            quantization_config={
                "algorithm": "gptq",
                "format": format_name,
            },
            format_name=format_name,
        )


@pytest.mark.parametrize("bits", [4, 8])
def test_sidecar_is_written_directly_to_gguf(
    tmp_path: Path, monkeypatch, bits: int
) -> None:
    model_path = tmp_path / "tiny-qwen2"
    model_path.mkdir()
    _write_tiny_qwen2(model_path, bits=bits)
    gguf_path = tmp_path / "tiny-qwen2-gptq.gguf"

    def set_test_vocab(instance: converter.Qwen2Model) -> None:
        instance.gguf_writer.add_tokenizer_model("gpt2")
        instance.gguf_writer.add_tokenizer_pre("qwen2")
        instance.gguf_writer.add_token_list(["<unk>", "<bos>", "<eos>"])

    monkeypatch.setattr(converter.Qwen2Model, "set_vocab", set_test_vocab)
    model = converter.Qwen2Model(
        model_path,
        gguf.LlamaFileType.MOSTLY_F16,
        gguf_path,
        eager=True,
    )
    model.set_vocab()
    model.write()
    expected_ftype = (
        gguf.LlamaFileType.MOSTLY_Q4_0_64
        if bits == 4
        else gguf.LlamaFileType.MOSTLY_Q8_0_64
    )
    assert model.ftype == expected_ftype
    tensors = validate_sidecar(model_path / f"gptq_q{bits}_0_64.pt")
    validate_gguf(tensors, gguf_path)


def test_legacy_sidecar_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny-qwen2-v2"
    model_path.mkdir()
    _write_tiny_qwen2(model_path, sidecar_version=2)
    gguf_path = tmp_path / "tiny-qwen2-v2-gptq.gguf"
    try:
        converter.Qwen2Model(
            model_path,
            gguf.LlamaFileType.MOSTLY_F16,
            gguf_path,
            eager=True,
        )
    except ValueError as exc:
        assert "expected version 3" in str(exc)
    else:
        raise AssertionError("legacy sidecar was accepted")


def test_multiple_sidecar_formats_are_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny-qwen2-multiple"
    model_path.mkdir()
    _write_tiny_qwen2(model_path, bits=4)
    _write_tiny_qwen2(model_path, bits=8)
    with pytest.raises(ValueError, match="multiple GPTQ sidecars"):
        converter.Qwen2Model(
            model_path,
            gguf.LlamaFileType.MOSTLY_F16,
            tmp_path / "unused.gguf",
            eager=True,
        )
