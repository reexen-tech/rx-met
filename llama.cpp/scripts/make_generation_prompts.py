#!/usr/bin/env python3
"""
Create unified generation-prompt JSONL files from benchmark-style datasets.

The output is meant for generation/evaluation, not for llama-perplexity. Each
line has a stable schema:

{
  "id": "...",
  "dataset": "gsm8k",
  "task_type": "math",
  "prompt": "...",
  "answer": null,
  "source": "cache/gsm8k.jsonl",
  "metadata": {...}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


MATH_PROMPT = """You are a careful reasoning assistant.

Solve the following problem. Show your reasoning step by step.
Put your final answer in \\boxed{{}}.

Problem:
{question}
"""

CODE_PROMPT = """You are a Python programming assistant.

Write a correct Python solution for the following task.
Return only the code.

Task:
{question}
"""

GENERAL_PROMPT = """You are a helpful assistant.

User:
{question}

Assistant:
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no}: expected object, got {type(obj).__name__}")
            records.append(obj)
    return records


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(text(item) for item in value if text(item))
    return str(value).strip()


def question_from_record(record: dict[str, Any]) -> str:
    return text(first_present(record, ("question", "problem", "prompt", "input", "text", "turns")))


def answer_from_record(record: dict[str, Any]) -> str | None:
    answer = text(first_present(record, ("answer", "solution", "target", "output", "final_answer", "boxed_answer")))
    return answer or None


def make_math_record(dataset: str, record: dict[str, Any], idx: int, source: Path) -> dict[str, Any]:
    question = question_from_record(record)
    return {
        "id": str(first_present(record, ("id", "question_id", "task_id")) or f"{dataset}-{idx:05d}"),
        "dataset": dataset,
        "task_type": "math",
        "prompt": MATH_PROMPT.format(question=question).rstrip(),
        "answer": answer_from_record(record),
        "source": str(source),
        "metadata": {
            "answer_extraction": "prefer regex \\\\boxed\\{([^{}]+)\\}; fallback to Final answer / last number",
        },
    }


def make_code_record(dataset: str, record: dict[str, Any], idx: int, source: Path) -> dict[str, Any]:
    question = question_from_record(record)
    return {
        "id": str(first_present(record, ("task_id", "id", "question_id")) or f"{dataset}-{idx:05d}"),
        "dataset": dataset,
        "task_type": "code",
        "prompt": CODE_PROMPT.format(question=question).rstrip(),
        "answer": answer_from_record(record),
        "source": str(source),
        "metadata": {
            "answer_extraction": "treat full model output as code; strip markdown fences if present",
        },
    }


def make_mt_bench_records(dataset: str, record: dict[str, Any], idx: int, source: Path) -> list[dict[str, Any]]:
    turns = first_present(record, ("turns", "messages", "question", "prompt"))
    if not isinstance(turns, list):
        turns = [text(turns)]

    base_id = str(first_present(record, ("question_id", "id")) or f"{dataset}-{idx:05d}")
    category = first_present(record, ("category", "domain"))
    out: list[dict[str, Any]] = []
    for turn_idx, turn in enumerate(turns, start=1):
        prompt = GENERAL_PROMPT.format(question=text(turn)).rstrip()
        out.append(
            {
                "id": f"{base_id}-turn{turn_idx}",
                "dataset": dataset,
                "task_type": "chat",
                "prompt": prompt,
                "answer": None,
                "source": str(source),
                "metadata": {
                    "question_id": base_id,
                    "turn_index": turn_idx,
                    "category": category,
                    "all_turns": turns,
                    "note": "True MT-Bench second turn should be run after feeding turn-1 model output.",
                },
            }
        )
    return out


def make_general_record(dataset: str, record: dict[str, Any], idx: int, source: Path) -> dict[str, Any]:
    question = question_from_record(record)
    return {
        "id": str(first_present(record, ("id", "question_id", "task_id")) or f"{dataset}-{idx:05d}"),
        "dataset": dataset,
        "task_type": "general",
        "prompt": GENERAL_PROMPT.format(question=question).rstrip(),
        "answer": answer_from_record(record),
        "source": str(source),
        "metadata": {},
    }


def infer_dataset_type(name: str) -> str:
    lowered = name.lower()
    if "gsm8k" in lowered:
        return "gsm8k"
    if "math500" in lowered or "math-500" in lowered:
        return "math500"
    if "mt-bench" in lowered or "mt_bench" in lowered:
        return "mt-bench"
    if "humaneval" in lowered:
        return "humaneval"
    if "mbpp" in lowered:
        return "mbpp"
    return "auto"


def make_records(input_path: Path, dataset: str) -> list[dict[str, Any]]:
    records = load_jsonl(input_path)
    output: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        if dataset in {"gsm8k", "math500"}:
            output.append(make_math_record(dataset, record, idx, input_path))
        elif dataset in {"humaneval", "mbpp"}:
            output.append(make_code_record(dataset, record, idx, input_path))
        elif dataset == "mt-bench":
            output.extend(make_mt_bench_records(dataset, record, idx, input_path))
        else:
            output.append(make_general_record(dataset, record, idx, input_path))
    return output


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prompts to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output generation JSONL")
    parser.add_argument(
        "--dataset",
        default="auto",
        choices=("auto", "gsm8k", "math500", "mt-bench", "humaneval", "mbpp"),
        help="Prompt template to use",
    )
    args = parser.parse_args()

    dataset = infer_dataset_type(args.input.name) if args.dataset == "auto" else args.dataset
    records = make_records(args.input, dataset)
    write_jsonl(records, args.output)


if __name__ == "__main__":
    main()
