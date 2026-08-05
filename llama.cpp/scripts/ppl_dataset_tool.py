#!/usr/bin/env python3
"""
Inspect and convert datasets for llama-perplexity plain-text input.

llama-perplexity reads the file passed by `-f` as one continuous text stream,
then tokenizes it and slices it into ctx-sized chunks. WikiText raw files are
just plain text, so task datasets need to be rendered into text first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


TEXT_SUFFIXES = {".raw", ".txt", ".text", ".md"}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
PARQUET_SUFFIXES = {".parquet"}


def iter_input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    files: list[Path] = []
    for suffix in sorted(TEXT_SUFFIXES | JSON_SUFFIXES | PARQUET_SUFFIXES):
        files.extend(sorted(path.rglob(f"*{suffix}")))
    return files


def load_records(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return [{"text": path.read_text(encoding="utf-8", errors="replace")}]

    if suffix == ".jsonl" or suffix == ".ndjson":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "records", "questions", "test"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
        raise TypeError(f"Unsupported JSON top-level type: {type(data).__name__}")

    if suffix in PARQUET_SUFFIXES:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pandas/pyarrow") from exc
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError(f"Unsupported file suffix: {path.suffix}")


def compact(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
    text = text.replace("\r\n", "\n")
    return text if len(text) <= limit else text[:limit] + " ..."


def inspect_path(path: Path, samples: int) -> None:
    files = iter_input_files(path)
    if not files:
        raise FileNotFoundError(f"No supported files under {path}")

    print(f"Input: {path}")
    print(f"Files: {len(files)}")
    for file in files:
        suffix = file.suffix.lower()
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"\n== {file} ==")
        print(f"suffix={suffix or '<none>'} size={size_mb:.2f} MiB")

        if suffix in TEXT_SUFFIXES:
            text = file.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            non_empty = [line for line in lines if line.strip()]
            print(f"format=plain_text chars={len(text)} lines={len(lines)} non_empty_lines={len(non_empty)}")
            print("sample:")
            print(compact("\n".join(lines[:samples]), 800))
            continue

        records = load_records(file)
        print(f"format=records records={len(records)}")
        if records:
            first = records[0]
            if isinstance(first, dict):
                print(f"fields={list(first.keys())}")
            for i, record in enumerate(records[:samples], start=1):
                print(f"\nrecord[{i}]:")
                print(compact(record, 1000))


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value if as_text(item))
    return str(value).strip()


def render_gsm8k(record: dict[str, Any]) -> str:
    question = as_text(first_present(record, ("question", "prompt", "input", "turns")))
    answer = as_text(first_present(record, ("answer", "solution", "target", "output")))
    if not answer:
        return f"Question:\n{question}"
    return f"Question:\n{question}\n\nAnswer:\n{answer}"


def render_math500(record: dict[str, Any]) -> str:
    problem = as_text(first_present(record, ("problem", "question", "prompt", "input", "turns")))
    solution = as_text(first_present(record, ("solution", "answer", "target", "output")))
    final_answer = as_text(first_present(record, ("answer", "final_answer", "boxed_answer")))
    if not solution and not final_answer:
        return f"Problem:\n{problem}"
    if final_answer and final_answer not in solution:
        solution = f"{solution}\n\nFinal answer: {final_answer}".strip()
    return f"Problem:\n{problem}\n\nSolution:\n{solution}"


def render_math500_prompt(record: dict[str, Any]) -> str:
    problem = as_text(first_present(record, ("problem", "question", "prompt", "input")))
    return problem


def render_mbpp_prompt(record: dict[str, Any]) -> str:
    return as_text(first_present(record, ("text", "prompt", "question", "input")))


def render_mt_bench(record: dict[str, Any]) -> str:
    question_id = first_present(record, ("question_id", "id"))
    category = first_present(record, ("category", "domain"))
    turns = first_present(record, ("turns", "question", "prompt", "messages"))

    parts: list[str] = []
    if question_id is not None:
        parts.append(f"Question ID: {question_id}")
    if category is not None:
        parts.append(f"Category: {category}")

    if isinstance(turns, list):
        for i, turn in enumerate(turns, start=1):
            if isinstance(turn, dict):
                role = turn.get("role", f"Turn {i}")
                content = first_present(turn, ("content", "text", "value"))
                parts.append(f"{role}:\n{as_text(content)}")
            else:
                parts.append(f"Turn {i}:\n{as_text(turn)}")
    else:
        parts.append(f"Prompt:\n{as_text(turns)}")

    return "\n\n".join(part for part in parts if part.strip())


def render_auto(record: dict[str, Any]) -> str:
    question = first_present(record, ("question", "problem", "prompt", "input", "text", "turns"))
    answer = first_present(record, ("answer", "solution", "target", "output", "response"))
    if question is not None and answer is not None:
        return f"Question:\n{as_text(question)}\n\nAnswer:\n{as_text(answer)}"
    if question is not None:
        return as_text(question)
    return json.dumps(record, ensure_ascii=False)


def render_record(record: Any, dataset_type: str) -> str:
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return json.dumps(record, ensure_ascii=False)

    if dataset_type == "gsm8k":
        return render_gsm8k(record)
    if dataset_type == "math500":
        return render_math500(record)
    if dataset_type == "math500-prompt":
        return render_math500_prompt(record)
    if dataset_type == "mbpp-prompt":
        return render_mbpp_prompt(record)
    if dataset_type in {"mt-bench", "mt_batch", "mt-batch"}:
        return render_mt_bench(record)
    if dataset_type == "plain":
        return as_text(first_present(record, ("text", "content", "prompt", "question")))
    return render_auto(record)


def convert_path(path: Path, output: Path, dataset_type: str, limit: int | None, separator: str) -> None:
    files = iter_input_files(path)
    if not files:
        raise FileNotFoundError(f"No supported files under {path}")

    rendered: list[str] = []
    for file in files:
        for record in load_records(file):
            text = render_record(record, dataset_type).strip()
            if text:
                rendered.append(text)
                if limit is not None and len(rendered) >= limit:
                    break
        if limit is not None and len(rendered) >= limit:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(separator.join(rendered).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(rendered)} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect raw/json/jsonl/parquet dataset files")
    p_inspect.add_argument("path", type=Path)
    p_inspect.add_argument("--samples", type=int, default=3)

    p_convert = sub.add_parser("convert", help="Render dataset records into llama-perplexity plain text")
    p_convert.add_argument("input", type=Path)
    p_convert.add_argument("-o", "--output", type=Path, required=True)
    p_convert.add_argument(
        "--type",
        choices=(
            "auto",
            "plain",
            "gsm8k",
            "math500",
            "math500-prompt",
            "mbpp-prompt",
            "mt-bench",
            "mt_batch",
            "mt-batch",
        ),
        default="auto",
        help="Dataset rendering template",
    )
    p_convert.add_argument("--limit", type=int, default=None)
    p_convert.add_argument("--separator", default="\n\n\n")

    args = parser.parse_args()
    if args.command == "inspect":
        inspect_path(args.path, args.samples)
    elif args.command == "convert":
        convert_path(args.input, args.output, args.type, args.limit, args.separator)


if __name__ == "__main__":
    main()
