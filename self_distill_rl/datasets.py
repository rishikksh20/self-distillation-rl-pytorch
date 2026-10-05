"""Normalize standard Hugging Face math datasets without mixing their splits."""

from __future__ import annotations

from typing import Any

from .rewards import extract_final_answer, normalize_answer

DATASETS = {
    "gsm8k": ("openai/gsm8k", "main"),
    "svamp": ("ChilleD/SVAMP", "default"),
}
ANSWER_INSTRUCTION = "Show concise reasoning. End with 'Final answer: <number>'."


def normalize_example(
    name: str, row: dict[str, Any], split: str, index: int
) -> dict[str, Any]:
    if name == "gsm8k":
        question = row["question"].strip()
        reference = row["answer"].strip()
        if "####" not in reference:
            raise ValueError("GSM8K reference is missing its final-answer delimiter")
        answer = extract_final_answer(reference)
    elif name == "svamp":
        question = f"{row['Body'].strip()} {row['Question'].strip()}"
        answer = normalize_answer(row["Answer"])
        reference = f"Compute {row['Equation']}. Final answer: {answer}"
    else:
        raise ValueError(f"unknown dataset: {name}")
    if not question or not answer:
        raise ValueError(f"empty question or answer in {name}/{split}/{index}")
    return {
        "id": f"{name}:{split}:{row.get('ID', index)}",
        "prompt": f"{question}\n\n{ANSWER_INSTRUCTION}",
        "answer": answer,
        "reference": reference,
        "dataset": DATASETS[name][0],
        "split": split,
    }


def load_examples(
    name: str,
    split: str,
    limit: int | None = None,
    seed: int = 7,
    revision: str = "main",
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if split not in ("train", "test"):
        raise ValueError("use the dataset's official train or test split")
    if limit is not None and limit < 1:
        raise ValueError("dataset limit must be positive")
    path, config = DATASETS[name]
    data = load_dataset(path, config, split=split, revision=revision)
    # Assign stable IDs before shuffling/subsetting, so changing a limit preserves identity.
    records = [
        normalize_example(name, dict(row), split, i) for i, row in enumerate(data)
    ]
    if split == "train":
        import random

        random.Random(seed).shuffle(records)
    return records if limit is None else records[:limit]
