"""Context construction for privileged-information self-distillation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .io import CausalBatch, causal_batch, chat_prompt_ids

OPSD_SYSTEM = (
    "You are a solution-aware teacher. Use the private verified solution to evaluate how a "
    "student should reason, but score the student's existing continuation token by token."
)

SDPO_SYSTEM = (
    "You are a retrospective teacher. Use the environment feedback to judge and improve the "
    "attempt as it unfolds. Score the existing continuation rather than writing a separate answer."
)


def opsd_teacher_prompt(record: dict[str, Any]) -> str:
    return (
        f"Problem:\n{record['prompt']}\n\n"
        f"Private verified solution:\n{record['reference']}\n\n"
        "Re-evaluate the student's attempted reasoning with this privileged information."
    )


def sdpo_teacher_prompt(record: dict[str, Any]) -> str:
    return (
        f"Problem:\n{record['prompt']}\n\n"
        f"Feedback received after the attempt:\n{record['feedback']}\n\n"
        "Re-evaluate the attempted reasoning in hindsight."
    )


def teacher_batch(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    method: str,
    max_seq_length: int | None = None,
) -> CausalBatch:
    if method == "opsd":
        contexts = [
            chat_prompt_ids(tokenizer, opsd_teacher_prompt(row), OPSD_SYSTEM)
            for row in records
        ]
    elif method == "sdpo":
        contexts = [
            chat_prompt_ids(tokenizer, sdpo_teacher_prompt(row), SDPO_SYSTEM)
            for row in records
        ]
    else:
        raise ValueError(f"unknown teacher context method: {method}")
    completions = [row["completion_token_ids"] for row in records]
    if max_seq_length is not None and any(
        len(p) + len(c) > max_seq_length for p, c in zip(contexts, completions)
    ):
        raise ValueError(
            "privileged teacher context exceeds --max-seq-length; increase the limit or shorten the data"
        )
    return causal_batch(contexts, completions, int(tokenizer.pad_token_id))
