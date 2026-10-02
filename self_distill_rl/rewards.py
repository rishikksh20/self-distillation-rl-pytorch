"""Tiny, auditable reward functions for the included arithmetic examples."""

from __future__ import annotations

import re
from typing import Any


_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER = re.compile(r"final answer\s*(?:is|:)?\s*([^\n.]+)", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def normalize_answer(answer: Any) -> str:
    text = str(answer).strip().lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(". \n")


def extract_final_answer(text: str) -> str:
    """Extract common math-answer formats, falling back to the final number or line."""
    boxed = _BOXED.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])
    hashes = _HASH_ANSWER.findall(text)
    if hashes:
        return normalize_answer(hashes[-1])
    finals = _FINAL_ANSWER.findall(text)
    if finals:
        return normalize_answer(finals[-1])
    numbers = _NUMBER.findall(text)
    if numbers:
        return normalize_answer(numbers[-1])
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return normalize_answer(nonempty_lines[-1] if nonempty_lines else text)


def exact_match_reward(completion: str, expected_answer: Any) -> float:
    return float(extract_final_answer(completion) == normalize_answer(expected_answer))


def environment_feedback(completion: str, expected_answer: Any) -> str:
    predicted = extract_final_answer(completion)
    expected = normalize_answer(expected_answer)
    if predicted == expected:
        return (
            "The answer passed the verifier. Preserve the correct reasoning and result."
        )
    return (
        f"The answer failed the verifier. It ended with {predicted!r}, but the verified "
        f"answer is {expected!r}. Identify the earliest mistake and reason toward the verified answer."
    )
