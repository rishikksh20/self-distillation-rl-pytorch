"""Tiny, auditable reward functions for the included arithmetic examples."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_ANSWER = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER = re.compile(r"final answer\s*(?:is|:)?\s*([^\n]+)", re.IGNORECASE)
_NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/[-+]?\d+(?:\.\d+)?)?"
)


def _numeric_fraction(text: str) -> Fraction:
    value = Decimal(text)
    if not value.is_finite():
        raise ValueError("nonfinite answer")
    digits = value.as_tuple()
    if len(digits.digits) > 1000 or abs(digits.exponent) > 1000:
        raise ValueError("numeric answer is too large to normalize")
    return Fraction(value)


def normalize_answer(answer: Any) -> str:
    text = str(answer).strip().lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(". \n")
    try:
        if "/" in text:
            numerator, denominator = text.split("/")
            value = _numeric_fraction(numerator) / _numeric_fraction(denominator)
        else:
            value = _numeric_fraction(text)
        return str(value)
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return text


def extract_final_answer(text: str) -> str:
    """Extract common math-answer formats, falling back to the final number or line."""
    # Ignore private reasoning when the model emits a thinking block.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif "<think>" in text:
        return ""
    text = text.replace(",", "")
    boxed = _BOXED.findall(text)
    if boxed:
        return normalize_answer(boxed[-1])
    hashes = _HASH_ANSWER.findall(text)
    if hashes:
        return normalize_answer(hashes[-1])
    finals = _FINAL_ANSWER.findall(text)
    if finals:
        final = finals[-1].strip()
        number = _NUMBER.match(final)
        return normalize_answer(number.group() if number else final)
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
