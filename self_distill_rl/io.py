"""JSONL, tokenization, and batching helpers shared by the tutorials."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Show concise reasoning, then give the final answer."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def shuffled_batches(
    items: Sequence[Any], batch_size: int, seed: int
) -> Iterator[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[index] for index in order[start : start + batch_size]]


def chat_prompt_ids(
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    *,
    enable_thinking: bool = False,
) -> list[int]:
    """Render one user turn, with a fallback for base models without a chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if hasattr(rendered, "keys"):
            rendered = rendered["input_ids"]
        return list(rendered)
    text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    return list(tokenizer.encode(text, add_special_tokens=True))


def response_ids(tokenizer: Any, text: str, add_eos: bool = True) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if add_eos and eos is not None and (not ids or ids[-1] != eos):
        ids.append(int(eos))
    return ids


@dataclass
class CausalBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    prompt_lengths: list[int]
    completion_lengths: list[int]
    old_logprobs: torch.Tensor | None = None

    def to(self, device: torch.device) -> CausalBatch:
        return CausalBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            prompt_lengths=self.prompt_lengths,
            completion_lengths=self.completion_lengths,
            old_logprobs=None
            if self.old_logprobs is None
            else self.old_logprobs.to(device),
        )


def causal_batch(
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    pad_token_id: int,
    old_token_logprobs: Sequence[Sequence[float]] | None = None,
) -> CausalBatch:
    """Right-pad prompt+completion sequences and mark only completion predictions."""
    if len(prompt_token_ids) != len(completion_token_ids):
        raise ValueError("prompt and completion batch sizes differ")
    if not prompt_token_ids:
        raise ValueError("cannot build an empty batch")
    if old_token_logprobs is not None and len(old_token_logprobs) != len(
        prompt_token_ids
    ):
        raise ValueError("old log-probability and sample batch sizes differ")

    prompt_lengths = [len(ids) for ids in prompt_token_ids]
    completion_lengths = [len(ids) for ids in completion_token_ids]
    if min(prompt_lengths) < 1 or min(completion_lengths) < 1:
        raise ValueError("every sample needs at least one prompt and completion token")
    max_length = max(p + c for p, c in zip(prompt_lengths, completion_lengths))

    input_ids = torch.full(
        (len(prompt_lengths), max_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    action_mask = torch.zeros((len(prompt_lengths), max_length - 1), dtype=torch.bool)
    old_logprobs = None
    if old_token_logprobs is not None:
        old_logprobs = torch.zeros(
            (len(prompt_lengths), max_length - 1), dtype=torch.float32
        )

    for row, (prompt_ids, completion_ids) in enumerate(
        zip(prompt_token_ids, completion_token_ids)
    ):
        sequence = list(prompt_ids) + list(completion_ids)
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[row, : len(sequence)] = True
        # logits[:, position] predicts input_ids[:, position + 1].
        start = len(prompt_ids) - 1
        stop = start + len(completion_ids)
        action_mask[row, start:stop] = True
        if old_logprobs is not None:
            row_logprobs = list(old_token_logprobs[row])
            if len(row_logprobs) != len(completion_ids):
                raise ValueError(
                    "one old_logprobs list does not match its completion length"
                )
            if not all(math.isfinite(value) for value in row_logprobs):
                raise ValueError("old log-probabilities must be finite")
            old_logprobs[row, start:stop] = torch.tensor(
                row_logprobs, dtype=torch.float32
            )

    return CausalBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_mask=action_mask,
        prompt_lengths=prompt_lengths,
        completion_lengths=completion_lengths,
        old_logprobs=old_logprobs,
    )


def rollout_batch(records: Sequence[dict[str, Any]], pad_token_id: int) -> CausalBatch:
    return causal_batch(
        [record["prompt_token_ids"] for record in records],
        [record["completion_token_ids"] for record in records],
        pad_token_id,
        [record["old_logprobs"] for record in records],
    )


def sft_batch(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    target_key: str,
    max_seq_length: int | None = None,
) -> CausalBatch:
    prompts = [chat_prompt_ids(tokenizer, record["prompt"]) for record in records]
    completions = [response_ids(tokenizer, record[target_key]) for record in records]
    if max_seq_length is not None and any(
        len(p) + len(c) > max_seq_length for p, c in zip(prompts, completions)
    ):
        raise ValueError(
            "SDFT example exceeds --max-seq-length; increase the limit or filter long examples"
        )
    return causal_batch(prompts, completions, int(tokenizer.pad_token_id))


def ensure_padding_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
