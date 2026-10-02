"""Model loading and token-level operations written directly with PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import CausalBatch, ensure_padding_token


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    ensure_padding_token(tokenizer)
    tokenizer.padding_side = "right"
    return tokenizer


def load_policy(
    model_name_or_path: str,
    device: torch.device,
    *,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=training_dtype(device),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.to(device)


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Return log p(x[t+1] | x[:t+1]) with shape [batch, sequence-1]."""
    next_token_logits = logits[:, :-1, :]
    next_tokens = input_ids[:, 1:].unsqueeze(-1)
    return (
        F.log_softmax(next_token_logits.float(), dim=-1)
        .gather(-1, next_tokens)
        .squeeze(-1)
    )


def policy_logprobs(model: nn.Module, batch: CausalBatch) -> torch.Tensor:
    output = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    return token_logprobs(output.logits, batch.input_ids)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def completion_logits(logits: torch.Tensor, batch: CausalBatch) -> list[torch.Tensor]:
    """Slice next-token logits into one [completion_length, vocab] tensor per sample."""
    rows: list[torch.Tensor] = []
    for row, (prompt_length, completion_length) in enumerate(
        zip(batch.prompt_lengths, batch.completion_lengths)
    ):
        start = prompt_length - 1
        rows.append(logits[row, start : start + completion_length, :].float())
    return rows


def reverse_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token KL(student || stop-gradient teacher)."""
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach(), dim=-1)
    student_p = student_logp.exp()
    return (student_p * (student_logp - teacher_logp)).sum(dim=-1)


def jensen_shannon(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token symmetric JSD with equal mixture weights."""
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach(), dim=-1)
    log_mixture = torch.logaddexp(student_logp, teacher_logp) - 0.6931471805599453
    student_term = (student_logp.exp() * (student_logp - log_mixture)).sum(dim=-1)
    teacher_term = (teacher_logp.exp() * (teacher_logp - log_mixture)).sum(dim=-1)
    return 0.5 * (student_term + teacher_term)


def distillation_loss(
    student_rows: Sequence[torch.Tensor],
    teacher_rows: Sequence[torch.Tensor],
    divergence: str,
) -> torch.Tensor:
    if len(student_rows) != len(teacher_rows):
        raise ValueError("student and teacher batch sizes differ")
    losses: list[torch.Tensor] = []
    for student, teacher in zip(student_rows, teacher_rows):
        if student.shape != teacher.shape:
            raise ValueError(
                f"unaligned distillation logits: {student.shape} vs {teacher.shape}"
            )
        if divergence == "reverse_kl":
            losses.append(reverse_kl(student, teacher).mean())
        elif divergence == "jsd":
            losses.append(jensen_shannon(student, teacher).mean())
        else:
            raise ValueError(f"unknown divergence: {divergence}")
    return torch.stack(losses).mean()


class ActorCritic(nn.Module):
    """A causal LM actor with one scalar value prediction per token state."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        hidden_size = getattr(policy.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = policy.config.text_config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False, dtype=torch.float32)

    def forward(self, batch: CausalBatch) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.policy(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            output_hidden_states=True,
        )
        logprobs = token_logprobs(output.logits, batch.input_ids)
        values = self.value_head(output.hidden_states[-1][:, :-1, :].float()).squeeze(
            -1
        )
        return logprobs, values

    def load_value_head(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint) / "value_head.pt"
        if path.exists():
            self.value_head.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )

    def save(self, output_dir: str | Path, tokenizer: Any) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(path)
        tokenizer.save_pretrained(path)
        torch.save(self.value_head.state_dict(), path / "value_head.pt")


def save_policy(model: nn.Module, tokenizer: Any, output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
