"""Model loading and token-level operations written directly with PyTorch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

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
    attention_implementation: str = "sdpa",
    trainable: bool = True,
) -> nn.Module:
    config = AutoConfig.from_pretrained(model_name_or_path)
    loader = (
        AutoModelForImageTextToText
        if config.model_type == "qwen3_5"
        else AutoModelForCausalLM
    )
    model = loader.from_pretrained(
        model_name_or_path,
        config=config,
        # Adam needs FP32 master parameters at the small RL learning rates. BF16
        # parameter updates can otherwise round to zero. Autocast keeps compute cheap.
        dtype=torch.float32 if trainable else training_dtype(device),
        attn_implementation=attention_implementation,
    )
    model.config.use_cache = False
    if not trainable:
        model.requires_grad_(False)
    # Preserve the native checkpoint layout for vLLM; text training skips the vision tower.
    if config.model_type == "qwen3_5":
        model.config.text_config.use_cache = False
        model.model.visual.requires_grad_(False)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
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


def text_backbone(model: nn.Module) -> nn.Module:
    if model.config.model_type == "qwen3_5":
        return model.model.language_model
    return model.base_model


def batch_hidden_states(model: nn.Module, batch: CausalBatch) -> torch.Tensor:
    with compute_autocast(model):
        return text_backbone(model)(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state


def compute_autocast(model: nn.Module):
    device = next(model.parameters()).device
    return torch.autocast(
        device_type=device.type,
        dtype=training_dtype(device),
        enabled=device.type == "cuda",
    )


def project_logits(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    with compute_autocast(model):
        return model.get_output_embeddings()(states)


def completion_hidden_states(
    model: nn.Module, batch: CausalBatch
) -> list[torch.Tensor]:
    hidden = batch_hidden_states(model, batch)
    return [
        hidden[i, p - 1 : p - 1 + c]
        for i, (p, c) in enumerate(zip(batch.prompt_lengths, batch.completion_lengths))
    ]


def hidden_logprobs(
    model: nn.Module, hidden: torch.Tensor, batch: CausalBatch, chunk_size: int = 32
) -> torch.Tensor:
    """Project only action states; recompute chunk softmaxes on backward."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    action_hidden = hidden[:, :-1][batch.action_mask]
    labels = batch.input_ids[:, 1:][batch.action_mask]

    def project(states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = project_logits(model, states).float()
        return -F.cross_entropy(logits, tokens, reduction="none")

    chunks = []
    for start in range(0, labels.numel(), chunk_size):
        states, tokens = (
            action_hidden[start : start + chunk_size],
            labels[start : start + chunk_size],
        )
        chunks.append(
            checkpoint(project, states, tokens, use_reentrant=False)
            if torch.is_grad_enabled()
            else project(states, tokens)
        )
    result = torch.zeros_like(batch.action_mask, dtype=torch.float32)
    return result.masked_scatter(batch.action_mask, torch.cat(chunks))


def policy_logprobs(
    model: nn.Module, batch: CausalBatch, chunk_size: int = 32
) -> torch.Tensor:
    return hidden_logprobs(model, batch_hidden_states(model, batch), batch, chunk_size)


def hidden_distillation_loss(
    student: nn.Module,
    teacher: nn.Module,
    student_rows: Sequence[torch.Tensor],
    teacher_rows: Sequence[torch.Tensor],
    divergence: str,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Exact full-vocabulary divergence with bounded projection/softmax memory."""
    if chunk_size < 1 or not student_rows or len(student_rows) != len(teacher_rows):
        raise ValueError("invalid distillation chunk size or batch")
    if divergence not in ("jsd", "reverse_kl"):
        raise ValueError(f"unknown divergence: {divergence}")
    divergence_fn = jensen_shannon if divergence == "jsd" else reverse_kl

    def project(states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # The live SDPO teacher head also needs stop-gradient, even when heads are tied.
        with torch.no_grad():
            teacher_logits = project_logits(teacher, targets).float()
        student_logits = project_logits(student, states).float()
        return divergence_fn(student_logits, teacher_logits).sum()

    losses = []
    for states, targets in zip(student_rows, teacher_rows):
        if states.shape[0] != targets.shape[0] or states.shape[0] == 0:
            raise ValueError("unaligned or empty distillation continuations")
        chunks = [
            checkpoint(
                project,
                states[i : i + chunk_size],
                targets[i : i + chunk_size],
                use_reentrant=False,
            )
            for i in range(0, states.shape[0], chunk_size)
        ]
        losses.append(torch.stack(chunks).sum() / states.shape[0])
    return torch.stack(losses).mean()


def require_matching_tokenizers(student_tokenizer: Any, teacher_name: str) -> None:
    teacher_tokenizer = load_tokenizer(teacher_name)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "teacher/reference and student must share the same token-to-ID vocabulary"
        )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def completion_logits(logits: torch.Tensor, batch: CausalBatch) -> list[torch.Tensor]:
    """Slice next-token logits into one [completion_length, vocab] tensor per sample."""
    rows: list[torch.Tensor] = []
    for row, (prompt_length, completion_length) in enumerate(
        zip(batch.prompt_lengths, batch.completion_lengths)
    ):
        start = prompt_length - 1
        rows.append(logits[row, start : start + completion_length, :])
    return rows


def reverse_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token KL(student || stop-gradient teacher)."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_p = student_logp.exp()
    return (student_p * (student_logp - teacher_logp)).sum(dim=-1)


def jensen_shannon(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Per-token symmetric JSD with equal mixture weights."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
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

    def predict_values(self, batch: CausalBatch) -> torch.Tensor:
        hidden = batch_hidden_states(self.policy, batch)
        return self.value_head(hidden[:, :-1, :].float()).squeeze(-1)

    def forward(
        self, batch: CausalBatch, chunk_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = batch_hidden_states(self.policy, batch)
        logprobs = hidden_logprobs(self.policy, hidden, batch, chunk_size)
        values = self.value_head(hidden[:, :-1, :].float()).squeeze(-1)
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
