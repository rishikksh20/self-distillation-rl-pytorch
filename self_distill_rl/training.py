"""Shared training controls, accumulation, and rollout provenance checks."""

import argparse
import math
from typing import Any

import torch

from .presets import PRESETS


def add_training_args(parser: argparse.ArgumentParser, algorithm: str) -> None:
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--max-seq-length", type=int, default=PRESETS[algorithm].max_seq_length
    )
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )


def validate_training_args(args: argparse.Namespace) -> None:
    for name in (
        "gradient_accumulation_steps",
        "max_seq_length",
        "logit_chunk_size",
        "batch_size",
        "groups_per_batch",
        "epochs",
        "ppo_epochs",
        "policy_epochs",
    ):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        not math.isfinite(args.learning_rate)
        or not math.isfinite(args.max_grad_norm)
        or args.learning_rate <= 0
        or args.max_grad_norm <= 0
    ):
        raise ValueError("learning rate and max gradient norm must be positive")
    for name in ("clip_epsilon", "value_clip_epsilon"):
        if hasattr(args, name) and not 0 < getattr(args, name) < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    for name in ("gamma", "gae_lambda"):
        if hasattr(args, name) and not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if hasattr(args, "beta") and (not math.isfinite(args.beta) or args.beta < 0):
        raise ValueError("beta must be finite and nonnegative")


def validate_rollouts(
    records: list[dict[str, Any]],
    model: str,
    *,
    policy_gradient: bool = False,
    max_seq_length: int,
) -> None:
    if not records:
        raise ValueError("rollout file is empty")
    for row in records:
        if row.get("split") == "test":
            raise ValueError("training rollouts contain held-out test examples")
        if row.get("model") is not None and row["model"] != model:
            raise ValueError(
                "rollouts came from a different checkpoint; regenerate with --model"
            )
        if (
            len(row["prompt_token_ids"]) + len(row["completion_token_ids"])
            > max_seq_length
        ):
            raise ValueError(
                "rollout exceeds --max-seq-length; regenerate with a shorter token budget"
            )
        if not math.isfinite(float(row["reward"])):
            raise ValueError("rollout reward must be finite")
        if policy_gradient:
            sampling = row.get("sampling", {})
            if (
                sampling.get("temperature", 1.0) != 1.0
                or sampling.get("top_p", 1.0) != 1.0
            ):
                raise ValueError(
                    "PPO/GRPO require untempered, untruncated rollouts (temperature=top-p=1)"
                )


class AccumulatingOptimizer:
    """Average microbatch gradients, including an incomplete final window."""

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        steps: int,
        max_grad_norm: float,
    ):
        self.model = model
        self.steps = steps
        self.max_grad_norm = max_grad_norm
        self.pending = 0
        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=0.0,
            fused=next(model.parameters()).device.type == "cuda",
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                next(model.parameters()).device.type == "cuda"
                and not torch.cuda.is_bf16_supported()
            ),
        )

    def backward(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        self.scaler.scale(loss / self.steps).backward()
        self.pending += 1
        if self.pending == self.steps:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        self.scaler.unscale_(self.optimizer)
        if self.pending != self.steps:
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(self.steps / self.pending)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
