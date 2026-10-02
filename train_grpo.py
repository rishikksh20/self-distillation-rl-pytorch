"""Minimal Group Relative Policy Optimization for causal LMs in plain PyTorch."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Any, Iterator

import torch

from self_distill_rl.io import read_jsonl, rollout_batch
from self_distill_rl.modeling import (
    default_device,
    load_policy,
    load_tokenizer,
    policy_logprobs,
    save_policy,
)
from self_distill_rl.objectives import grpo_loss, group_relative_advantages


def group_batches(
    records: list[dict[str, Any]], groups_per_batch: int, seed: int
) -> Iterator[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["group_id"])].append(record)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    for start in range(0, len(groups), groups_per_batch):
        yield [
            record
            for group in groups[start : start + groups_per_batch]
            for record in group
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--rollouts", default="artifacts/rollouts.jsonl")
    parser.add_argument("--output", default="artifacts/grpo_model")
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--policy-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    tokenizer = load_tokenizer(args.model)
    model = load_policy(
        args.model, device, gradient_checkpointing=args.gradient_checkpointing
    )
    model.train()
    reference = None
    if args.beta:
        reference = load_policy(args.reference_model or args.model, device)
        reference.eval()
        reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    records = read_jsonl(args.rollouts)

    for batch_index, rows in enumerate(
        group_batches(records, args.groups_per_batch, args.seed), start=1
    ):
        batch = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
        assert batch.old_logprobs is not None
        rewards = torch.tensor([float(row["reward"]) for row in rows], device=device)
        advantages = group_relative_advantages(
            rewards, [str(row["group_id"]) for row in rows]
        )
        with torch.no_grad():
            reference_logprobs = (
                None if reference is None else policy_logprobs(reference, batch)
            )

        for policy_epoch in range(args.policy_epochs):
            new_logprobs = policy_logprobs(model, batch)
            loss, metrics = grpo_loss(
                new_logprobs,
                batch.old_logprobs,
                advantages,
                batch.action_mask,
                args.clip_epsilon,
                args.beta,
                reference_logprobs,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            print(
                f"batch={batch_index} epoch={policy_epoch + 1} reward={rewards.mean().item():.3f} "
                f"grpo_loss={metrics['loss']:.4f} reference_kl={metrics['reference_kl']:.4f}"
            )

    save_policy(model, tokenizer, args.output)
    print(f"saved GRPO policy to {args.output}")


if __name__ == "__main__":
    main()
