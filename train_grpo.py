"""Minimal Group Relative Policy Optimization for causal LMs in plain PyTorch."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch

from self_distill_rl.io import read_jsonl, rollout_batch
from self_distill_rl.modeling import (
    default_device,
    load_policy,
    load_tokenizer,
    policy_logprobs,
    require_matching_tokenizers,
    save_policy,
)
from self_distill_rl.objectives import group_relative_advantages, grpo_loss
from self_distill_rl.presets import PRESETS
from self_distill_rl.training import (
    AccumulatingOptimizer,
    add_training_args,
    validate_rollouts,
    validate_training_args,
)


def group_batches(
    records: list[dict[str, Any]], groups_per_batch: int, seed: int
) -> Iterator[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if groups_per_batch < 1:
        raise ValueError("groups_per_batch must be positive")
    for record in records:
        grouped[str(record["group_id"])].append(record)
    groups = list(grouped.values())
    for group in groups:
        if len(group) < 2:
            raise ValueError(
                "GRPO needs at least two completions in every prompt group"
            )
        if len({tuple(row["prompt_token_ids"]) for row in group}) != 1:
            raise ValueError("a GRPO group contains different prompts")
    random.Random(seed).shuffle(groups)
    for start in range(0, len(groups), groups_per_batch):
        yield [
            record
            for group in groups[start : start + groups_per_batch]
            for record in group
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRESETS["grpo"].model)
    parser.add_argument("--reference-model", default=None)
    parser.add_argument("--rollouts", default=PRESETS["grpo"].rollout_path)
    parser.add_argument("--output", default="artifacts/grpo_model")
    parser.add_argument("--groups-per-batch", type=int, default=1)
    parser.add_argument("--policy-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    add_training_args(parser, "grpo")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    records = read_jsonl(args.rollouts)
    validate_rollouts(
        records, args.model, policy_gradient=True, max_seq_length=args.max_seq_length
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    tokenizer = load_tokenizer(args.model)
    model = load_policy(
        args.model,
        device,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_implementation=args.attention_implementation,
    )
    model.train()
    reference = None
    if args.beta:
        require_matching_tokenizers(tokenizer, args.reference_model or args.model)
        reference = load_policy(
            args.reference_model or args.model,
            device,
            attention_implementation=args.attention_implementation,
            trainable=False,
        )
        reference.eval()
        reference.requires_grad_(False)
    optimizer = AccumulatingOptimizer(
        model, args.learning_rate, args.gradient_accumulation_steps, args.max_grad_norm
    )

    for policy_epoch in range(args.policy_epochs):
        for batch_index, rows in enumerate(
            group_batches(records, args.groups_per_batch, args.seed + policy_epoch),
            start=1,
        ):
            batch = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
            assert batch.old_logprobs is not None
            rewards = torch.tensor(
                [float(row["reward"]) for row in rows], device=device
            )
            advantages = group_relative_advantages(
                rewards, [str(row["group_id"]) for row in rows]
            )
            with torch.no_grad():
                reference_logprobs = (
                    None
                    if reference is None
                    else policy_logprobs(reference, batch, args.logit_chunk_size)
                )
            new_logprobs = policy_logprobs(model, batch, args.logit_chunk_size)
            loss, metrics = grpo_loss(
                new_logprobs,
                batch.old_logprobs,
                advantages,
                batch.action_mask,
                args.clip_epsilon,
                args.beta,
                reference_logprobs,
            )
            optimizer.backward(loss)
            print(
                f"batch={batch_index} epoch={policy_epoch + 1} reward={rewards.mean().item():.3f} "
                f"grpo_loss={metrics['loss']:.4f} reference_kl={metrics['reference_kl']:.4f}"
            )
        optimizer.flush()

    optimizer.flush()
    save_policy(model, tokenizer, args.output)
    print(f"saved GRPO policy to {args.output}")


if __name__ == "__main__":
    main()
