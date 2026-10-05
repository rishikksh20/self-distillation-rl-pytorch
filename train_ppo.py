"""Minimal clipped PPO for a causal LM, implemented with vanilla PyTorch."""

from __future__ import annotations

import argparse
import random

import torch

from self_distill_rl.io import read_jsonl, rollout_batch, shuffled_batches
from self_distill_rl.modeling import (
    ActorCritic,
    default_device,
    load_policy,
    load_tokenizer,
)
from self_distill_rl.objectives import (
    generalized_advantage_estimate,
    ppo_loss,
    terminal_rewards,
)
from self_distill_rl.presets import PRESETS
from self_distill_rl.training import (
    AccumulatingOptimizer,
    add_training_args,
    validate_rollouts,
    validate_training_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRESETS["ppo"].model)
    parser.add_argument("--rollouts", default=PRESETS["ppo"].rollout_path)
    parser.add_argument("--output", default="artifacts/ppo_model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    add_training_args(parser, "ppo")
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
    policy = load_policy(
        args.model,
        device,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_implementation=args.attention_implementation,
    )
    actor_critic = ActorCritic(policy).to(device)
    actor_critic.load_value_head(args.model)
    actor_critic.train()
    optimizer = AccumulatingOptimizer(
        actor_critic,
        args.learning_rate,
        args.gradient_accumulation_steps,
        args.max_grad_norm,
    )

    # Snapshot the entire rollout buffer before any actor/critic update. Computing these
    # lazily per minibatch would use an already-updated critic for later trajectories.
    frozen_batches = []
    actor_critic.eval()
    for rows in shuffled_batches(records, args.batch_size, args.seed):
        batch = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
        assert batch.old_logprobs is not None
        rewards = torch.tensor([float(row["reward"]) for row in rows], device=device)

        with torch.no_grad():
            old_values = actor_critic.predict_values(batch)
            old_values = old_values.detach().float().clone()
            token_rewards = terminal_rewards(rewards, batch.action_mask)
            advantages, returns = generalized_advantage_estimate(
                old_values,
                token_rewards,
                batch.action_mask,
                args.gamma,
                args.gae_lambda,
            )
        frozen_batches.append((rows, old_values.cpu(), advantages.cpu(), returns.cpu()))

    actor_critic.train()
    for ppo_epoch in range(args.ppo_epochs):
        for batch_index, (rows, old_values, advantages, returns) in enumerate(
            frozen_batches, start=1
        ):
            batch = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
            rewards = torch.tensor(
                [float(row["reward"]) for row in rows], device=device
            )
            new_logprobs, new_values = actor_critic(batch, args.logit_chunk_size)
            loss, metrics = ppo_loss(
                new_logprobs,
                batch.old_logprobs,
                advantages.to(device),
                new_values.float(),
                old_values.to(device),
                returns.to(device),
                batch.action_mask,
                args.clip_epsilon,
                args.value_clip_epsilon,
                args.value_coef,
            )
            optimizer.backward(loss)
            print(
                f"batch={batch_index} epoch={ppo_epoch + 1} reward={rewards.mean().item():.3f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
                f"clip_fraction={metrics['clip_fraction']:.3f}"
            )
        optimizer.flush()

    optimizer.flush()
    actor_critic.save(args.output, tokenizer)
    print(f"saved PPO actor and value head to {args.output}")


if __name__ == "__main__":
    main()
