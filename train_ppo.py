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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rollouts", default="artifacts/rollouts.jsonl")
    parser.add_argument("--output", default="artifacts/ppo_model")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
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
    policy = load_policy(
        args.model, device, gradient_checkpointing=args.gradient_checkpointing
    )
    actor_critic = ActorCritic(policy).to(device)
    actor_critic.load_value_head(args.model)
    actor_critic.train()
    optimizer = torch.optim.AdamW(actor_critic.parameters(), lr=args.learning_rate)
    records = read_jsonl(args.rollouts)

    for batch_index, rows in enumerate(
        shuffled_batches(records, args.batch_size, args.seed), start=1
    ):
        batch = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
        assert batch.old_logprobs is not None
        rewards = torch.tensor([float(row["reward"]) for row in rows], device=device)

        # Freeze the behavior values and advantages for all PPO epochs on this mini-buffer.
        with torch.no_grad():
            _, old_values = actor_critic(batch)
            old_values = old_values.float()
            token_rewards = terminal_rewards(rewards, batch.action_mask)
            advantages, returns = generalized_advantage_estimate(
                old_values,
                token_rewards,
                batch.action_mask,
                args.gamma,
                args.gae_lambda,
            )

        for ppo_epoch in range(args.ppo_epochs):
            new_logprobs, new_values = actor_critic(batch)
            loss, metrics = ppo_loss(
                new_logprobs,
                batch.old_logprobs,
                advantages,
                new_values.float(),
                old_values,
                returns,
                batch.action_mask,
                args.clip_epsilon,
                args.value_clip_epsilon,
                args.value_coef,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor_critic.parameters(), args.max_grad_norm
            )
            optimizer.step()
            print(
                f"batch={batch_index} epoch={ppo_epoch + 1} reward={rewards.mean().item():.3f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
                f"clip_fraction={metrics['clip_fraction']:.3f}"
            )

    actor_critic.save(args.output, tokenizer)
    print(f"saved PPO actor and value head to {args.output}")


if __name__ == "__main__":
    main()
