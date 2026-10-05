"""The small mathematical core of PPO and GRPO."""

from __future__ import annotations

import torch

from .modeling import masked_mean


def terminal_rewards(rewards: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    token_rewards = torch.zeros_like(action_mask, dtype=torch.float32)
    lengths = action_mask.sum(dim=1)
    for row, length in enumerate(lengths.tolist()):
        if length:
            positions = action_mask[row].nonzero(as_tuple=False).squeeze(-1)
            token_rewards[row, positions[-1]] = rewards[row]
    return token_rewards


def generalized_advantage_estimate(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE for right-padded, terminal language-model trajectories."""
    advantages = torch.zeros_like(values, dtype=torch.float32)
    for row in range(values.shape[0]):
        positions = action_mask[row].nonzero(as_tuple=False).squeeze(-1).tolist()
        running_advantage = torch.tensor(0.0, device=values.device)
        for index in reversed(range(len(positions))):
            position = positions[index]
            next_value = (
                values[row, positions[index + 1]] if index + 1 < len(positions) else 0.0
            )
            delta = rewards[row, position] + gamma * next_value - values[row, position]
            running_advantage = delta + gamma * gae_lambda * running_advantage
            advantages[row, position] = running_advantage
    returns = advantages + values
    valid = advantages[action_mask]
    if valid.numel() > 1:
        advantages[action_mask] = (valid - valid.mean()) / (
            valid.std(unbiased=False) + 1e-8
        )
    return advantages, returns


def ppo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    action_mask: torch.Tensor,
    clip_epsilon: float,
    value_clip_epsilon: float,
    value_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_ratio = (new_logprobs - old_logprobs).clamp(-20, 20)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -masked_mean(torch.minimum(unclipped, clipped), action_mask)

    clipped_values = old_values + (new_values - old_values).clamp(
        -value_clip_epsilon, value_clip_epsilon
    )
    value_loss = 0.5 * masked_mean(
        torch.maximum(
            (new_values - returns).square(), (clipped_values - returns).square()
        ),
        action_mask,
    )
    loss = policy_loss + value_coef * value_loss
    clipped_fraction = masked_mean(
        (torch.abs(ratio - 1) > clip_epsilon).float(), action_mask
    )
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "clip_fraction": float(clipped_fraction.detach()),
    }


def group_relative_advantages(
    rewards: torch.Tensor, group_ids: list[str]
) -> torch.Tensor:
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group_id in dict.fromkeys(group_ids):
        indices = [index for index, value in enumerate(group_ids) if value == group_id]
        group_rewards = rewards[indices]
        std = group_rewards.std(unbiased=False)
        if std > 1e-8:
            advantages[indices] = (group_rewards - group_rewards.mean()) / (std + 1e-8)
    return advantages


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    sequence_advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_epsilon: float,
    beta: float = 0.0,
    reference_logprobs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = (new_logprobs - old_logprobs).clamp(-20, 20).exp()
    advantages = sequence_advantages[:, None].expand_as(ratio)
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages,
    )
    objective = surrogate
    kl_value = torch.tensor(0.0, device=new_logprobs.device)
    if beta:
        if reference_logprobs is None:
            raise ValueError("beta > 0 requires reference log-probabilities")
        # Positive, low-variance estimator of KL(policy || reference).
        log_ratio = (reference_logprobs - new_logprobs).clamp(-20, 20)
        per_token_kl = log_ratio.exp() - log_ratio - 1
        objective = objective - beta * per_token_kl
        kl_value = masked_mean(per_token_kl, action_mask)
    # GRPO gives each response equal weight: average tokens inside a response first,
    # then average responses. A global token mean would overweight long completions.
    token_counts = action_mask.sum(dim=1).clamp_min(1)
    per_response_objective = (objective * action_mask).sum(dim=1) / token_counts
    loss = -per_response_objective.mean()
    return loss, {
        "loss": float(loss.detach()),
        "reference_kl": float(kl_value.detach()),
    }
