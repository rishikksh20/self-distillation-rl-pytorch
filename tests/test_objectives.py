import unittest

import torch

from self_distill_rl.objectives import (
    generalized_advantage_estimate,
    group_relative_advantages,
    grpo_loss,
    terminal_rewards,
)
from self_distill_rl.rewards import exact_match_reward, extract_final_answer


class RewardTests(unittest.TestCase):
    def test_answer_extractors(self) -> None:
        self.assertEqual(extract_final_answer("work... \\boxed{42}"), "42")
        self.assertEqual(extract_final_answer("reasoning\n#### 17"), "17")
        self.assertEqual(exact_match_reward("Final answer: 25", "25"), 1.0)


class ObjectiveTests(unittest.TestCase):
    def test_group_advantages_are_normalized_per_prompt(self) -> None:
        rewards = torch.tensor([0.0, 1.0, 2.0, 2.0])
        advantages = group_relative_advantages(rewards, ["a", "a", "b", "b"])
        self.assertTrue(torch.allclose(advantages[:2], torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.equal(advantages[2:], torch.zeros(2)))

    def test_gae_puts_terminal_reward_on_valid_tokens(self) -> None:
        mask = torch.tensor([[True, True, False], [True, False, False]])
        rewards = terminal_rewards(torch.tensor([1.0, 2.0]), mask)
        self.assertTrue(
            torch.equal(rewards, torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]))
        )
        advantages, returns = generalized_advantage_estimate(
            torch.zeros_like(rewards), rewards, mask, gamma=1.0, gae_lambda=1.0
        )
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)

    def test_grpo_has_zero_loss_for_constant_group_reward_without_kl(self) -> None:
        logprobs = torch.zeros((2, 3))
        mask = torch.ones((2, 3), dtype=torch.bool)
        advantages = group_relative_advantages(torch.ones(2), ["same", "same"])
        loss, _ = grpo_loss(logprobs, logprobs, advantages, mask, 0.2)
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
