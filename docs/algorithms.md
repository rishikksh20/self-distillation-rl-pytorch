# Concepts and equations

All five methods start with a causal language model, but they are not interchangeable. PPO and GRPO
learn from scalar outcomes. SDFT rewrites an offline supervised dataset. OPSD and SDPO turn extra
context available during training into dense token-level supervision.

The default models are Qwen3.5-0.8B for SDFT/OPSD/SDPO and LFM2.5-350M for PPO/GRPO. GSM8K
provides worked solutions for distillation; SVAMP provides shorter verifiable arithmetic for the
smaller policy. Both use official training splits, with separate test splits for evaluation.

Notation used below:

- `x`: prompt or problem
- `y = (y1, ..., yT)`: generated response
- `pi_theta`: trainable policy
- `pi_old`: policy that produced the rollout
- `pi_ref`: fixed reference policy
- `r`: scalar outcome reward
- `f`: tokenized environment feedback
- `y*`: privileged verified solution

## The shared rollout boundary

The student samples `y ~ pi_old(. | x)` with vLLM. For every generated token the rollout file stores
`log pi_old(yt | x, y<t)`. PyTorch then reloads the same checkpoint and evaluates the same token IDs.
The importance ratio is therefore

```text
rho_t(theta) = exp(log pi_theta(yt | x, y<t) - log pi_old(yt | x, y<t)).
```

Keeping token IDs is important. Decoding and re-tokenizing text can change whitespace tokens and
invalidate the behavior log-probabilities.

## PPO

PPO uses a critic `V_phi(s_t)` and generalized advantage estimation. The toy environment gives one
terminal reward, so intermediate reward entries are zero. Working backward,

```text
delta_t = r_t + gamma V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma lambda A_{t+1}.
```

The actor minimizes the negative clipped surrogate:

```text
L_policy = -mean_t min(rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t).
```

The critic uses the larger of the unclipped and clipped squared errors. In
[`train_ppo.py`](../train_ppo.py), the causal LM and a one-layer value head share hidden states. PPO
is useful when rewards vary independently per trajectory and a learned baseline is worth its memory
and complexity.

Old values, advantages, and returns are frozen for the entire rollout buffer before the first
update. Later minibatches therefore use the same behavior critic. The default is one PPO epoch per
buffer, with new rollouts generated from each updated checkpoint by the driver.

## GRPO

GRPO removes the learned critic. It samples a group of `G` responses for the same prompt and
standardizes rewards inside that group:

```text
A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + epsilon).
```

The same response-level advantage is applied to every generated token of response `i`, then a
PPO-style clipped ratio is optimized. Tokens are averaged within each response before responses are
averaged, so long completions do not receive extra weight. An optional reference penalty uses the positive estimator

```text
KL_hat = exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1.
```

If every response in a group gets the same reward, its relative advantages are zero and that group
does not produce a policy-gradient update. This is expected, and explains why group diversity and a
non-saturated verifier matter. See [`train_grpo.py`](../train_grpo.py).

## SDFT

Self-Distillation Fine-Tuning addresses distribution shift in ordinary fine-tuning. Given a task
pair `(x, y*)`, the seed model is prompted with both the task and reference response and asked to
rewrite the response in its own style:

```text
y_tilde ~ pi_seed(. | x, y*).
```

A task-specific verifier selects the rewrite if it remains correct and otherwise retains the
original reference. Training is then standard masked negative log likelihood:

```text
L_SDFT = -sum_t log pi_theta(y_selected,t | x, y_selected,<t).
```

This repository implements those distinct stages in
[`generate_sdft_data.py`](../generate_sdft_data.py) and [`train_sdft.py`](../train_sdft.py). It does
not mislabel logit matching as SDFT; the original method is a data-rewriting pipeline.

## OPSD

On-Policy Self-Distillation samples a response from the question-only student, then scores the same
response prefixes with a self-teacher that sees privileged information:

```text
student: pi_theta(. | x, y<t)
teacher: pi_seed(. | x, y*, y<t).
```

The paper fixes the teacher to the initial policy for stability and uses a full-vocabulary
Jensen-Shannon divergence in its main setup. With `m = 0.5(student + teacher)`,

```text
JSD(student, teacher) = 0.5 KL(student || m) + 0.5 KL(teacher || m).
```

[`train_opsd.py`](../train_opsd.py) defaults to that JSD and also exposes reverse KL for comparison.
Only the student receives gradients. The trajectory remains on-policy for the student even though
the fixed teacher gets more context.

The implementation computes the exact divergence in checkpointed token chunks, projecting only
completion hidden states. This avoids storing prompt vocabulary logits and full response-sized
softmax intermediates for Qwen's 248,320-token vocabulary; it does not approximate the distribution.

## SDPO

Self-Distillation Policy Optimization targets environments that return rich text: compiler errors,
test failures, judge critiques, tool results, or another successful rollout. The current model plays
both roles:

```text
student: pi_theta(. | x, y<t)
teacher: stopgrad(pi_theta(. | x, f, y<t)).
```

Its published objective is the full-vocabulary reverse KL

```text
L_SDPO = sum_t KL(student || stopgrad(teacher)).
```

[`train_sdpo.py`](../train_sdpo.py) first evaluates the feedback-conditioned branch under `no_grad`,
then evaluates the ordinary context with gradients. This is one model with two contexts, not an
external teacher. Because feedback describes the finished attempt, the teacher performs hindsight
credit assignment at every response prefix.

## Comparison

| Method | On-policy samples? | Extra learned model? | Training-only privilege | Credit granularity |
|---|---:|---:|---|---|
| PPO | Yes | Value head | Scalar reward | Token advantage from GAE |
| GRPO | Yes | Optional reference | Group rewards | One advantage per response |
| SDFT | No | No | Reference target during rewrite | Supervised target tokens |
| OPSD | Yes | Frozen seed copy | Verified solution | Full next-token distribution |
| SDPO | Yes | No | Environment feedback | Full next-token distribution |

## Primary references

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Yang et al., [Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning](https://aclanthology.org/2024.acl-long.58/)
- Zhao et al., [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734)
- Hübotter et al., [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
