# Code walkthrough

The project favors visible intermediate data over a highly optimized trainer. One JSONL file is the
boundary between inference and learning, and each loss fits in one source file.

## 1. Input data

[`data/toy_math.jsonl`](../data/toy_math.jsonl) contains:

```json
{
  "id": "add",
  "prompt": "Mina has 17 marbles ...",
  "answer": "25",
  "reference": "Mina starts with ... Final answer: 25"
}
```

`answer` is the compact value used by the verifier. `reference` is a worked solution used by SDFT
and as OPSD privileged information. Real tasks should replace the regex verifier with unit tests,
symbolic checks, a simulator, or another deterministic environment.

## 2. vLLM rollout

[`self_distill_rl/rollout.py`](../self_distill_rl/rollout.py) applies the tokenizer's chat template,
passes prompt token IDs to vLLM, and requests one log-probability entry per sampled token. A rollout
record contains:

```json
{
  "group_id": "add",
  "prompt_token_ids": [1, 2, 3],
  "completion_token_ids": [4, 5],
  "old_logprobs": [-0.7, -0.2],
  "completion": "... Final answer: 25",
  "reward": 1.0,
  "feedback": "The answer passed the verifier ..."
}
```

`group_id` joins GRPO responses sampled from one problem. The raw token IDs prevent a decode/encode
round trip from changing the action sequence. `old_logprobs` are `log pi_old` in PPO and GRPO.
The rollout defaults to temperature 1 and top-p 1 so PyTorch's ordinary full-softmax log-probability
describes the same behavior distribution. If sampling is tempered or truncated, the trainer must
reproduce that transform before forming importance ratios.

## 3. Causal token alignment

For a prompt of length `P` followed by `C` completion tokens, model logit position `P-1` predicts
the first completion token. The action mask is therefore true on the slice `[P-1 : P+C-1]` of the
shifted token arrays. [`causal_batch`](../self_distill_rl/io.py) creates that mask and right-pads all
other positions.

This same alignment is used for:

- old and new policy log-probabilities;
- PPO values, advantages, and returns;
- GRPO response advantages;
- response-only SDFT cross entropy;
- student and privileged-teacher completion logits.

## 4. PPO and GRPO

[`self_distill_rl/objectives.py`](../self_distill_rl/objectives.py) contains no model code. It accepts
tensors and implements terminal reward placement, GAE, PPO clipping, group normalization, the GRPO
surrogate, and the optional reference KL. This separation makes the signs and masks easy to test.

PPO freezes old value predictions and advantages before its inner epochs. GRPO preserves complete
prompt groups in a mini-batch; splitting a group would change its mean and standard deviation.

## 5. The three self-distillation paths

SDFT generates a new offline target and then uses ordinary cross entropy. No rollout
log-probabilities appear in its loss.

OPSD concatenates each sampled completion to two different contexts:

```text
student context = original chat prompt
teacher context = problem + private verified solution
```

The frozen seed teacher and trainable student produce aligned `[C, vocabulary]` tensors. Their JSD
is averaged over response tokens.

SDPO uses:

```text
student context = original chat prompt
teacher context = problem + post-attempt environment feedback
```

Both passes use the current model weights. The teacher pass occurs first under `torch.no_grad()`,
which implements `stopgrad`; only the question-only student branch is updated.

## 6. One-GPU lifecycle

[`on_policy_loop.py`](../on_policy_loop.py) runs this sequence:

```text
checkpoint N -> vLLM rollout process -> rollout JSONL -> PyTorch train process -> checkpoint N+1
```

Process exit is the resource boundary. It reliably releases vLLM's engine and KV cache before model
weights plus Adam states are loaded for training. This is deliberately less efficient than a hybrid
engine with in-memory weight synchronization, but much easier to inspect.

## 7. Extending the tutorial

The first useful extensions are:

1. Replace `exact_match_reward` and `environment_feedback` in
   [`rewards.py`](../self_distill_rl/rewards.py).
2. Add length limits before batching; attention cost grows quadratically with sequence length.
3. Add gradient accumulation if batch size 1 is too noisy.
4. Replace full-vocabulary distillation with top-k logits when vocabulary memory dominates.
5. Move orchestration to veRL when rollouts involve asynchronous tools or multiple GPUs.

Avoid blindly mixing rollout files from old checkpoints. PPO/GRPO importance clipping tolerates
small policy lag, not an arbitrary replay buffer. OPSD/SDPO are also defined on current-policy
states, so refresh their generations each round.
