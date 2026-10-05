# Code walkthrough

One JSONL file is the boundary between inference and learning, and each loss fits in one source
file. Model loading and vocabulary operations are optimized for Qwen3.5-0.8B and LFM2.5-350M while
keeping the objectives explicit.

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

[`prepare_datasets.py`](../prepare_datasets.py) also downloads and normalizes official Hugging Face
GSM8K and SVAMP train/test splits. Defaults in [`presets.py`](../self_distill_rl/presets.py) select
GSM8K for the Qwen self-distillation methods and SVAMP for LFM PPO/GRPO. Each normalized record
retains dataset provenance and split identity. The driver rejects train/eval prompt overlap.

`answer` is the compact value used by the verifier. `reference` is a worked solution used by SDFT
and as OPSD privileged information. Real tasks should replace the regex verifier with unit tests,
symbolic checks, a simulator, or another deterministic environment.

## 2. Rollout

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
describes the same behavior distribution. Checkpoint generation defaults are disabled and top-k is
unrestricted. Both vLLM and the portable Transformers backend return sampled IDs and probabilities.
PPO/GRPO reject tempered or truncated sampling. Model identity and sampling settings are retained in
each record. vLLM also bounds sequence length and concurrency and skips Qwen's vision encoder.

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

PPO freezes old value predictions and advantages for the entire buffer before any optimizer step.
GRPO preserves complete
prompt groups in a mini-batch; splitting a group would change its mean and standard deviation.

## 5. The three self-distillation paths

SDFT generates a new offline target and then uses ordinary cross entropy. No rollout
log-probabilities appear in its loss.

OPSD concatenates each sampled completion to two different contexts:

```text
student context = original chat prompt
teacher context = problem + private verified solution
```

The frozen seed teacher and trainable student produce aligned completion hidden states. Output
projection and exact full-vocabulary JSD are computed in small token chunks, checkpointing those
operations for backward. Prompt vocabulary logits are never materialized. The loss remains the
mean over response tokens, then the mean over responses.

SDPO uses:

```text
student context = original chat prompt
teacher context = problem + post-attempt environment feedback
```

Both passes use the current model weights. The teacher pass occurs first under `torch.no_grad()`,
which implements `stopgrad`; only the question-only student branch is updated.

## 6. Precision and one-GPU lifecycle

Trainable parameters and Adam state use FP32 so small RL updates remain representable. Forward
computation uses BF16 autocast on supported CUDA devices or scaled FP16 on older devices. Frozen
reference/teacher weights use the compute dtype. Qwen's vision weights stay frozen and its native
checkpoint layout is retained for vLLM. Gradient checkpointing and accumulation are enabled by
default; partial accumulation windows retain the same average gradient.

[`on_policy_loop.py`](../on_policy_loop.py) runs this sequence:

```text
checkpoint N -> vLLM rollout process -> rollout JSONL -> PyTorch train process -> checkpoint N+1
```

Process exit is the resource boundary. It reliably releases vLLM's engine and KV cache before model
weights plus Adam states are loaded for training. This is deliberately less efficient than a hybrid
engine with in-memory weight synchronization, but much easier to inspect.

The driver evaluates the seed checkpoint and every saved checkpoint through [`evaluate.py`](../evaluate.py).
The held-out branch has no privileged solutions or feedback in its prompt. Metrics include answer
accuracy, response length, and truncation, alongside per-example predictions. SDFT uses the same
driver with a rewrite stage in place of on-policy rollouts. `--rollout-python` separates vLLM's
dependency environment from the training environment when their Transformers pins differ.

## 7. Extending the tutorial

The first useful extensions are:

1. Replace `exact_match_reward` and `environment_feedback` in
   [`rewards.py`](../self_distill_rl/rewards.py).
2. Tune the existing length limits before batching; attention cost increases with sequence length.
3. Tune gradient accumulation if batch size 1 is too noisy.
4. Tune `--logit-chunk-size` when vocabulary workspace dominates; exact divergences are retained.
5. Move orchestration to veRL when rollouts involve asynchronous tools or multiple GPUs.

Avoid blindly mixing rollout files from old checkpoints. PPO/GRPO importance clipping tolerates
small policy lag, not an arbitrary replay buffer. OPSD/SDPO are also defined on current-policy
states, so refresh their generations each round.
