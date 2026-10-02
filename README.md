# Self-Distillation RL in Vanilla PyTorch

A small, readable tutorial repository for PPO, GRPO, SDFT, OPSD, and SDPO on causal language
models. Rollouts use **vLLM**; optimization uses direct **PyTorch** tensor operations. There is no
TRL, OpenRLHF, or hidden trainer abstraction.

This is teaching code for small models on one NVIDIA GPU, not a production trainer. The defaults use
`Qwen/Qwen2.5-0.5B-Instruct`, short sequences, tiny batches, and a generated arithmetic dataset so
that every reward can be inspected.

## What is implemented

| Method | Signal | Main file |
|---|---|---|
| PPO | Scalar verifier reward + learned value head | `train_ppo.py` |
| GRPO | Group-normalized verifier rewards | `train_grpo.py` |
| SDFT | Verified model-written replacements for SFT targets | `generate_sdft_data.py`, `train_sdft.py` |
| OPSD | Frozen self-teacher with a privileged reference solution | `train_opsd.py` |
| SDPO | Live self-teacher with rich environment feedback | `train_sdpo.py` |

See [Concepts and equations](docs/algorithms.md), [code walkthrough](docs/code_walkthrough.md), and
the [veRL agentic guide](docs/agentic_verl.md).

## Installation

Linux, Python 3.10+, an NVIDIA GPU, and a CUDA-compatible PyTorch build are expected. Install the
right PyTorch wheel for your CUDA version first if the default pip wheel is not appropriate, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

vLLM has stricter CUDA/platform requirements than the training code. Consult its installation guide
if the wheel does not match your system.

## Quick start

Generate four completions per question. The output stores token IDs, reward, feedback, and each
sampled token's behavior-policy log-probability:

```bash
python3 -m self_distill_rl.rollout \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts data/toy_math.jsonl \
  --samples-per-prompt 4 \
  --output artifacts/rollouts.jsonl
```

Train one pass with any online method:

```bash
python3 train_ppo.py  --rollouts artifacts/rollouts.jsonl --output artifacts/ppo_model
python3 train_grpo.py --rollouts artifacts/rollouts.jsonl --output artifacts/grpo_model
python3 train_opsd.py --rollouts artifacts/rollouts.jsonl --output artifacts/opsd_model
python3 train_sdpo.py --rollouts artifacts/rollouts.jsonl --output artifacts/sdpo_model
```

PPO can use one rollout per prompt; GRPO needs multiple rollouts in each group and should keep the
default of four or more. OPSD and GRPO load a frozen second copy of the 0.5B model. If memory is
tight, disable GRPO's reference KL with `--beta 0`, reduce sequence length, and use batch size 1.

For genuinely on-policy rounds, use the driver:

```bash
python3 on_policy_loop.py --algorithm grpo --rounds 2 --samples-per-prompt 4
```

The driver launches rollout and training as separate processes. vLLM exits before the optimizer is
created, so their large memory pools never coexist. It also refreshes rollout behavior
log-probabilities from each new checkpoint.

## SDFT quick start

SDFT is a two-stage data pipeline rather than policy-gradient RL:

```bash
python3 generate_sdft_data.py --data data/toy_math.jsonl --output artifacts/sdft_data.jsonl
python3 train_sdft.py --data artifacts/sdft_data.jsonl --output artifacts/sdft_model
```

The first stage asks the seed model to rewrite each reference response in its own style. A simple
final-answer verifier accepts the rewrite or falls back to the original reference. The second stage
is response-masked causal-language-model cross entropy.

## Repository layout

```text
self_distill_rl/
  io.py              JSONL, chat formatting, right-padded action masks
  rollout.py         vLLM generation and behavior log-probabilities
  modeling.py        model loading, log-probs, value head, divergences
  objectives.py      PPO/GAE and GRPO equations in PyTorch
  distillation.py    OPSD/SDPO privileged-context construction
data/toy_math.jsonl  tiny inspectable task/reference dataset
tests/               objective and reward smoke tests
examples/verl_agentic/  optional multi-turn veRL example
```

## Checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q self_distill_rl *.py
```

The unit tests require PyTorch but do not download a model. Full training requires a GPU and model
download. Start with one round and inspect `artifacts/rollouts.jsonl` before scaling anything.

## Scope and caveats

- The included exact-match reward is intentionally narrow. Replace it with your own deterministic
  verifier, tests, or environment before using real data.
- The tutorial SDPO feedback includes the verified answer when a rollout fails. That makes the
  privileged signal obvious, but it is stronger than a compiler error or ordinary scalar reward.
- Full-vocabulary OPSD/SDPO losses are easier to understand but use more memory than sampled-token or
  top-k approximations.
- Optimizer state is restarted between rounds by `on_policy_loop.py`; that keeps the orchestration
  transparent but is not ideal for long runs.
- Generated checkpoints and rollout buffers belong under `artifacts/`, which is git-ignored.
