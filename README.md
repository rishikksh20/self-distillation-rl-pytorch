# Self-Distillation RL in Vanilla PyTorch

Readable single-GPU implementations of PPO, GRPO, SDFT, OPSD, and SDPO. Optimization uses direct
PyTorch operations. Generation supports vLLM and a portable Hugging Face Transformers backend.
There is no TRL or hidden trainer abstraction.

The model and dataset defaults are specific to each method:

| Method | Default model | Training data | Held-out evaluation | Response budget | Sequence limit |
|---|---|---|---|---:|---:|
| SDFT | `Qwen/Qwen3.5-0.8B` | GSM8K train, verified self-rewrites | GSM8K test | 512 | 2048 |
| OPSD | `Qwen/Qwen3.5-0.8B` | GSM8K train, reference-conditioned frozen teacher | GSM8K test | 512 | 2048 |
| SDPO | `Qwen/Qwen3.5-0.8B` | GSM8K train, verifier feedback | GSM8K test | 512 | 2048 |
| PPO | `LiquidAI/LFM2.5-350M` | SVAMP train, scalar answer reward | SVAMP test | 256 | 1024 |
| GRPO | `LiquidAI/LFM2.5-350M` | SVAMP train, groups of four responses | SVAMP test | 256 | 1024 |

SDPO is Self-Distillation Policy Optimization. LFM2.5-350M is the public checkpoint selected for
PPO/GRPO. Override `--model` to use a compatible local checkpoint. See the official
[Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-350M).

## Installation

Use Python 3.10+ and install a CUDA-compatible PyTorch build for GPU training.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Transformers 5.3+ is required for Qwen3.5. The Transformers generation backend works without vLLM
and can run small smoke tests on CPU. Full-parameter training of the target models is intended for
a CUDA GPU; CPU runs are much slower.

For faster Qwen training, optionally install the CUDA kernels after PyTorch:

```bash
python -m pip install -r requirements-kernels.txt
```

These provide optimized causal convolution and gated linear attention. PyTorch fallbacks preserve
the equations when the kernels are absent. See [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5).

vLLM has its own CUDA, Torch, and Transformers requirements. Use a separate inference environment
when its dependency pins differ from training:

```bash
uv venv .venv-rollout
uv pip install --python .venv-rollout/bin/python -r requirements-rollout.txt
```

The driver accepts `--rollout-python .venv-rollout/bin/python` for both generation and evaluation.
See the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/) and
[Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

## Jupyter training tutorials

The [notebook guide](notebooks/README.md) links five separate tutorials for
[SDFT](notebooks/01_sdft_qwen35.ipynb), [SDPO](notebooks/02_sdpo_qwen35.ipynb),
[OPSD](notebooks/03_opsd_qwen35.ipynb), [PPO](notebooks/04_ppo_lfm25.ipynb), and
[GRPO](notebooks/05_grpo_lfm25.ipynb). Each explains the equations, model and optimizer components,
detailed dataset formats, real training stages, validation, and seed-versus-trained comparisons
on both GSM8K and SVAMP test splits. Training progress uses a validation partition reserved from
official train; final benchmark cells use official test.

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab notebooks/
```

Select a kernel using the training environment. Notebook outputs are cleared; benchmark results
are produced by running the evaluation cells. Full target-model training is intended for CUDA.

For Google Colab, the [standalone editions](notebooks/README.md#standalone-google-colab-editions)
embed all required training and evaluation code and include their own dependency-install cell.
Upload any `_standalone.ipynb` file directly; no repository checkout is needed.
Standalone recipes can train on math, Python coding, tool use, or a configurable mixture,
with domain-specific validation and paired before/after benchmarks. Their tutorials describe
the MBPP/HumanEval execution protocol and adapted BFCL argument-matching score.

## Standard datasets

Download and normalize the official Hugging Face splits:

```bash
python prepare_datasets.py
```

This writes:

```text
artifacts/datasets/gsm8k/train.jsonl    7473 examples
artifacts/datasets/gsm8k/test.jsonl     1319 examples
artifacts/datasets/svamp/train.jsonl     700 examples
artifacts/datasets/svamp/test.jsonl      300 examples
```

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) supplies questions, worked solutions, and
numeric final answers. [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) supplies shorter
arithmetic word problems, equations, and answers, suitable for the smaller policy model. Either
dataset can be used with any method by passing its JSONL path.

All records contain `id`, `prompt`, `answer`, `reference`, `dataset`, and `split`. Training prompts
contain the question and response-format instruction. Solutions enter SDFT rewriting and privileged
teacher contexts; evaluation uses only the ordinary question prompt.

Use `--dataset gsm8k` or `--dataset svamp` to prepare one dataset, `--train-limit`/`--eval-limit` for
small subsets, and `--revision <commit-sha>` to pin a dataset version. A manifest records the source,
revision, counts, and shuffle seed. Train and test retain separate official splits and stable IDs.
The original `data/toy_math.jsonl` remains available for inspectable local examples.

## Run training and evaluation

The driver prepares missing default data, takes 256 training examples and 64 test examples, evaluates
the seed, and evaluates every new checkpoint. Use the Transformers backend for a single environment:

```bash
python on_policy_loop.py --algorithm sdft --backend transformers --rounds 1
python on_policy_loop.py --algorithm opsd --backend transformers --rounds 1
python on_policy_loop.py --algorithm sdpo --backend transformers --rounds 1
python on_policy_loop.py --algorithm ppo  --backend transformers --rounds 1
python on_policy_loop.py --algorithm grpo --backend transformers --rounds 1
```

For faster GPU generation with the separate vLLM environment:

```bash
python on_policy_loop.py --algorithm grpo --rounds 2 \
  --rollout-python .venv-rollout/bin/python
```

Generation, optimization, and evaluation run in separate processes, freeing inference memory before
training. GRPO's reference and OPSD's teacher stay fixed at the initial checkpoint across rounds.
SDFT uses the same driver to alternate rewriting and supervised fine-tuning.

Outputs live under `artifacts/on_policy/<algorithm>/`: selected training prompts, round buffers,
checkpoints, `baseline_eval.json`, and `round_XX_eval.json`, plus per-example prediction files.
Use `--train-limit` and `--eval-limit` to change subset sizes; `--prompts` and `--eval-prompts` select
custom files. The driver rejects overlapping train/eval prompts. `--skip-eval` skips evaluation.

For individual stages:

```bash
# PPO and GRPO use their own model/dataset presets.
python -m self_distill_rl.rollout --algorithm ppo --backend transformers \
  --output artifacts/ppo_rollouts.jsonl
python train_ppo.py --rollouts artifacts/ppo_rollouts.jsonl

python -m self_distill_rl.rollout --algorithm grpo --backend transformers \
  --output artifacts/grpo_rollouts.jsonl
python train_grpo.py --rollouts artifacts/grpo_rollouts.jsonl

# Generate Qwen rollouts separately from LFM rollouts.
python -m self_distill_rl.rollout --algorithm opsd --backend transformers \
  --output artifacts/qwen_rollouts.jsonl
python train_opsd.py --rollouts artifacts/qwen_rollouts.jsonl
python train_sdpo.py --rollouts artifacts/qwen_rollouts.jsonl

# SDFT verifies model rewrites and falls back to original worked solutions.
python generate_sdft_data.py --backend transformers
python train_sdft.py
```

To evaluate any saved checkpoint:

```bash
python evaluate.py --algorithm sdft --model artifacts/sdft_model \
  --backend transformers --limit 128 --output artifacts/eval/sdft.json
python evaluate.py --algorithm grpo --model artifacts/grpo_model \
  --backend transformers --limit 128 --output artifacts/eval/grpo.json
```

Evaluation reports numeric final-answer accuracy, mean response length, and truncation rate, with
raw predictions alongside the metrics. It defaults to one greedy response per prompt. With
`--samples-per-prompt N`, it samples at temperature 1 and also reports the fraction of questions
with any correct sample. These are checkpoint comparison metrics; official leaderboard protocols
may use different prompts and decoding settings. Evaluation rejects records marked as training data.

## Training controls and correctness

- Response states alone enter the output projection. Vocabulary operations are checkpointed in
  chunks of 32 tokens, retaining exact full-vocabulary JSD/reverse KL and masked cross entropy.
- Qwen's native multimodal checkpoint layout is preserved. Text training bypasses and freezes the
  vision encoder; vLLM uses `language_model_only=True` to avoid vision profiling and weights.
- Trainable weights and Adam state use FP32. CUDA computation uses BF16 autocast when supported,
  otherwise FP16 autocast with gradient scaling. This prevents small RL updates rounding away in
  BF16 parameters. Frozen reference/teacher weights use the compute dtype.
- Gradient checkpointing is on by default. SDFT/OPSD/SDPO/PPO default to one response per microbatch
  and four accumulation steps; GRPO defaults to one complete group per microbatch. Partial final
  accumulation windows are averaged correctly. Use `--no-gradient-checkpointing` to disable it.
- `--max-seq-length` bounds training input, including privileged context. Inputs exceeding the
  budget raise an error, preserving sampled tokens and their behavior probabilities. The driver
  forwards its `--max-model-len` to training. Increase the budget for longer custom data.
- PPO snapshots all old values and advantages before its first update. PPO and GRPO default to one
  policy epoch per rollout buffer; refresh trajectories through the driver for later rounds.
- PPO/GRPO generation uses temperature 1, top-p 1, no top-k truncation, and no repetition penalty.
  vLLM checkpoint sampling defaults are disabled explicitly. Both backends store the sampled token
  IDs and behavior log-probabilities; trainers reject mismatched model provenance or sampling.
- Numeric verification handles decimals, thousands separators, fractions, and common final-answer
  delimiters. Closed thinking blocks are excluded from answer extraction; unfinished thinking
  blocks do not earn reward. SDFT rejects truncated rewrites.

Tuning flags include `--gradient-accumulation-steps`, `--logit-chunk-size`, `--learning-rate`,
`--max-grad-norm`, and `--attention-implementation` on trainers. Lower chunk size to reduce vocabulary
workspace, and lower sequence/group sizes when activations dominate. Full-parameter FP32 optimizer
state still requires significant memory; chunking reduces vocabulary workspaces rather than model
and optimizer storage.

## Repository and checks

```text
self_distill_rl/
  presets.py         model, dataset, and token-budget defaults
  datasets.py        Hugging Face dataset normalization
  io.py              JSONL, chat formatting, causal masks
  rollout.py         vLLM / Transformers generation and log-probabilities
  modeling.py        model loading, chunked response projection, value head
  training.py        accumulation and rollout validation
  objectives.py      PPO/GAE and GRPO equations
  distillation.py    privileged teacher contexts
prepare_datasets.py  standard train/test data preparation
evaluate.py          held-out final-answer evaluation
on_policy_loop.py    all five pipelines with baseline/round evaluation
```

```bash
python -m unittest discover -s tests -v
python -m compileall -q self_distill_rl tests *.py
```

Tests use tiny real Qwen3.5 and LFM2 architectures without downloading weights. They compare values
and gradients against dense losses, verify teacher stop-gradient, causal alignment, behavior
log-probabilities, rewards, accumulation, and dataset/evaluation schemas.

See [concepts and equations](docs/algorithms.md), [code walkthrough](docs/code_walkthrough.md), and
the optional [veRL agentic guide](docs/agentic_verl.md).

The SDPO feedback verifier reveals the correct answer after a failed attempt, which is stronger
than ordinary tool feedback. The reward is specific to numeric math tasks; other domains require
their own verifier. Optimizer state is restarted between driver rounds; model weights and PPO's
value head are preserved. Generated data, checkpoints, and metrics are git-ignored under `artifacts/`.
