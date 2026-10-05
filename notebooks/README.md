# Training recipe notebooks

Each notebook is an independent tutorial with compact code that calls the repository's shared
PyTorch trainers. Markdown explains the objective, model components, token alignment, input and
generated dataset fields, optimization, validation, and final benchmark protocol in detail.

| Notebook | Model | Training / main test | Additional test |
|---|---|---|---|
| [SDFT](01_sdft_qwen35.ipynb) | Qwen/Qwen3.5-0.8B | GSM8K | SVAMP |
| [SDPO](02_sdpo_qwen35.ipynb) | Qwen/Qwen3.5-0.8B | GSM8K | SVAMP |
| [OPSD](03_opsd_qwen35.ipynb) | Qwen/Qwen3.5-0.8B | GSM8K | SVAMP |
| [PPO](04_ppo_lfm25.ipynb) | LiquidAI/LFM2.5-350M | SVAMP | GSM8K |
| [GRPO](05_grpo_lfm25.ipynb) | LiquidAI/LFM2.5-350M | SVAMP | GSM8K |

From the repository root, using the Python environment intended for training:

```bash
python -m pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name self-distill-rl --display-name "Self-distillation RL"
python -m jupyter lab notebooks/
```

Choose the **Self-distillation RL** kernel and run cells in order. Start Jupyter in the repository
root or `notebooks/`; the setup cell locates the repository automatically. First runs download
datasets and model weights. Full-parameter training of the selected checkpoints is intended for
CUDA. The default Transformers inference backend needs only the training environment. To use vLLM,
install `requirements-rollout.txt` in a separate compatible environment and edit `BACKEND` and
`ROLLOUT_PYTHON` in the configuration cell.

The tutorial defaults train on 64 questions, reserve 16 different official training questions for
validation, and perform one round. Change these counts before starting a larger experiment. Each
notebook includes a CPU tensor demonstration of its actual loss and a masking example before the
expensive training stage. `recipe_utils.py` handles subprocesses, partitions, and reports; model
training and answer scoring remain in the shared repository modules.

Training and intermediate evaluation use only the training/validation partitions. Final cells
compare seed and trained checkpoints on both official test splits with one greedy answer per
question, no reference or feedback in the evaluation prompt, fixed budgets, and the numeric
verifier. `BENCHMARK_LIMIT=None` evaluates all available test rows; an integer requests a labeled
subset. Check available counts against the official 1,319 GSM8K and 300 SVAMP test sizes before
claiming a full test. Any exact overlap removed from the secondary benchmark is recorded.

Reports include accuracy, 95% Wilson intervals, completion length, truncation rate, paired gains and
losses, and saved per-question predictions. The notebooks explain why these measurements do not
establish reasoning correctness, freedom from pretrained benchmark exposure, or an isolated
comparison between algorithms using different model/data presets.

Artifacts go under `artifacts/notebooks/<algorithm>/`. Source data manifests, file hashes, selected
IDs, configuration, package versions, stage logs, generated buffers, checkpoints, validation
plots, and benchmark predictions are retained. Rerunning in the same directory replaces the
experiment; change `RUN_DIR` to retain another run. Committed notebooks have cleared outputs and
contain no asserted benchmark scores.

For automated execution checks, set `RL_NOTEBOOK_SMOKE=1`, `RL_NOTEBOOK_MODEL` to a compatible tiny
local Qwen3.5 or LFM2 checkpoint, and `RL_NOTEBOOK_DATA_ROOT` to a prepared toy dataset root containing
`gsm8k/{train,test}.jsonl` and `svamp/{train,test}.jsonl`. Each training file needs at least three
distinct prompts, and each test file needs a separate prompt. The smoke configuration uses two
training questions, one validation question, four generated tokens, and one test question per
dataset. `RL_NOTEBOOK_RUN_ROOT` can direct these outputs to a temporary directory. These checks
execute real training but explicitly label their results as toy smoke checks, not benchmarks.
