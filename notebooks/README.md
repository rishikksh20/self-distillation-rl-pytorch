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

## Standalone Google Colab editions

These alternatives embed every required function, including data normalization, generation,
token masking, model loading, optimizer, method-specific training, and benchmark reporting:

| Method | Standalone notebook |
|---|---|
| SDFT | [01_sdft_qwen35_standalone.ipynb](01_sdft_qwen35_standalone.ipynb) |
| SDPO | [02_sdpo_qwen35_standalone.ipynb](02_sdpo_qwen35_standalone.ipynb) |
| OPSD | [03_opsd_qwen35_standalone.ipynb](03_opsd_qwen35_standalone.ipynb) |
| PPO | [04_ppo_lfm25_standalone.ipynb](04_ppo_lfm25_standalone.ipynb) |
| GRPO | [05_grpo_lfm25_standalone.ipynb](05_grpo_lfm25_standalone.ipynb) |

Upload a single file using **Colab → File → Upload notebook**, choose a GPU runtime, and run cells
in order. The first code cell lists the complete external dependencies, writes a requirements
file locally, and installs those libraries. No checkout, repository imports, helper files, or
external requirements file are needed. The training and evaluation loops are visible notebook code.

All five standalone notebooks support math, Python coding, and single-turn tool use. Select
`DOMAINS = "all"`, one domain such as `["coding"]`, or a combination such as
`["coding", "tool_use"]`. `TRAIN_SOURCES` selects source datasets; `TRAIN_LIMITS` controls
the mixture with a count per domain. The default fits 16 training / 8 validation questions
per domain, uses 256 generated tokens, and evaluates a labeled 32-question subset per benchmark.

| Domain | Training | Validation | Final before/after tests |
|---|---|---|---|
| Math | GSM8K or SVAMP official train | Reserved train questions | GSM8K and SVAMP official tests |
| Coding | MBPP `full/train`, with reference test checks | MBPP `full/validation` | MBPP test and HumanEval test |
| Tool use | Hermes single-turn; optional authenticated xLAM | Reserved source questions | Separate source holdout and BFCL simple calls |

Selected domains get separate training/validation JSONL files and a shuffled training mixture.
Verifiers dispatch by domain: numeric equality, Python unit tests, or tool schemas and argument
matching. SDPO feedback comes from the policy's actual new attempt. Multi-turn agent datasets
are excluded because these notebooks train single-response tasks without an environment loop.
Each standalone recipe also explains and checks how its selected datasets supply the actual
objective: SDFT targets, SDPO feedback, OPSD private references, or PPO/GRPO verifier rewards.
`dataset_recipe.json` and the pre-training table record the sources that actually contributed
sampled tasks. Buffer checks preserve each selected task's public prompt, domain, and verifier,
reject held-out examples and missing objective fields, and enforce complete GRPO groups.

Final reports compare the original pretrained checkpoint and final policy on identical question
IDs, prompts, decoding budgets, and verifiers. They include domain scores, paired gains/losses,
Wilson intervals, token/truncation diagnostics, a before/after plot, and saved predictions.
Coding runs in a restricted child process; its instruction-style execution protocol and the
local BFCL JSON argument matcher are explicitly labeled adaptations, not official leaderboard
scores. Tool calls are checked against references, not executed against live APIs.

Set `BENCHMARK_LIMIT=None` to evaluate every prepared test row. The protocol manifest records
source, normalization, overlap, reference-check, and context exclusions; a full prepared test
can differ from the original publisher's full split. The tool source holdout is explicitly
identified as reserved training data. Optional xLAM requires accepting its HF terms and setting
`HF_TOKEN`; Hermes is ungated. Dependencies for every adapter/evaluator are in the install cell.

OPSD uses a CPU teacher by default to reduce GPU memory use; a GPU
teacher can be selected when memory permits. Colab hardware and session limits vary, so the
notebooks report the actual device and explain full-parameter memory requirements.

Outputs default to `/content/rl_tutorials/<method>/<domains>_<config-digest>/`. Set the output root
to persistent storage or use
the optional results-download cell to retain them. Checkpoints are excluded from the ZIP by
default because they can be large; a setting includes them when requested.

An optional `STANDALONE_SMOKE=1` mode creates its own tiny random Qwen/LFM architecture,
tokenizer, and toy math/coding/tool data within the notebook, then executes two training rounds offline. No
repository or external fixture is required. `STANDALONE_SKIP_INSTALL=1` is only for validation
in an environment that already has the listed packages. These checks are labeled as smoke
checks and make no claims about pretrained-model benchmark accuracy.

## Repository-backed editions

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
