# Agentic environments with veRL

The vanilla scripts are intentionally single-turn. A tool-using agent introduces asynchronous tool
latency, multiple assistant turns, per-turn loss masks, environment lifecycle management, and weight
synchronization with the inference engine. Those are systems problems, so the repository includes a
veRL path instead of recreating them in a tutorial loop.

veRL's agent loop supports asynchronous rollout servers, vLLM, custom multi-turn loops, and tools.
The included example registers a safe arithmetic function tool in
[`examples/verl_agentic/arithmetic_tools.py`](../examples/verl_agentic/arithmetic_tools.py) and
provides a compact GRPO launch template in
[`examples/verl_agentic/run_agentic_grpo.sh`](../examples/verl_agentic/run_agentic_grpo.sh).

## Setup

Install veRL in a separate environment using its current official instructions. Its vLLM, PyTorch,
Ray, and CUDA pins move together, so adding veRL to this repository's small `requirements.txt` would
make the basic examples unnecessarily fragile. Recent veRL releases use `uv` extras, for example:

```bash
git clone https://github.com/volcengine/verl.git
cd verl
uv sync --extra vllm --extra fsdp
```

Use the exact extras documented by the veRL revision you checkout.

## Prepare agent data

veRL expects Parquet records with raw chat turns and an `agent_name`. Its maintained GSM8K tool-loop
preprocessor is a convenient smoke test:

```bash
cd /path/to/verl
uv run --extra vllm --extra fsdp \
  python examples/data_preprocess/gsm8k_tool_agent_loop.py
```

Locate the resulting train and test Parquet files, then return to this repository and set
`TRAIN_FILE` and `VAL_FILE`. The exact default output location can change between veRL versions.

## Run

Inside the veRL environment:

```bash
TRAIN_FILE=/absolute/path/to/train.parquet \
VAL_FILE=/absolute/path/to/test.parquet \
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
bash examples/verl_agentic/run_agentic_grpo.sh
```

The important settings in the template are:

```text
data.return_raw_chat=True
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.function_tool_path=...
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
```

The model's chat template must support the selected tool-call format (`hermes` in the example). Start
with very small token limits and one GPU. If Ray workers or vLLM run out of memory, lower rollout GPU
utilization and group size before increasing CPU offload.

## When to use which path

Use this repository's vanilla loop when you are studying loss equations, checking a new reward, or
debugging one-turn behavior. Use veRL when trajectories call tools, wait on an environment, span
multiple turns, or require efficient weight synchronization. The learning algorithm may still be
GRPO or PPO; veRL supplies the distributed rollout and environment machinery.

Primary documentation:

- [veRL agentic RL training](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)
- [veRL agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [veRL multi-turn tool configuration](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
