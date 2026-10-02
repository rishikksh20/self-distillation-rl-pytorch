#!/usr/bin/env bash
set -euo pipefail

# Prepare veRL's GSM8K tool-agent parquet first; see docs/agentic_verl.md.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$PWD/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$PWD/data/gsm8k/test.parquet}
TOOLS_FILE=${TOOLS_FILE:-$PWD/examples/verl_agentic/arithmetic_tools.py}
N_GPUS=${N_GPUS:-1}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.return_raw_chat=True \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="$TOOLS_FILE" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  trainer.logger='["console"]' \
  trainer.project_name=self-distillation-tutorial \
  trainer.experiment_name=agentic-grpo \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False
