"""Generate on-policy trajectories with vLLM and save an auditable JSONL buffer.

The trainer consumes the sampled token IDs and their behavior-policy log-probabilities.
Keeping rollout and optimization in separate processes is slow but makes the memory
lifecycle obvious and lets both phases share one modest GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .io import chat_prompt_ids, ensure_padding_token, read_jsonl, write_jsonl
from .rewards import environment_feedback, exact_match_reward


def chosen_logprobs(
    token_ids: Sequence[int], logprob_rows: Sequence[dict[int, Any]]
) -> list[float]:
    values: list[float] = []
    if len(token_ids) != len(logprob_rows):
        raise ValueError("vLLM returned different token and log-probability lengths")
    for token_id, candidates in zip(token_ids, logprob_rows):
        candidate = candidates.get(token_id)
        if candidate is None:
            # Some vLLM versions expose integer-like keys rather than plain int keys.
            candidate = next(
                (value for key, value in candidates.items() if int(key) == token_id),
                None,
            )
        if candidate is None:
            raise KeyError(f"sampled token {token_id} missing from vLLM logprobs")
        values.append(float(candidate.logprob))
    return values


def generate_with_vllm(
    model_name_or_path: str,
    prompt_ids: Sequence[Sequence[int]],
    *,
    samples_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    seed: int,
) -> list[Any]:
    engine = LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        seed=seed,
    )
    sampling = SamplingParams(
        n=samples_per_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=1,
        seed=seed,
    )
    inputs = [{"prompt_token_ids": list(ids)} for ids in prompt_ids]
    return list(engine.generate(inputs, sampling, use_tqdm=True))


def make_rollouts(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = read_jsonl(args.prompts)
    if not examples:
        raise ValueError("prompt file is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_padding_token(tokenizer)
    prompt_ids = [
        chat_prompt_ids(tokenizer, row["prompt"], enable_thinking=args.enable_thinking)
        for row in examples
    ]
    outputs = generate_with_vllm(
        args.model,
        prompt_ids,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    records: list[dict[str, Any]] = []
    for example, expected_prompt_ids, request_output in zip(
        examples, prompt_ids, outputs
    ):
        actual_prompt_ids = list(request_output.prompt_token_ids or expected_prompt_ids)
        for sample_index, sample in enumerate(request_output.outputs):
            completion_ids = list(sample.token_ids)
            if not completion_ids:
                continue
            reward = exact_match_reward(sample.text, example["answer"])
            feedback = example.get("feedback") or environment_feedback(
                sample.text, example["answer"]
            )
            records.append(
                {
                    "id": f"{example.get('id', len(records))}:{sample_index}",
                    "group_id": str(example.get("id", example["prompt"])),
                    "prompt": example["prompt"],
                    "answer": str(example["answer"]),
                    "reference": example.get("reference", str(example["answer"])),
                    "prompt_token_ids": actual_prompt_ids,
                    "completion": sample.text,
                    "completion_token_ids": completion_ids,
                    "old_logprobs": chosen_logprobs(completion_ids, sample.logprobs),
                    "reward": reward,
                    "feedback": feedback,
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompts", default="data/toy_math.jsonl")
    parser.add_argument("--output", default="artifacts/rollouts.jsonl")
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    # PPO/GRPO recompute untempered full-softmax log-probs in PyTorch. Keep these at 1
    # for an exact behavior-policy ratio; change them only if the trainer mirrors sampling.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = make_rollouts(args)
    write_jsonl(args.output, records)
    mean_reward = sum(row["reward"] for row in records) / max(len(records), 1)
    print(
        f"wrote {len(records)} rollouts to {Path(args.output)}; mean reward={mean_reward:.3f}"
    )


if __name__ == "__main__":
    main()
