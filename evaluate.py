"""Evaluate a checkpoint on held-out math prompts without references or feedback in its context."""

import argparse
import json
from pathlib import Path
from typing import Any

from self_distill_rl.io import chat_prompt_ids, read_jsonl, write_jsonl
from self_distill_rl.modeling import load_tokenizer
from self_distill_rl.presets import PRESETS
from self_distill_rl.rewards import exact_match_reward, extract_final_answer
from self_distill_rl.rollout import generate


def score_outputs(
    examples: list[dict[str, Any]], outputs: list[Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not examples or len(examples) != len(outputs):
        raise ValueError("evaluation requires one output group for every example")
    predictions = []
    passed = 0
    for example, output in zip(examples, outputs):
        if not output.outputs:
            raise ValueError("evaluation output has no completions")
        rewards = []
        for sample_index, sample in enumerate(output.outputs):
            reward = exact_match_reward(sample.text, example["answer"])
            rewards.append(reward)
            predictions.append(
                {
                    "id": example["id"],
                    "sample_index": sample_index,
                    "prompt": example["prompt"],
                    "answer": example["answer"],
                    "completion": sample.text,
                    "predicted_answer": extract_final_answer(sample.text),
                    "reward": reward,
                    "completion_tokens": len(sample.token_ids),
                    "finish_reason": sample.finish_reason,
                }
            )
        passed += int(any(rewards))
    metrics = {
        "num_examples": len(examples),
        "num_completions": len(predictions),
        "accuracy": sum(row["reward"] for row in predictions) / len(predictions),
        "any_sample_accuracy": passed / len(examples),
        "mean_completion_tokens": sum(row["completion_tokens"] for row in predictions)
        / len(predictions),
        "truncation_rate": sum(row["finish_reason"] == "length" for row in predictions)
        / len(predictions),
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(PRESETS), default="grpo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="artifacts/eval/metrics.json")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    preset = PRESETS[args.algorithm]
    args.model = args.model or preset.model
    args.data = args.data or preset.eval_path
    args.max_tokens = (
        args.max_tokens if args.max_tokens is not None else preset.max_tokens
    )
    args.max_model_len = (
        args.max_model_len if args.max_model_len is not None else preset.max_seq_length
    )
    if (
        min(
            args.limit,
            args.samples_per_prompt,
            args.max_tokens,
            args.max_model_len,
            args.max_num_seqs,
        )
        < 1
    ):
        parser.error("limits and sample counts must be positive")
    examples = read_jsonl(args.data)[: args.limit]
    if not examples:
        raise ValueError("evaluation data is empty")
    if any(row.get("split") == "train" for row in examples):
        raise ValueError(
            "evaluation requires held-out data; the supplied file contains training examples"
        )
    tokenizer = load_tokenizer(args.model)
    prompt_ids = [
        chat_prompt_ids(tokenizer, row["prompt"], enable_thinking=args.enable_thinking)
        for row in examples
    ]
    outputs = generate(
        args.model,
        prompt_ids,
        backend=args.backend,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=0.0 if args.samples_per_prompt == 1 else 1.0,
        top_p=1.0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        require_logprobs=False,
    )
    metrics, predictions = score_outputs(examples, outputs)
    metrics.update(
        {
            "model": args.model,
            "data": args.data,
            "backend": args.backend,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "samples_per_prompt": args.samples_per_prompt,
            "enable_thinking": args.enable_thinking,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output.with_suffix(".predictions.jsonl"), predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
