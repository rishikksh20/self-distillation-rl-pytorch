"""Stage 1 of SDFT: ask the seed model to rewrite targets in its own style."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from self_distill_rl.io import (
    chat_prompt_ids,
    ensure_padding_token,
    read_jsonl,
    write_jsonl,
)
from self_distill_rl.rewards import extract_final_answer, normalize_answer
from self_distill_rl.rollout import generate_with_vllm


REWRITE_TEMPLATE = """Rewrite the reference response in your own natural style.
Keep its meaning and final answer unchanged. Return only the rewritten response.

Task:
{prompt}

Reference response:
{response}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="data/toy_math.jsonl")
    parser.add_argument("--output", default="artifacts/sdft_data.jsonl")
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_padding_token(tokenizer)
    rewrite_prompts = [
        REWRITE_TEMPLATE.format(prompt=row["prompt"], response=row["reference"])
        for row in records
    ]
    prompt_ids = [chat_prompt_ids(tokenizer, prompt) for prompt in rewrite_prompts]
    outputs = generate_with_vllm(
        args.model,
        prompt_ids,
        samples_per_prompt=1,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    distilled: list[dict[str, Any]] = []
    accepted = 0
    for row, output in zip(records, outputs):
        rewritten = output.outputs[0].text.strip()
        expected = normalize_answer(row["answer"])
        verified = extract_final_answer(rewritten) == expected
        selected = rewritten if verified else row["reference"]
        accepted += int(verified)
        distilled.append(
            {
                **row,
                "rewritten_response": rewritten,
                "selected_response": selected,
                "rewrite_verified": verified,
            }
        )
    write_jsonl(args.output, distilled)
    print(
        f"wrote {len(distilled)} examples to {Path(args.output)}; accepted rewrites={accepted}"
    )


if __name__ == "__main__":
    main()
