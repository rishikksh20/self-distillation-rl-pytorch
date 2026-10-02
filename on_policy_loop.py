"""Alternate vLLM rollout and PyTorch optimization in separate single-GPU processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TRAINERS = {
    "ppo": "train_ppo.py",
    "grpo": "train_grpo.py",
    "opsd": "train_opsd.py",
    "sdpo": "train_sdpo.py",
}


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompts", default="data/toy_math.jsonl")
    parser.add_argument("--output-root", default="artifacts/on_policy")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root) / args.algorithm
    root.mkdir(parents=True, exist_ok=True)
    initial_model = args.model
    current_model = args.model

    for round_index in range(1, args.rounds + 1):
        rollouts = root / f"round_{round_index:02d}_rollouts.jsonl"
        checkpoint = root / f"round_{round_index:02d}_model"
        run(
            [
                sys.executable,
                "-m",
                "self_distill_rl.rollout",
                "--model",
                str(current_model),
                "--prompts",
                args.prompts,
                "--output",
                str(rollouts),
                "--samples-per-prompt",
                str(args.samples_per_prompt),
                "--max-tokens",
                str(args.max_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--seed",
                str(args.seed + round_index - 1),
            ]
        )

        train_command = [
            sys.executable,
            TRAINERS[args.algorithm],
            "--model",
            str(current_model),
            "--rollouts",
            str(rollouts),
            "--output",
            str(checkpoint),
            "--seed",
            str(args.seed + round_index - 1),
        ]
        if args.algorithm == "grpo":
            train_command.extend(["--reference-model", initial_model])
        elif args.algorithm == "opsd":
            train_command.extend(["--teacher-model", initial_model])
        run(train_command)
        current_model = checkpoint

    print(f"\nfinished {args.rounds} rounds; final checkpoint: {current_model}")


if __name__ == "__main__":
    main()
