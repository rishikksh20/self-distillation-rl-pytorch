"""Alternate vLLM rollout and PyTorch optimization in separate single-GPU processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from self_distill_rl.io import read_jsonl
from self_distill_rl.presets import PRESETS

TRAINERS = {
    "sdft": "train_sdft.py",
    "ppo": "train_ppo.py",
    "grpo": "train_grpo.py",
    "opsd": "train_opsd.py",
    "sdpo": "train_sdpo.py",
}


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent)
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    subprocess.run(command, check=True, env=environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--eval-prompts", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--output-root", default="artifacts/on_policy")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--samples-per-prompt", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument(
        "--rollout-python",
        default=sys.executable,
        help="Python executable in the inference environment (also used for eval)",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    preset = PRESETS[args.algorithm]
    for name, value in (
        ("model", preset.model),
        ("prompts", preset.train_path),
        ("eval_prompts", preset.eval_path),
        ("max_tokens", preset.max_tokens),
        ("max_model_len", preset.max_seq_length),
        ("samples_per_prompt", preset.samples_per_prompt),
    ):
        if getattr(args, name) is None:
            setattr(args, name, value)
    if (
        min(
            args.rounds,
            args.train_limit,
            args.eval_limit,
            args.max_tokens,
            args.max_model_len,
            args.max_num_seqs,
            args.samples_per_prompt,
            args.gradient_accumulation_steps,
            args.logit_chunk_size,
        )
        < 1
    ):
        parser.error("rounds, limits, and sample counts must be positive")
    if args.algorithm == "grpo" and args.samples_per_prompt < 2:
        parser.error("GRPO needs at least two samples per prompt")
    return args


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    preset = PRESETS[args.algorithm]
    if (args.prompts == preset.train_path and not Path(args.prompts).exists()) or (
        not args.skip_eval
        and args.eval_prompts == preset.eval_path
        and not Path(args.eval_prompts).exists()
    ):
        run(
            [
                sys.executable,
                str(script_root / "prepare_datasets.py"),
                "--dataset",
                preset.dataset,
                "--seed",
                str(args.seed),
            ]
        )
    root = Path(args.output_root) / args.algorithm
    root.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(args.prompts)[: args.train_limit]
    if not examples or any(row.get("split") == "test" for row in examples):
        raise ValueError(
            "training input must be nonempty and must not contain test examples"
        )
    from self_distill_rl.io import write_jsonl

    training_prompts = root / "training_prompts.jsonl"
    write_jsonl(training_prompts, examples)
    if not args.skip_eval:
        evaluation_examples = read_jsonl(args.eval_prompts)[: args.eval_limit]
        if {row["prompt"] for row in examples} & {
            row["prompt"] for row in evaluation_examples
        }:
            raise ValueError("training and evaluation prompts overlap")
    initial_model = args.model
    current_model = args.model

    def evaluate(model: str | Path, label: str) -> None:
        if args.skip_eval:
            return
        run(
            [
                args.rollout_python,
                str(script_root / "evaluate.py"),
                "--algorithm",
                args.algorithm,
                "--model",
                str(model),
                "--data",
                args.eval_prompts,
                "--limit",
                str(args.eval_limit),
                "--output",
                str(root / f"{label}_eval.json"),
                "--backend",
                args.backend,
                "--max-tokens",
                str(args.max_tokens),
                "--max-model-len",
                str(args.max_model_len),
                "--max-num-seqs",
                str(args.max_num_seqs),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--seed",
                str(args.seed),
            ]
        )

    evaluate(initial_model, "baseline")

    for round_index in range(1, args.rounds + 1):
        rollouts = root / f"round_{round_index:02d}_rollouts.jsonl"
        checkpoint = root / f"round_{round_index:02d}_model"
        if args.algorithm == "sdft":
            rollouts = root / f"round_{round_index:02d}_sdft.jsonl"
            run(
                [
                    args.rollout_python,
                    str(script_root / "generate_sdft_data.py"),
                    "--model",
                    str(current_model),
                    "--data",
                    str(training_prompts),
                    "--output",
                    str(rollouts),
                    "--backend",
                    args.backend,
                    "--max-tokens",
                    str(args.max_tokens),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-num-seqs",
                    str(args.max_num_seqs),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--seed",
                    str(args.seed + round_index - 1),
                ]
            )
        else:
            run(
                [
                    args.rollout_python,
                    "-m",
                    "self_distill_rl.rollout",
                    "--model",
                    str(current_model),
                    "--algorithm",
                    args.algorithm,
                    "--prompts",
                    str(training_prompts),
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
                    "--backend",
                    args.backend,
                    "--max-model-len",
                    str(args.max_model_len),
                    "--max-num-seqs",
                    str(args.max_num_seqs),
                ]
            )

        train_command = [
            sys.executable,
            str(script_root / TRAINERS[args.algorithm]),
            "--model",
            str(current_model),
            "--data" if args.algorithm == "sdft" else "--rollouts",
            str(rollouts),
            "--output",
            str(checkpoint),
            "--seed",
            str(args.seed + round_index - 1),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--max-seq-length",
            str(args.max_model_len),
            "--logit-chunk-size",
            str(args.logit_chunk_size),
        ]
        if args.learning_rate is not None:
            train_command.extend(["--learning-rate", str(args.learning_rate)])
        if args.algorithm == "grpo":
            train_command.extend(["--reference-model", initial_model])
        elif args.algorithm == "opsd":
            train_command.extend(["--teacher-model", initial_model])
        run(train_command)
        current_model = checkpoint
        evaluate(current_model, f"round_{round_index:02d}")

    print(f"\nfinished {args.rounds} rounds; final checkpoint: {current_model}")


if __name__ == "__main__":
    main()
