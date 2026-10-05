"""On-Policy Self-Distillation with a fixed, privileged-information self-teacher."""

from __future__ import annotations

import argparse
import random

import torch

from self_distill_rl.distillation import teacher_batch
from self_distill_rl.io import read_jsonl, rollout_batch, shuffled_batches
from self_distill_rl.modeling import (
    completion_hidden_states,
    default_device,
    hidden_distillation_loss,
    load_policy,
    load_tokenizer,
    require_matching_tokenizers,
    save_policy,
)
from self_distill_rl.presets import PRESETS
from self_distill_rl.training import (
    AccumulatingOptimizer,
    add_training_args,
    validate_rollouts,
    validate_training_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRESETS["opsd"].model)
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Frozen initial checkpoint. Defaults to --model for a one-round run.",
    )
    parser.add_argument("--rollouts", default=PRESETS["opsd"].rollout_path)
    parser.add_argument("--output", default="artifacts/opsd_model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--divergence", choices=("jsd", "reverse_kl"), default="jsd")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    add_training_args(parser, "opsd")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    records = read_jsonl(args.rollouts)
    validate_rollouts(
        records, args.model, policy_gradient=False, max_seq_length=args.max_seq_length
    )
    for row in records:
        if not row.get("reference"):
            raise ValueError(
                "OPSD requires a non-empty 'reference' solution in every rollout"
            )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    tokenizer = load_tokenizer(args.model)
    student = load_policy(
        args.model,
        device,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_implementation=args.attention_implementation,
    )
    require_matching_tokenizers(tokenizer, args.teacher_model or args.model)
    teacher = load_policy(
        args.teacher_model or args.model,
        device,
        attention_implementation=args.attention_implementation,
        trainable=False,
    )
    teacher.eval()
    teacher.requires_grad_(False)
    optimizer = AccumulatingOptimizer(
        student,
        args.learning_rate,
        args.gradient_accumulation_steps,
        args.max_grad_norm,
    )

    step = 0
    for epoch in range(args.epochs):
        for rows in shuffled_batches(records, args.batch_size, args.seed + epoch):
            student_inputs = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
            teacher_inputs = teacher_batch(
                rows, tokenizer, "opsd", args.max_seq_length
            ).to(device)
            with torch.no_grad():
                teacher_rows = completion_hidden_states(teacher, teacher_inputs)

            student.train()
            student_rows = completion_hidden_states(student, student_inputs)
            loss = hidden_distillation_loss(
                student,
                teacher,
                student_rows,
                teacher_rows,
                args.divergence,
                args.logit_chunk_size,
            )
            optimizer.backward(loss)
            step += 1
            print(
                f"epoch={epoch + 1} step={step} opsd_{args.divergence}={loss.item():.5f}"
            )
        optimizer.flush()

    optimizer.flush()
    save_policy(student, tokenizer, args.output)
    print(f"saved OPSD policy to {args.output}")


if __name__ == "__main__":
    main()
