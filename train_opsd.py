"""On-Policy Self-Distillation with a fixed, privileged-information self-teacher."""

from __future__ import annotations

import argparse
import random

import torch

from self_distill_rl.distillation import teacher_batch
from self_distill_rl.io import read_jsonl, rollout_batch, shuffled_batches
from self_distill_rl.modeling import (
    completion_logits,
    default_device,
    distillation_loss,
    load_policy,
    load_tokenizer,
    save_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--teacher-model",
        default=None,
        help="Frozen initial checkpoint. Defaults to --model for a one-round run.",
    )
    parser.add_argument("--rollouts", default="artifacts/rollouts.jsonl")
    parser.add_argument("--output", default="artifacts/opsd_model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--divergence", choices=("jsd", "reverse_kl"), default="jsd")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    tokenizer = load_tokenizer(args.model)
    student = load_policy(
        args.model, device, gradient_checkpointing=args.gradient_checkpointing
    )
    teacher = load_policy(args.teacher_model or args.model, device)
    teacher.eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    records = read_jsonl(args.rollouts)
    for row in records:
        if not row.get("reference"):
            raise ValueError(
                "OPSD requires a non-empty 'reference' solution in every rollout"
            )

    step = 0
    for epoch in range(args.epochs):
        for rows in shuffled_batches(records, args.batch_size, args.seed + epoch):
            student_inputs = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
            teacher_inputs = teacher_batch(rows, tokenizer, "opsd").to(device)
            with torch.no_grad():
                teacher_output = teacher(
                    input_ids=teacher_inputs.input_ids,
                    attention_mask=teacher_inputs.attention_mask,
                )
                teacher_rows = completion_logits(teacher_output.logits, teacher_inputs)

            student.train()
            student_output = student(
                input_ids=student_inputs.input_ids,
                attention_mask=student_inputs.attention_mask,
            )
            student_rows = completion_logits(student_output.logits, student_inputs)
            loss = distillation_loss(student_rows, teacher_rows, args.divergence)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            print(
                f"epoch={epoch + 1} step={step} opsd_{args.divergence}={loss.item():.5f}"
            )

    save_policy(student, tokenizer, args.output)
    print(f"saved OPSD policy to {args.output}")


if __name__ == "__main__":
    main()
