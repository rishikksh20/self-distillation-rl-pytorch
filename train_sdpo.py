"""Self-Distillation Policy Optimization using tokenized environment feedback."""

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
    parser.add_argument("--rollouts", default="artifacts/rollouts.jsonl")
    parser.add_argument("--output", default="artifacts/sdpo_model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
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
    model = load_policy(
        args.model, device, gradient_checkpointing=args.gradient_checkpointing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    records = read_jsonl(args.rollouts)
    for row in records:
        if not row.get("feedback"):
            raise ValueError(
                "SDPO requires non-empty tokenized 'feedback' in every rollout"
            )

    step = 0
    for epoch in range(args.epochs):
        for rows in shuffled_batches(records, args.batch_size, args.seed + epoch):
            student_inputs = rollout_batch(rows, int(tokenizer.pad_token_id)).to(device)
            hindsight_inputs = teacher_batch(rows, tokenizer, "sdpo").to(device)

            # Same weights, different context. no_grad is the stop-gradient teacher.
            model.eval()
            with torch.no_grad():
                teacher_output = model(
                    input_ids=hindsight_inputs.input_ids,
                    attention_mask=hindsight_inputs.attention_mask,
                )
                teacher_rows = completion_logits(
                    teacher_output.logits, hindsight_inputs
                )

            model.train()
            student_output = model(
                input_ids=student_inputs.input_ids,
                attention_mask=student_inputs.attention_mask,
            )
            student_rows = completion_logits(student_output.logits, student_inputs)
            # The SDPO paper uses KL(student || stopgrad(feedback-conditioned teacher)).
            loss = distillation_loss(student_rows, teacher_rows, "reverse_kl")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            print(f"epoch={epoch + 1} step={step} sdpo_reverse_kl={loss.item():.5f}")

    save_policy(model, tokenizer, args.output)
    print(f"saved SDPO policy to {args.output}")


if __name__ == "__main__":
    main()
