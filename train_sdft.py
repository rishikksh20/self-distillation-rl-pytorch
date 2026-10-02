"""Stage 2 of SDFT: plain supervised fine-tuning on verified self-rewrites."""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from self_distill_rl.io import read_jsonl, sft_batch, shuffled_batches
from self_distill_rl.modeling import (
    default_device,
    load_policy,
    load_tokenizer,
    masked_mean,
    save_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="artifacts/sdft_data.jsonl")
    parser.add_argument("--output", default="artifacts/sdft_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
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
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    records = read_jsonl(args.data)

    step = 0
    for epoch in range(args.epochs):
        for rows in shuffled_batches(records, args.batch_size, args.seed + epoch):
            batch = sft_batch(rows, tokenizer, "selected_response").to(device)
            output = model(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask
            )
            labels = batch.input_ids[:, 1:]
            per_token_loss = F.cross_entropy(
                output.logits[:, :-1, :].float().transpose(1, 2),
                labels,
                reduction="none",
            )
            loss = masked_mean(per_token_loss, batch.action_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            step += 1
            print(f"epoch={epoch + 1} step={step} sdft_loss={loss.item():.4f}")

    save_policy(model, tokenizer, args.output)
    print(f"saved SDFT policy to {args.output}")


if __name__ == "__main__":
    main()
