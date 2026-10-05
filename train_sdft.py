"""Stage 2 of SDFT: plain supervised fine-tuning on verified self-rewrites."""

from __future__ import annotations

import argparse
import random

import torch

from self_distill_rl.io import read_jsonl, sft_batch, shuffled_batches
from self_distill_rl.modeling import (
    default_device,
    load_policy,
    load_tokenizer,
    masked_mean,
    policy_logprobs,
    save_policy,
)
from self_distill_rl.presets import PRESETS
from self_distill_rl.training import (
    AccumulatingOptimizer,
    add_training_args,
    validate_training_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PRESETS["sdft"].model)
    parser.add_argument("--data", default="artifacts/sdft_data.jsonl")
    parser.add_argument("--output", default="artifacts/sdft_model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    add_training_args(parser, "sdft")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    records = read_jsonl(args.data)
    if not records:
        raise ValueError("SDFT data file is empty")
    if any(row.get("split") == "test" for row in records):
        raise ValueError("SDFT training data contains held-out test examples")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    tokenizer = load_tokenizer(args.model)
    model = load_policy(
        args.model,
        device,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_implementation=args.attention_implementation,
    )
    model.train()
    optimizer = AccumulatingOptimizer(
        model, args.learning_rate, args.gradient_accumulation_steps, args.max_grad_norm
    )

    step = 0
    for epoch in range(args.epochs):
        for rows in shuffled_batches(records, args.batch_size, args.seed + epoch):
            batch = sft_batch(
                rows, tokenizer, "selected_response", args.max_seq_length
            ).to(device)
            loss = -masked_mean(
                policy_logprobs(model, batch, args.logit_chunk_size), batch.action_mask
            )
            optimizer.backward(loss)
            step += 1
            print(f"epoch={epoch + 1} step={step} sdft_loss={loss.item():.4f}")
        optimizer.flush()

    optimizer.flush()
    save_policy(model, tokenizer, args.output)
    print(f"saved SDFT policy to {args.output}")


if __name__ == "__main__":
    main()
