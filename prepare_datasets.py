"""Download official GSM8K/SVAMP train and test splits into the shared JSONL schema."""

import argparse
import json
from pathlib import Path

from self_distill_rl.datasets import DATASETS, load_examples
from self_distill_rl.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--output-root", default="artifacts/datasets")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--revision",
        default="main",
        help="HF revision; use a commit SHA for reproducibility",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for name in DATASETS if args.dataset == "all" else (args.dataset,):
        root = Path(args.output_root) / name
        counts = {}
        for split, limit in (("train", args.train_limit), ("test", args.eval_limit)):
            records = load_examples(name, split, limit, args.seed, args.revision)
            write_jsonl(root / f"{split}.jsonl", records)
            counts[split] = len(records)
            print(
                f"wrote {len(records)} {name}/{split} examples to {root / (split + '.jsonl')}"
            )
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": DATASETS[name][0],
                    "config": DATASETS[name][1],
                    "revision": args.revision,
                    "seed": args.seed,
                    "counts": counts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
