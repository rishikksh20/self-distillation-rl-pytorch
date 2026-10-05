"""Small orchestration helpers; training and scoring stay in the shared trainers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

from IPython.display import Markdown, display

from self_distill_rl.io import read_jsonl, write_jsonl


def run_cli(root, arguments, log_path, python=None):
    """Run one stage in a fresh process, retain its log, and show the final lines."""
    command = [str(python or sys.executable), *map(str, arguments)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", shlex.join(command), flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print("\n".join(log_path.read_text(encoding="utf-8").splitlines()[-12:]))
    print(f"Full log: {log_path}")
    result.check_returncode()
    return time.perf_counter() - started


def prepare_recipe_data(
    root, data_root, run_dir, dataset, train_limit, val_limit, seed
):
    """Reserve validation from official train; never use official test for fitting."""
    for name in ("gsm8k", "svamp"):
        if not all(
            (data_root / name / f"{split}.jsonl").exists()
            for split in ("train", "test")
        ):
            run_cli(
                root,
                [
                    "prepare_datasets.py",
                    "--dataset",
                    name,
                    "--output-root",
                    data_root,
                    "--seed",
                    seed,
                ],
                run_dir / f"prepare_{name}.log",
            )
    pool = read_jsonl(data_root / dataset / "train.jsonl")
    if any(row.get("split") != "train" for row in pool):
        raise ValueError("The training pool must contain only official train records")
    if len({row["prompt"] for row in pool}) != len(pool):
        raise ValueError("Deduplicate training prompts before partitioning")
    if min(train_limit, val_limit) < 1 or train_limit + val_limit > len(pool):
        raise ValueError(
            "Need enough train records for disjoint training and validation"
        )
    random.Random(seed).shuffle(pool)
    train = pool[:train_limit]
    validation = [
        {**row, "split": "validation", "source_split": "train"}
        for row in pool[train_limit : train_limit + val_limit]
    ]
    paths = {
        "train": run_dir / "train.jsonl",
        "validation": run_dir / "validation.jsonl",
    }
    write_jsonl(paths["train"], train)
    write_jsonl(paths["validation"], validation)
    excluded = {row["prompt"] for row in train + validation}
    provenance = {
        "seed": seed,
        "training_ids": [row["id"] for row in train],
        "validation_ids": [row["id"] for row in validation],
        "datasets": {},
    }
    for name in ("gsm8k", "svamp"):
        source = data_root / name / "test.jsonl"
        rows = read_jsonl(source)
        if not rows or any(row.get("split") != "test" for row in rows):
            raise ValueError("Benchmark input must contain only official test records")
        filtered = [row for row in rows if row["prompt"] not in excluded]
        if name == dataset and len(filtered) != len(rows):
            raise ValueError("Official training and test prompts overlap")
        if not filtered:
            raise ValueError("No disjoint benchmark records remain")
        paths[name] = run_dir / f"{name}_test.jsonl"
        write_jsonl(paths[name], filtered)
        manifest = data_root / name / "manifest.json"
        provenance["datasets"][name] = {
            "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
            "train_sha256": hashlib.sha256(
                (data_root / name / "train.jsonl").read_bytes()
            ).hexdigest(),
            "test_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "test_count": len(filtered),
            "excluded_overlap": len(rows) - len(filtered),
        }
    (run_dir / "data_protocol.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return paths


def evaluate_models(
    root, run_dir, algorithm, models, test_paths, generation_args, python, limit=None
):
    """Compare seed and trained policy using the identical greedy test protocol."""
    results = {}
    for dataset, data_path in test_paths.items():
        rows = read_jsonl(data_path)
        count = len(rows) if limit is None else min(limit, len(rows))
        for label, model in models.items():
            output = run_dir / f"benchmark_{dataset}_{label}.json"
            seconds = run_cli(
                root,
                [
                    "evaluate.py",
                    "--algorithm",
                    algorithm,
                    "--model",
                    model,
                    "--data",
                    data_path,
                    "--output",
                    output,
                    "--limit",
                    count,
                    "--samples-per-prompt",
                    1,
                    *generation_args,
                ],
                output.with_suffix(".log"),
                python=python,
            )
            metrics = json.loads(output.read_text())
            metrics.update(
                dataset=dataset,
                label=label,
                elapsed_seconds=seconds,
                scope="full available test" if limit is None else "test subset",
            )
            output.write_text(json.dumps(metrics, indent=2) + "\n")
            results[(dataset, label)] = metrics
    return results


def wilson_interval(accuracy, count, z=1.96):
    """Approximate 95% binomial interval for one greedy answer per question."""
    center = (accuracy + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * (
        (accuracy * (1 - accuracy) / count + z * z / (4 * count * count)) ** 0.5
    )
    radius /= 1 + z * z / count
    return max(0, center - radius), min(1, center + radius)


def report_benchmarks(results, run_dir):
    lines = [
        "| Dataset | Model | N | Accuracy | 95% Wilson interval | Tokens | Truncated | Scope |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for (dataset, label), metrics in results.items():
        lo, hi = wilson_interval(metrics["accuracy"], metrics["num_examples"])
        lines.append(
            f"| {dataset} | {label} | {metrics['num_examples']} | "
            f"{metrics['accuracy']:.1%} | {lo:.1%}–{hi:.1%} | "
            f"{metrics['mean_completion_tokens']:.1f} | "
            f"{metrics['truncation_rate']:.1%} | {metrics['scope']} |"
        )
    display(Markdown("\n".join(lines)))
    paired = {}
    for dataset in dict.fromkeys(key[0] for key in results):
        predictions = {
            label: {
                row["id"]: row["reward"]
                for row in read_jsonl(
                    run_dir / f"benchmark_{dataset}_{label}.predictions.jsonl"
                )
            }
            for label in ("seed", "trained")
        }
        before, after = predictions["seed"], predictions["trained"]
        if before.keys() != after.keys():
            raise ValueError("Paired comparisons require the same question IDs")
        gained = sum(before[key] == 0 and after[key] == 1 for key in before)
        lost = sum(before[key] == 1 and after[key] == 0 for key in before)
        paired[dataset] = {
            "n": len(before),
            "gained": gained,
            "lost": lost,
            "accuracy_delta": (gained - lost) / len(before),
        }
        print(
            f"{dataset}: {gained} gained, {lost} lost; "
            f"accuracy change = {(gained - lost) / len(before):+.1%}"
        )
    (run_dir / "paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")
    return paired
