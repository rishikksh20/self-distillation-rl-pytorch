"""Generate on-policy trajectories with vLLM and save an auditable JSONL buffer.

The trainer consumes the sampled token IDs and their behavior-policy log-probabilities.
Keeping rollout and optimization in separate processes is slow but makes the memory
lifecycle obvious and lets both phases share one modest GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from .io import chat_prompt_ids, read_jsonl, write_jsonl
from .modeling import default_device, load_tokenizer, training_dtype
from .presets import PRESETS
from .rewards import environment_feedback, exact_match_reward


def chosen_logprobs(
    token_ids: Sequence[int], logprob_rows: Sequence[dict[int, Any]]
) -> list[float]:
    values: list[float] = []
    if len(token_ids) != len(logprob_rows):
        raise ValueError("vLLM returned different token and log-probability lengths")
    for token_id, candidates in zip(token_ids, logprob_rows):
        candidate = candidates.get(token_id)
        if candidate is None:
            # Some vLLM versions expose integer-like keys rather than plain int keys.
            candidate = next(
                (value for key, value in candidates.items() if int(key) == token_id),
                None,
            )
        if candidate is None:
            raise KeyError(f"sampled token {token_id} missing from vLLM logprobs")
        values.append(float(candidate.logprob))
    return values


def generate_with_vllm(
    model_name_or_path: str,
    prompt_ids: Sequence[Sequence[int]],
    *,
    samples_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    seed: int,
    max_model_len: int = 2048,
    max_num_seqs: int = 4,
    require_logprobs: bool = True,
) -> list[Any]:
    from vllm import LLM, SamplingParams

    if any(len(ids) + max_tokens > max_model_len for ids in prompt_ids):
        raise ValueError(
            "prompt + generation budget exceeds --max-model-len; increase it or shorten the input"
        )
    config = AutoConfig.from_pretrained(model_name_or_path)
    engine = LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=False,
        seed=seed,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        language_model_only=config.model_type == "qwen3_5",
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        dtype=str(training_dtype(default_device())).removeprefix("torch."),
    )
    sampling = SamplingParams(
        n=samples_per_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=-1,
        logprobs=1 if require_logprobs else None,
        seed=seed,
    )
    inputs = [{"prompt_token_ids": list(ids)} for ids in prompt_ids]
    return list(engine.generate(inputs, sampling, use_tqdm=True))


def generate_with_transformers(
    model_name_or_path: str,
    prompt_ids: Sequence[Sequence[int]],
    *,
    samples_per_prompt: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_model_len: int = 2048,
    require_logprobs: bool = True,
    **_: Any,
) -> list[Any]:
    """Portable fallback; generate serially to keep memory bounded on CPU/small GPUs."""
    from types import SimpleNamespace

    import torch
    from transformers import GenerationConfig

    from .modeling import default_device, load_policy

    if any(len(ids) + max_tokens > max_model_len for ids in prompt_ids):
        raise ValueError("prompt + generation budget exceeds --max-model-len")
    torch.manual_seed(seed)
    device = default_device()
    model = load_policy(model_name_or_path, device, trainable=False).eval()
    tokenizer = load_tokenizer(model_name_or_path)
    generation = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p,
        top_k=0 if temperature > 0 else None,
        repetition_penalty=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        return_dict_in_generate=True,
        output_scores=require_logprobs,
        use_cache=True,
    )
    outputs = []
    with torch.inference_mode():
        for ids in prompt_ids:
            samples = []
            for _ in range(samples_per_prompt):
                inputs = torch.tensor([ids], device=device)
                result = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation,
                )
                tokens = result.sequences[0, len(ids) :].tolist()
                logprobs = None
                if require_logprobs:
                    transitions = model.compute_transition_scores(
                        result.sequences, result.scores, normalize_logits=True
                    )[0].tolist()
                    logprobs = [
                        {token: SimpleNamespace(logprob=value)}
                        for token, value in zip(tokens, transitions)
                    ]
                eos = generation.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                samples.append(
                    SimpleNamespace(
                        text=tokenizer.decode(tokens, skip_special_tokens=True),
                        token_ids=tokens,
                        logprobs=logprobs,
                        finish_reason="stop"
                        if tokens and tokens[-1] in eos_ids
                        else "length",
                    )
                )
            outputs.append(SimpleNamespace(prompt_token_ids=list(ids), outputs=samples))
    return outputs


def generate(
    model_name_or_path: str,
    prompt_ids: Sequence[Sequence[int]],
    *,
    backend: str = "vllm",
    **kwargs: Any,
) -> list[Any]:
    if (
        not prompt_ids
        or min(
            kwargs.get("samples_per_prompt", 1),
            kwargs.get("max_tokens", 1),
            kwargs.get("max_model_len", 2048),
            kwargs.get("max_num_seqs", 4),
        )
        < 1
    ):
        raise ValueError(
            "generation needs nonempty prompts and positive token/sample limits"
        )
    if kwargs.get("temperature", 1.0) < 0 or not 0 < kwargs.get("top_p", 1.0) <= 1:
        raise ValueError("temperature must be nonnegative and top_p must be in (0, 1]")
    if backend not in ("vllm", "transformers"):
        raise ValueError(f"unknown generation backend: {backend}")
    return (generate_with_vllm if backend == "vllm" else generate_with_transformers)(
        model_name_or_path, prompt_ids, **kwargs
    )


def make_rollouts(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = read_jsonl(args.prompts)
    if not examples:
        raise ValueError("prompt file is empty")
    tokenizer = load_tokenizer(args.model)
    prompt_ids = [
        chat_prompt_ids(tokenizer, row["prompt"], enable_thinking=args.enable_thinking)
        for row in examples
    ]
    outputs = generate(
        args.model,
        prompt_ids,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        backend=args.backend,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    if len(outputs) != len(examples):
        raise ValueError("generation returned an incomplete prompt batch")

    records: list[dict[str, Any]] = []
    for example, expected_prompt_ids, request_output in zip(
        examples, prompt_ids, outputs
    ):
        actual_prompt_ids = list(request_output.prompt_token_ids or expected_prompt_ids)
        for sample_index, sample in enumerate(request_output.outputs):
            completion_ids = list(sample.token_ids)
            if not completion_ids:
                continue
            reward = exact_match_reward(sample.text, example["answer"])
            feedback = example.get("feedback") or environment_feedback(
                sample.text, example["answer"]
            )
            records.append(
                {
                    "id": f"{example.get('id', len(records))}:{sample_index}",
                    "group_id": str(example.get("id", example["prompt"])),
                    "prompt": example["prompt"],
                    "answer": str(example["answer"]),
                    "reference": example.get("reference", str(example["answer"])),
                    "prompt_token_ids": actual_prompt_ids,
                    "completion": sample.text,
                    "completion_token_ids": completion_ids,
                    "old_logprobs": chosen_logprobs(completion_ids, sample.logprobs),
                    "reward": reward,
                    "feedback": feedback,
                    "model": str(args.model),
                    "sampling": {"temperature": args.temperature, "top_p": args.top_p},
                    "enable_thinking": args.enable_thinking,
                    "finish_reason": sample.finish_reason,
                    "dataset": example.get("dataset", "local"),
                    "split": example.get("split", "unknown"),
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=tuple(PRESETS), default="grpo")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    # PPO/GRPO recompute untempered full-softmax log-probs in PyTorch. Keep these at 1
    # for an exact behavior-policy ratio; change them only if the trainer mirrors sampling.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()
    preset = PRESETS[args.algorithm]
    for key, value in (
        ("model", preset.model),
        ("prompts", preset.train_path),
        ("max_tokens", preset.max_tokens),
        ("max_model_len", preset.max_seq_length),
        ("samples_per_prompt", preset.samples_per_prompt),
        ("output", preset.rollout_path),
    ):
        if getattr(args, key) is None:
            setattr(args, key, value)
    if (
        min(
            args.max_tokens,
            args.max_model_len,
            args.max_num_seqs,
            args.samples_per_prompt,
        )
        < 1
    ):
        parser.error("token limits and sample counts must be positive")
    if args.algorithm in ("ppo", "grpo") and (args.temperature != 1 or args.top_p != 1):
        parser.error(
            "PPO/GRPO require --temperature 1 --top-p 1 for matching behavior log-probabilities"
        )
    if args.algorithm == "grpo" and args.samples_per_prompt < 2:
        parser.error("GRPO requires at least two samples per prompt")
    return args


def main() -> None:
    args = parse_args()
    records = make_rollouts(args)
    write_jsonl(args.output, records)
    mean_reward = sum(row["reward"] for row in records) / max(len(records), 1)
    print(
        f"wrote {len(records)} rollouts to {Path(args.output)}; mean reward={mean_reward:.3f}"
    )


if __name__ == "__main__":
    main()
