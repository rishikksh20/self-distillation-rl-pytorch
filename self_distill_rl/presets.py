"""Small single-GPU defaults shared by every command."""

from dataclasses import dataclass
from pathlib import Path

QWEN_MODEL = "Qwen/Qwen3.5-0.8B"
LFM_MODEL = "LiquidAI/LFM2.5-350M"


@dataclass(frozen=True)
class Preset:
    model: str
    dataset: str
    max_tokens: int
    max_seq_length: int
    samples_per_prompt: int

    @property
    def train_path(self) -> str:
        return str(Path("artifacts/datasets") / self.dataset / "train.jsonl")

    @property
    def eval_path(self) -> str:
        return str(Path("artifacts/datasets") / self.dataset / "test.jsonl")

    @property
    def rollout_path(self) -> str:
        family = (
            "qwen"
            if self.model == QWEN_MODEL
            else ("grpo" if self.samples_per_prompt > 1 else "ppo")
        )
        return f"artifacts/{family}_rollouts.jsonl"


PRESETS = {
    "sdft": Preset(QWEN_MODEL, "gsm8k", 512, 2048, 1),
    "opsd": Preset(QWEN_MODEL, "gsm8k", 512, 2048, 1),
    "sdpo": Preset(QWEN_MODEL, "gsm8k", 512, 2048, 1),
    "ppo": Preset(LFM_MODEL, "svamp", 256, 1024, 1),
    "grpo": Preset(LFM_MODEL, "svamp", 256, 1024, 4),
}
