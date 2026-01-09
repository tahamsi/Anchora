"""Configuration structures and defaults for UGRR pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    base_model_name: str = "mistral-7b"
    reward_model_name: str = "bert-base-uncased"
    sft_data_path: str = "data/sft.json"
    preference_data_path: str = "data/preferences.json"
    output_dir: str = "models"
    log_dir: str = "logs"
    seed: int = 42
    batch_size: int = 2
    eval_batch_size: int = 2
    sft_lr: float = 2e-5
    reward_lr: float = 1e-5
    rl_lr: float = 5e-6
    sft_epochs: int = 1
    reward_epochs: int = 1
    rl_iterations: int = 100
    kl_beta_min: float = 0.02
    kl_beta_max: float = 0.2
    max_length: int = 512
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = True
    use_bf16: bool = True
    uncertainty_dropout_samples: int = 4
    uncertainty_diversity_weight: float = 0.25
    uncertainty_entropy_weight: float = 0.5
    uncertainty_disagreement_weight: float = 0.25
    uncertainty_threshold: float = 0.6
    regret_buffer_size: int = 1000
    regret_priority_mix: float = 0.7
    dpo_beta: float = 0.1
    eval_max_new_tokens: int = 128
    eval_output_path: str = "logs/eval_results.json"
    cache_dir: str | None = None
    extra: dict = field(default_factory=dict)
