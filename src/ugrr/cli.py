"""Command-line interface for the UGRR pipeline."""
from __future__ import annotations

import argparse
import os
from typing import Dict

from .config import TrainConfig
from .evals import evaluate_alpacaeval, evaluate_gsm8k, evaluate_mtbench, save_eval_results
from .pipeline import run_regret_loop
from .reward import train_reward_model, load_reward_models
from .sft import run_sft
from .utils import save_config, set_seed, setup_logging


def _load_config(path: str | None) -> TrainConfig:
    if not path:
        return TrainConfig()
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".json"):
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TrainConfig(**data)

    if path.endswith(".yaml") or path.endswith(".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required for YAML config files") from exc
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)

    raise ValueError("Config must be JSON or YAML")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")


def _resolve_config(args: argparse.Namespace) -> TrainConfig:
    config = _load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    return config


def _build_weights(config: TrainConfig) -> Dict[str, float]:
    return {
        "entropy": config.uncertainty_entropy_weight,
        "diversity": config.uncertainty_diversity_weight,
        "disagreement": config.uncertainty_disagreement_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UGRR pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sft_parser = subparsers.add_parser("sft", help="Run supervised fine-tuning")
    _add_common_args(sft_parser)

    reward_parser = subparsers.add_parser("train_reward", help="Train reward model")
    _add_common_args(reward_parser)

    rl_parser = subparsers.add_parser("regret_loop", help="Run UGRR regret loop")
    _add_common_args(rl_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluations")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.add_argument("--baseline_path", type=str, default=None)

    args = parser.parse_args()
    config = _resolve_config(args)
    set_seed(config.seed)
    logger = setup_logging(os.path.join(config.log_dir, "ugrr.log"))
    save_config(config, os.path.join(config.log_dir, "config.json"))

    if args.mode == "sft":
        run_sft(
            model_name=config.base_model_name,
            dataset_path=config.sft_data_path,
            output_path=os.path.join(config.output_dir, "mistral_sft"),
            batch_size=config.batch_size,
            lr=config.sft_lr,
            epochs=config.sft_epochs,
            max_length=config.max_length,
            grad_accum_steps=config.grad_accum_steps,
            gradient_checkpointing=config.gradient_checkpointing,
            use_bf16=config.use_bf16,
            cache_dir=config.cache_dir,
            logger=logger,
        )
        return

    if args.mode == "train_reward":
        train_reward_model(
            model_name=config.reward_model_name,
            dataset_path=config.preference_data_path,
            output_path=os.path.join(config.output_dir, "reward_model"),
            batch_size=config.batch_size,
            lr=config.reward_lr,
            epochs=config.reward_epochs,
            max_length=config.max_length,
            cache_dir=config.cache_dir,
            logger=logger,
        )
        return

    if args.mode == "regret_loop":
        run_regret_loop(
            sft_model_path=os.path.join(config.output_dir, "mistral_sft"),
            reward_model_paths=[os.path.join(config.output_dir, "reward_model")],
            output_path=os.path.join(config.output_dir, "ugrr_final"),
            sft_data_path=config.sft_data_path,
            iterations=config.rl_iterations,
            batch_size=config.batch_size,
            rl_lr=config.rl_lr,
            max_length=config.max_length,
            max_new_tokens=config.eval_max_new_tokens,
            kl_beta_min=config.kl_beta_min,
            kl_beta_max=config.kl_beta_max,
            dpo_beta=config.dpo_beta,
            regret_buffer_size=config.regret_buffer_size,
            regret_priority_mix=config.regret_priority_mix,
            uncertainty_threshold=config.uncertainty_threshold,
            dropout_samples=config.uncertainty_dropout_samples,
            weights=_build_weights(config),
            cache_dir=config.cache_dir,
            logger=logger,
        )
        return

    if args.mode == "evaluate":
        reward_models = []
        reward_path = os.path.join(config.output_dir, "reward_model")
        if os.path.exists(reward_path):
            reward_models = load_reward_models([reward_path], config.cache_dir)
        results = {}
        if args.baseline_path:
            results.update(
                evaluate_alpacaeval(
                    model_path=args.model_path,
                    baseline_path=args.baseline_path,
                    reward_models=reward_models,
                    max_new_tokens=config.eval_max_new_tokens,
                    cache_dir=config.cache_dir,
                    limit=20,
                    logger=logger,
                )
            )
        results.update(
            evaluate_mtbench(
                model_path=args.model_path,
                reward_models=reward_models,
                max_new_tokens=config.eval_max_new_tokens,
                cache_dir=config.cache_dir,
                limit=20,
                logger=logger,
            )
        )
        results.update(
            evaluate_gsm8k(
                model_path=args.model_path,
                max_new_tokens=config.eval_max_new_tokens,
                cache_dir=config.cache_dir,
                limit=50,
                logger=logger,
            )
        )
        save_eval_results(results, config.eval_output_path)
        logger.info("Saved eval results to %s", config.eval_output_path)
        return


if __name__ == "__main__":
    main()
