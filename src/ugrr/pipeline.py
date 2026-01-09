"""UGRR regret loop orchestration."""
from __future__ import annotations

from typing import Dict, List

import random
import torch

from .data import format_prompt_only, load_sft_dataset
from .regret import RegretBuffer, detect_regret
from .reward import load_reward_models, score_with_reward_model
from .rl import init_policy_models, rl_update
from .uncertainty import score_uncertainty
from .utils import get_device


def _generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    tokens = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=True)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def _self_correct(model, tokenizer, prompt: str, answer: str, max_new_tokens: int) -> str:
    correction_prompt = (
        "You are revising a previous answer. Improve it to be more accurate and complete.\n"
        f"Question: {prompt}\nPrevious answer: {answer}\nRevised answer:"
    )
    return _generate_response(model, tokenizer, correction_prompt, max_new_tokens)


def populate_regret_buffer(
    buffer: RegretBuffer,
    model,
    tokenizer,
    reward_models,
    prompts: List[str],
    max_new_tokens: int,
    uncertainty_threshold: float,
    max_length: int,
    dropout_samples: int,
    weights: Dict[str, float],
    logger,
) -> None:
    for prompt in prompts:
        formatted_prompt = format_prompt_only(prompt)
        response = _generate_response(model, tokenizer, formatted_prompt, max_new_tokens)
        verifier_scores = [
            score_with_reward_model(formatted_prompt, response, rm, max_length) for rm in reward_models
        ]
        uncertainty = score_uncertainty(
            formatted_prompt,
            response,
            model,
            tokenizer,
            reward_models,
            max_length,
            dropout_samples,
            weights,
        )
        regret = detect_regret(
            formatted_prompt,
            response,
            uncertainty,
            verifier_scores,
            uncertainty_threshold,
        )
        if regret:
            regret["good_answer"] = _self_correct(model, tokenizer, formatted_prompt, response, max_new_tokens)
            buffer.add(regret)
    logger.info("Regret buffer size: %d", len(buffer))


def run_regret_loop(
    sft_model_path: str,
    reward_model_paths: List[str],
    output_path: str,
    sft_data_path: str,
    iterations: int,
    batch_size: int,
    rl_lr: float,
    max_length: int,
    max_new_tokens: int,
    kl_beta_min: float,
    kl_beta_max: float,
    dpo_beta: float,
    regret_buffer_size: int,
    regret_priority_mix: float,
    uncertainty_threshold: float,
    dropout_samples: int,
    weights: Dict[str, float],
    cache_dir: str | None,
    logger,
) -> str:
    device = get_device()
    policy_bundle = init_policy_models(sft_model_path, cache_dir, use_bf16=device.type == "cuda")
    policy = policy_bundle["policy"]
    ref = policy_bundle["ref"]
    tokenizer = policy_bundle["tokenizer"]

    reward_models = load_reward_models(reward_model_paths, cache_dir) if reward_model_paths else []

    dataset = load_sft_dataset(sft_data_path)
    prompts = [ex.get("prompt") or ex.get("instruction") for ex in dataset]
    prompts = [p for p in prompts if p]
    random.shuffle(prompts)

    buffer = RegretBuffer(regret_buffer_size, regret_priority_mix)
    populate_regret_buffer(
        buffer,
        policy,
        tokenizer,
        reward_models,
        prompts[: max(10, batch_size * 2)],
        max_new_tokens,
        uncertainty_threshold,
        max_length,
        dropout_samples,
        weights,
        logger,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=rl_lr)

    for iteration in range(iterations):
        batch = buffer.sample_batch(batch_size)
        if not batch:
            logger.warning("Regret buffer empty; repopulating.")
            populate_regret_buffer(
                buffer,
                policy,
                tokenizer,
                reward_models,
                prompts[: max(10, batch_size * 2)],
                max_new_tokens,
                uncertainty_threshold,
                max_length,
                dropout_samples,
                weights,
                logger,
            )
            batch = buffer.sample_batch(batch_size)
        metrics = rl_update(
            policy,
            ref,
            tokenizer,
            batch,
            optimizer,
            max_length,
            kl_beta_min,
            kl_beta_max,
            dpo_beta,
        )
        if (iteration + 1) % 5 == 0:
            logger.info("RL iter %d | loss %.4f | kl %.4f", iteration + 1, metrics["loss"], metrics["kl"])
        if (iteration + 1) % 20 == 0:
            populate_regret_buffer(
                buffer,
                policy,
                tokenizer,
                reward_models,
                prompts[: max(10, batch_size * 2)],
                max_new_tokens,
                uncertainty_threshold,
                max_length,
                dropout_samples,
                weights,
                logger,
            )
            policy.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

    policy.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved UGRR model to %s", output_path)
    return output_path
