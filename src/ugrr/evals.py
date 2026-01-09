"""Evaluation routines for UGRR models."""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .reward import score_with_reward_model
from .utils import get_device


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**tokens, max_new_tokens=max_new_tokens)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt) :].strip()
    return decoded.strip()


def _extract_last_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+\.?\d*", text)
    if not matches:
        return None
    return matches[-1]


def evaluate_gsm8k(model_path: str, max_new_tokens: int, cache_dir: str | None, limit: int | None, logger) -> Dict[str, float]:
    dataset = load_dataset("gsm8k", "main", split="test")
    if limit:
        dataset = dataset.select(range(limit))

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    correct = 0
    for item in dataset:
        prompt = item["question"]
        response = _generate(model, tokenizer, prompt, max_new_tokens)
        pred = _extract_last_number(response)
        gold = _extract_last_number(item["answer"])
        if pred is not None and gold is not None and pred == gold:
            correct += 1
    accuracy = correct / max(1, len(dataset))
    logger.info("GSM8K accuracy %.4f", accuracy)
    return {"gsm8k_accuracy": accuracy}


def evaluate_alpacaeval(
    model_path: str,
    baseline_path: str,
    reward_models: List[Dict[str, object]],
    max_new_tokens: int,
    cache_dir: str | None,
    limit: int | None,
    logger,
) -> Dict[str, float]:
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    if limit:
        dataset = dataset.select(range(limit))

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_path, cache_dir=cache_dir)
    baseline_model = AutoModelForCausalLM.from_pretrained(baseline_path, cache_dir=cache_dir)

    model.to(device)
    baseline_model.to(device)
    model.eval()
    baseline_model.eval()

    wins = 0
    for item in dataset:
        prompt = item["instruction"]
        candidate = _generate(model, tokenizer, prompt, max_new_tokens)
        baseline = _generate(baseline_model, baseline_tokenizer, prompt, max_new_tokens)

        if reward_models:
            score_c = sum(score_with_reward_model(prompt, candidate, rm, 512) for rm in reward_models) / len(reward_models)
            score_b = sum(score_with_reward_model(prompt, baseline, rm, 512) for rm in reward_models) / len(reward_models)
            if score_c >= score_b:
                wins += 1
        else:
            if len(candidate) >= len(baseline):
                wins += 1

    win_rate = wins / max(1, len(dataset))
    logger.info("AlpacaEval proxy win-rate %.4f", win_rate)
    return {"alpaca_eval_win_rate": win_rate}


def evaluate_mtbench(
    model_path: str,
    reward_models: List[Dict[str, object]],
    max_new_tokens: int,
    cache_dir: str | None,
    limit: int | None,
    logger,
) -> Dict[str, float]:
    dataset = load_dataset("HuggingFaceH4/mt_bench", "default", split="train")
    if limit:
        dataset = dataset.select(range(limit))

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    scores = []
    for item in dataset:
        prompt = item["prompt"][0] if isinstance(item["prompt"], list) else item["prompt"]
        response = _generate(model, tokenizer, prompt, max_new_tokens)
        if reward_models:
            score = sum(score_with_reward_model(prompt, response, rm, 512) for rm in reward_models) / len(reward_models)
        else:
            score = float(len(response))
        scores.append(score)

    avg_score = sum(scores) / max(1, len(scores))
    logger.info("MT-Bench proxy score %.4f", avg_score)
    return {"mtbench_score": avg_score}


def save_eval_results(results: Dict[str, float], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
