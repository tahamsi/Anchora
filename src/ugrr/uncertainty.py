"""Uncertainty scoring for model outputs."""
from __future__ import annotations

from typing import List

import math
import torch

from .reward import score_with_reward_model


def _token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def compute_entropy(prompt_ids: torch.Tensor, response_ids: torch.Tensor, model) -> float:
    input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0))
    logits = outputs.logits[:, :-1, :]
    response_logits = logits[:, prompt_ids.shape[-1] - 1 : -1, :]
    ent = _token_entropy(response_logits).mean().item()
    return float(ent)


def jaccard_diversity(texts: List[str]) -> float:
    if len(texts) < 2:
        return 0.0
    sets = [set(t.lower().split()) for t in texts]
    distances = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i].union(sets[j])
            inter = sets[i].intersection(sets[j])
            if not union:
                distances.append(0.0)
            else:
                distances.append(1.0 - (len(inter) / len(union)))
    return float(sum(distances) / max(1, len(distances)))


def mc_dropout_diversity(model, tokenizer, prompt: str, num_samples: int, max_new_tokens: int) -> float:
    model.train()  # enable dropout
    responses = []
    for _ in range(num_samples):
        tokens = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            output_ids = model.generate(**tokens, max_new_tokens=max_new_tokens, do_sample=True)
        responses.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    model.eval()
    return jaccard_diversity(responses)


def score_uncertainty(
    prompt: str,
    response: str,
    model,
    tokenizer,
    reward_models,
    max_length: int,
    dropout_samples: int,
    weights: dict,
) -> float:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids.squeeze(0).to(device)
    response_ids = tokenizer(response, return_tensors="pt", truncation=True, max_length=max_length).input_ids.squeeze(0).to(device)

    entropy = compute_entropy(prompt_ids, response_ids, model)
    vocab_size = model.get_output_embeddings().weight.shape[0]
    entropy_norm = min(1.0, entropy / math.log(vocab_size + 1e-8))

    diversity = mc_dropout_diversity(model, tokenizer, prompt, dropout_samples, max_new_tokens=64)

    scores = []
    for verifier in reward_models:
        scores.append(score_with_reward_model(prompt, response, verifier, max_length))
    if scores:
        score_std = float(torch.tensor(scores).std().item())
    else:
        score_std = 0.0

    entropy_w = weights.get("entropy", 0.5)
    diversity_w = weights.get("diversity", 0.25)
    disagreement_w = weights.get("disagreement", 0.25)
    uncertainty = entropy_w * entropy_norm + diversity_w * diversity + disagreement_w * min(1.0, score_std)
    return float(min(1.0, uncertainty))
