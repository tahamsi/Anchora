"""RL fine-tuning with KL-regularized DPO updates."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_device


def _sequence_logprob(model, input_ids, attention_mask) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    return (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def _tokenize_pair(tokenizer, prompt: str, response: str, max_length: int, device) -> Dict[str, torch.Tensor]:
    tokens = tokenizer(
        prompt + response,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in tokens.items()}


def rl_update(
    policy_model,
    ref_model,
    tokenizer,
    batch: List[Dict[str, object]],
    optimizer,
    max_length: int,
    kl_beta_min: float,
    kl_beta_max: float,
    dpo_beta: float,
) -> Dict[str, float]:
    device = next(policy_model.parameters()).device
    losses = []
    kls = []
    for example in batch:
        prompt = example["prompt"]
        good = example.get("good_answer") or ""
        bad = example.get("bad_answer") or ""
        uncertainty = float(example.get("uncertainty", 0.0))
        beta = kl_beta_min + (kl_beta_max - kl_beta_min) * min(1.0, uncertainty)

        tokens_good = _tokenize_pair(tokenizer, prompt, good, max_length, device)
        tokens_bad = _tokenize_pair(tokenizer, prompt, bad, max_length, device)

        logp_good = _sequence_logprob(policy_model, tokens_good["input_ids"], tokens_good["attention_mask"])
        logp_bad = _sequence_logprob(policy_model, tokens_bad["input_ids"], tokens_bad["attention_mask"])

        with torch.no_grad():
            ref_logp_good = _sequence_logprob(ref_model, tokens_good["input_ids"], tokens_good["attention_mask"])
            ref_logp_bad = _sequence_logprob(ref_model, tokens_bad["input_ids"], tokens_bad["attention_mask"])

        dpo_loss = -F.logsigmoid(dpo_beta * (logp_good - logp_bad)).mean()
        kl = ((logp_good - ref_logp_good) + (logp_bad - ref_logp_bad)).mean()
        loss = dpo_loss + beta * kl

        losses.append(loss)
        kls.append(kl.detach())

    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    return {
        "loss": float(total_loss.item()),
        "kl": float(torch.stack(kls).mean().item()),
    }


def init_policy_models(model_path: str, cache_dir: str | None, use_bf16: bool) -> Dict[str, object]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 and device.type == "cuda" else None,
        cache_dir=cache_dir,
    )
    ref = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 and device.type == "cuda" else None,
        cache_dir=cache_dir,
    )
    policy.to(device)
    ref.to(device)
    ref.eval()

    return {"policy": policy, "ref": ref, "tokenizer": tokenizer}
