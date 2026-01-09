"""Reward model training and scoring utilities."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_preference_dataset
from .utils import get_device


def _tokenize_pair(tokenizer, prompt: str, response: str, max_length: int):
    return tokenizer(
        f"{prompt}\n{response}",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _collate(batch):
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [example[key] for example in batch], batch_first=True, padding_value=0
        )
        for key in batch[0]
    }


def prepare_preference_tensors(example: Dict[str, str], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    token_a = _tokenize_pair(tokenizer, example["prompt"], example["response_a"], max_length)
    token_b = _tokenize_pair(tokenizer, example["prompt"], example["response_b"], max_length)
    label = torch.tensor([float(example["label"])])

    return {
        "input_ids_a": token_a.input_ids.squeeze(0),
        "attention_mask_a": token_a.attention_mask.squeeze(0),
        "input_ids_b": token_b.input_ids.squeeze(0),
        "attention_mask_b": token_b.attention_mask.squeeze(0),
        "label": label,
    }


def train_reward_model(
    model_name: str,
    dataset_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    epochs: int,
    max_length: int,
    cache_dir: str | None,
    logger,
) -> str:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        cache_dir=cache_dir,
    )
    model.to(device)

    dataset = load_preference_dataset(dataset_path)
    processed = dataset.map(
        lambda ex: prepare_preference_tensors(ex, tokenizer, max_length),
        remove_columns=dataset.column_names,
    )

    loader = DataLoader(processed, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            label = batch["label"].to(device)

            score_a = model(input_ids=input_ids_a, attention_mask=mask_a).logits.squeeze(-1)
            score_b = model(input_ids=input_ids_b, attention_mask=mask_b).logits.squeeze(-1)
            diff = score_a - score_b
            loss = loss_fn(diff, label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("Reward epoch %d | loss %.4f", epoch + 1, total_loss / max(1, len(loader)))

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved reward model to %s", output_path)
    return output_path


def load_reward_models(model_paths: List[str], cache_dir: str | None) -> List[Dict[str, object]]:
    models = []
    device = get_device()
    for path in model_paths:
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(path, cache_dir=cache_dir)
        model.to(device)
        model.eval()
        models.append({"tokenizer": tokenizer, "model": model})
    return models


def score_with_reward_model(prompt: str, response: str, model_bundle: Dict[str, object], max_length: int) -> float:
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = next(model.parameters()).device
    tokens = _tokenize_pair(tokenizer, prompt, response, max_length)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        score = model(**tokens).logits.squeeze(-1).item()
    return float(score)
