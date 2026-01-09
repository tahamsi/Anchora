"""Supervised fine-tuning loop for instruction-following data."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import format_prompt_completion, load_sft_dataset, split_dataset
from .utils import get_device


def _extract_prompt_completion(example: Dict[str, str]) -> Tuple[str, str]:
    if "prompt" in example and "completion" in example:
        return example["prompt"], example["completion"]
    if "instruction" in example and "output" in example:
        return example["instruction"], example["output"]
    raise KeyError("Expected fields (prompt, completion) or (instruction, output)")


def _tokenize_for_sft(example: Dict[str, str], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    prompt, completion = _extract_prompt_completion(example)
    full_text = format_prompt_completion(prompt, completion)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = tokenized.input_ids.clone()
    prompt_only = tokenizer(
        format_prompt_completion(prompt, ""),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_len = prompt_only.input_ids.shape[1]
    labels[:, :prompt_len] = -100
    tokenized["labels"] = labels
    return {k: v.squeeze(0) for k, v in tokenized.items()}


def _collate(batch):
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [example[key] for example in batch], batch_first=True, padding_value=0
        )
        for key in batch[0]
    }


def run_sft(
    model_name: str,
    dataset_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    epochs: int,
    max_length: int,
    grad_accum_steps: int,
    gradient_checkpointing: bool,
    use_bf16: bool,
    cache_dir: str | None,
    logger,
) -> str:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if use_bf16 and device.type == "cuda" else None,
        cache_dir=cache_dir,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    dataset = load_sft_dataset(dataset_path)
    split = split_dataset(dataset)
    train_dataset = split["train"].map(
        lambda ex: _tokenize_for_sft(ex, tokenizer, max_length),
        remove_columns=split["train"].column_names,
    )
    val_dataset = split["val"].map(
        lambda ex: _tokenize_for_sft(ex, tokenizer, max_length),
        remove_columns=split["val"].column_names,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = max(1, len(train_loader) * epochs // max(1, grad_accum_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / max(1, grad_accum_steps)
            loss.backward()
            epoch_loss += loss.item()
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
        logger.info("SFT epoch %d | loss %.4f", epoch + 1, epoch_loss / max(1, len(train_loader)))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        logger.info("SFT val loss %.4f", val_loss / max(1, len(val_loader)))
        model.train()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    logger.info("Saved SFT model to %s", output_path)
    return output_path
