"""Data loading and preprocessing utilities."""
from __future__ import annotations

from typing import Dict, List

from datasets import Dataset, load_dataset


def load_sft_dataset(path: str) -> Dataset:
    if path.endswith(".json"):
        return load_dataset("json", data_files=path)["train"]
    if path.endswith(".csv"):
        return load_dataset("csv", data_files=path)["train"]
    raise ValueError(f"Unsupported SFT dataset format: {path}")


def load_preference_dataset(path: str) -> Dataset:
    if path.endswith(".json"):
        return load_dataset("json", data_files=path)["train"]
    if path.endswith(".csv"):
        return load_dataset("csv", data_files=path)["train"]
    raise ValueError(f"Unsupported preference dataset format: {path}")


def split_dataset(dataset: Dataset, val_fraction: float = 0.05) -> Dict[str, Dataset]:
    split = dataset.train_test_split(test_size=val_fraction, seed=42)
    return {"train": split["train"], "val": split["test"]}


def format_prompt_completion(prompt: str, completion: str) -> str:
    return f"<s>[INST] {prompt.strip()} [/INST] {completion.strip()}"


def format_prompt_only(prompt: str) -> str:
    return f"<s>[INST] {prompt.strip()} [/INST]"


def build_preference_pairs(record: Dict[str, str]) -> List[Dict[str, str]]:
    """Normalize preference data into a canonical format."""
    if {"prompt", "response_a", "response_b", "label"}.issubset(record.keys()):
        return [record]
    raise ValueError("Preference record missing required fields.")
