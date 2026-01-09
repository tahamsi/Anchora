"""Regret detection and replay buffer management."""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import random


class RegretBuffer:
    def __init__(self, max_size: int, priority_mix: float = 0.7):
        self.buffer: Deque[Dict[str, object]] = deque(maxlen=max_size)
        self.priority_mix = priority_mix

    def add(self, example: Dict[str, object]) -> None:
        self.buffer.append(example)

    def sample_batch(self, batch_size: int) -> List[Dict[str, object]]:
        if not self.buffer:
            return []
        if random.random() < self.priority_mix:
            sorted_buf = sorted(self.buffer, key=lambda x: x.get("priority", 0.0), reverse=True)
            return list(sorted_buf[:batch_size])
        return random.sample(list(self.buffer), k=min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def detect_regret(
    prompt: str,
    model_answer: str,
    uncertainty: float,
    verifier_scores: List[float],
    uncertainty_threshold: float,
    score_threshold: float | None = None,
    correct_answer: str | None = None,
) -> Dict[str, object] | None:
    regret = False
    if uncertainty >= uncertainty_threshold:
        regret = True
    if score_threshold is not None and any(score < score_threshold for score in verifier_scores):
        regret = True
    if correct_answer is not None and model_answer.strip() != correct_answer.strip():
        regret = True

    if not regret:
        return None

    return {
        "prompt": prompt,
        "bad_answer": model_answer,
        "good_answer": correct_answer,
        "uncertainty": uncertainty,
        "priority": uncertainty,
    }
