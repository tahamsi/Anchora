import torch

from ugrr.regret import RegretBuffer, detect_regret
from ugrr.uncertainty import jaccard_diversity


def test_regret_buffer_sampling():
    buffer = RegretBuffer(max_size=3, priority_mix=1.0)
    buffer.add({"prompt": "a", "priority": 0.1})
    buffer.add({"prompt": "b", "priority": 0.9})
    batch = buffer.sample_batch(1)
    assert batch[0]["prompt"] == "b"


def test_detect_regret_uncertainty():
    regret = detect_regret(
        prompt="q",
        model_answer="a",
        uncertainty=0.9,
        verifier_scores=[0.2],
        uncertainty_threshold=0.6,
    )
    assert regret is not None


def test_jaccard_diversity():
    diversity = jaccard_diversity(["a b c", "a b", "d e"])
    assert 0.0 <= diversity <= 1.0


def test_detect_regret_correct_answer():
    regret = detect_regret(
        prompt="q",
        model_answer="wrong",
        uncertainty=0.1,
        verifier_scores=[0.9],
        uncertainty_threshold=0.6,
        correct_answer="right",
    )
    assert regret is not None
