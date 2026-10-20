import numpy as np
import pytest

from persistent_scene_memory.baseline_memory import ConfidenceWeightedMemory, ThresholdEMAMemory
from persistent_scene_memory.memory import GeometryObservation


def _obs(position, frame, confidence=0.8, key="obj"):
    return GeometryObservation(key, np.asarray(position, float), np.zeros(2), confidence, frame)


def test_threshold_ema_ignores_small_residuals_and_blends_large_ones():
    memory = ThresholdEMAMemory(tau=0.05, alpha=0.2)
    memory.update(_obs([0, 0, 0], 0))
    assert np.allclose(memory.update(_obs([0.03, 0, 0], 1)).position, [0, 0, 0])
    moved = memory.update(_obs([1.0, 0, 0], 2)).position
    assert np.allclose(moved, [0.2, 0, 0])


def test_threshold_ema_converges_to_persistent_change():
    memory = ThresholdEMAMemory(tau=0.05, alpha=0.2)
    memory.update(_obs([0, 0, 0], 0))
    for frame in range(1, 30):
        token = memory.update(_obs([1.0, 0, 0], frame))
    assert abs(token.position[0] - 1.0) <= 0.05 + 1e-9


def test_confidence_weighted_fusion_matches_running_average():
    memory = ConfidenceWeightedMemory(lam=1.0)
    memory.update(_obs([0, 0, 0], 0, confidence=0.5))
    token = memory.update(_obs([1, 0, 0], 1, confidence=0.5))
    assert np.allclose(token.position, [0.5, 0, 0])
    token = memory.update(_obs([1, 0, 0], 2, confidence=1.0))
    assert np.allclose(token.position, [0.75, 0, 0])


def test_confidence_weighted_decay_lets_change_win_faster():
    slow, fast = ConfidenceWeightedMemory(lam=1.0), ConfidenceWeightedMemory(lam=0.5)
    for memory in (slow, fast):
        for frame in range(5):
            memory.update(_obs([0, 0, 0], frame))
        memory.update(_obs([1, 0, 0], 5))
    assert fast.tokens(5)[0].position[0] > slow.tokens(5)[0].position[0]


def test_query_time_decay_is_shared_with_persistent_memory():
    memory = ThresholdEMAMemory(confidence_decay=0.5)
    memory.update(_obs([0, 0, 0], 0, confidence=0.8))
    assert memory.tokens(2)[0].confidence == pytest.approx(0.2)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        ThresholdEMAMemory(tau=0.0)
    with pytest.raises(ValueError):
        ThresholdEMAMemory(alpha=0.0)
    with pytest.raises(ValueError):
        ConfidenceWeightedMemory(lam=0.0)
