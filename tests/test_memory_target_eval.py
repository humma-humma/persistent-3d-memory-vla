import numpy as np
import pytest

from persistent_scene_memory.memory import GeometryObservation
from persistent_scene_memory.memory_target_eval import (
    CONDITIONS,
    PositionEstimator,
    make_condition,
)


def _obs(x, frame, confidence=0.9):
    return GeometryObservation("obj", np.array([x, 0.0, 0.0]), np.zeros(1), confidence, frame)


def test_current_view_forgets_when_nothing_is_visible():
    estimator = PositionEstimator("current")
    estimator.update([("left", _obs(1.0, 0))])
    assert np.allclose(estimator.estimate(0), [1, 0, 0])
    estimator.update([])
    assert estimator.estimate(1) is None


@pytest.mark.parametrize("rule", ["persistent", "echo", "atlas", "gated"])
def test_memory_rules_persist_through_occlusion(rule):
    estimator = PositionEstimator(rule)
    estimator.update([("left", _obs(1.0, 0))])
    for frame in range(1, 20):
        estimator.update([])
    assert np.allclose(estimator.estimate(19), [1, 0, 0])


def test_gate_rejects_single_camera_outliers_but_ungated_memory_jumps():
    gated, ungated = PositionEstimator("gated"), PositionEstimator("persistent")
    for estimator in (gated, ungated):
        estimator.update([("left", _obs(0.0, 0)), ("right", _obs(0.0, 0))])
        estimator.update([("left", _obs(0.25, 1, confidence=0.95))])
    assert np.allclose(gated.estimate(1), [0, 0, 0])
    assert not np.allclose(ungated.estimate(1), [0, 0, 0])


def test_gate_accepts_change_confirmed_by_two_cameras():
    estimator = PositionEstimator("gated")
    estimator.update([("left", _obs(0.0, 0))])
    estimator.update([("left", _obs(0.12, 6))])
    estimator.update([("right", _obs(0.12, 12))])
    assert np.allclose(estimator.estimate(12), [0.12, 0, 0])


def test_conditions_are_well_formed():
    for name in CONDITIONS:
        assert make_condition(name).name == name
    assert make_condition("moved").stale_offset is not None
    with pytest.raises(ValueError):
        make_condition("unknown")
    with pytest.raises(ValueError):
        PositionEstimator("unknown")
