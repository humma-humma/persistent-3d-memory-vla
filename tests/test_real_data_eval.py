import numpy as np
import pytest

from persistent_scene_memory.real_data_eval import action_metrics, blend_with_hold
from persistent_scene_memory.residual_blend import fit_hold_blend


def test_action_metrics_reports_raw_and_scale_normalized_error():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    prediction = target + np.array([[1.0, -2.0], [1.0, -2.0]])

    metrics = action_metrics(prediction, target, np.array([1.0, 2.0]))

    assert metrics["mae"] == 1.5
    assert np.isclose(metrics["rmse"], np.sqrt(2.5))
    assert metrics["normalized_rmse"] == 1.0
    assert metrics["per_dimension_mae"] == [1.0, 2.0]


def test_action_metrics_rejects_incompatible_scales():
    with pytest.raises(ValueError, match="action_std"):
        action_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.ones(3))


def test_hold_blend_and_fit_recover_known_correction_scale():
    hold = np.zeros((2, 2))
    prediction = np.array([[2.0, 4.0], [6.0, 8.0]])
    target = 0.25 * prediction

    alpha = fit_hold_blend(prediction, hold, target)

    assert np.isclose(alpha, 0.25)
    assert np.allclose(blend_with_hold(prediction, hold, alpha), target)


def test_per_dimension_hold_blend_recovers_channel_scales():
    hold = np.zeros((2, 2))
    prediction = np.array([[2.0, 4.0], [6.0, 8.0]])
    target = prediction * np.array([0.25, 0.75])

    alpha = fit_hold_blend(prediction, hold, target, per_dimension=True)

    assert np.allclose(alpha, [0.25, 0.75])
    assert np.allclose(blend_with_hold(prediction, hold, alpha), target)
