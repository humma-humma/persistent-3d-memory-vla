import numpy as np
import pytest

from persistent_scene_memory.robocasa_policy_eval import blend_predictions, prediction_metrics


def test_prediction_metrics_reports_policy_improvement_and_fields():
    target = np.ones((4, 12), dtype=np.float32)
    prediction = np.full((4, 12), 0.25, dtype=np.float32)

    metrics = prediction_metrics(prediction, target)

    assert metrics["beats_zero_action"] is True
    assert metrics["policy_rmse"] == pytest.approx(0.75)
    assert metrics["zero_action_rmse"] == pytest.approx(1.0)
    assert set(metrics["per_field"]) == {
        "end_effector_position",
        "end_effector_rotation",
        "gripper_close",
        "base_motion",
        "control_mode",
    }


def test_prediction_metrics_rejects_wrong_action_contract():
    with pytest.raises(ValueError, match="shape"):
        prediction_metrics(np.zeros((2, 6)), np.zeros((2, 6)))


def test_blend_predictions_replaces_only_selected_fields():
    primary = np.zeros((2, 12), dtype=np.float32)
    secondary = np.ones((2, 12), dtype=np.float32)

    blended = blend_predictions(primary, secondary, ["end_effector_position"])

    np.testing.assert_array_equal(blended[:, :3], 1.0)
    np.testing.assert_array_equal(blended[:, 3:], 0.0)
    np.testing.assert_array_equal(primary, 0.0)


def test_blend_predictions_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown action fields"):
        blend_predictions(np.zeros((2, 12)), np.zeros((2, 12)), ["unknown"])
