import numpy as np
import pytest

from persistent_scene_memory.robocasa_temporal_diagnostics import switching_stats


def test_switching_stats_counts_binary_events():
    actions = np.zeros((4, 12), dtype=np.float32)
    actions[1:3, 6] = 1
    actions[2:, 11] = 1
    actions[2, 7] = 0.2

    result = switching_stats(actions)

    assert result["gripper_close_fraction"] == 0.5
    assert result["gripper_transitions"] == 2
    assert result["base_control_fraction"] == 0.5
    assert result["control_transitions"] == 1
    assert result["base_active_fraction"] == 0.25


def test_switching_stats_rejects_wrong_contract():
    with pytest.raises(ValueError, match="shape"):
        switching_stats(np.zeros((3, 6)))
