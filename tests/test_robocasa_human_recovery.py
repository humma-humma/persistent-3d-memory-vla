import numpy as np

from persistent_scene_memory.robocasa_human_recovery import (
    _camera_names,
    keyboard_input_to_action,
    summarize_attempt,
)


def test_keyboard_action_uses_live_12d_convention():
    result = keyboard_input_to_action(
        {
            "right_delta": np.arange(6, dtype=np.float32),
            "right_gripper": np.asarray([1.0]),
            "base": np.asarray([0.1, 0.2, 0.3]),
            "torso": np.asarray([0.4]),
            "base_mode": np.asarray([-1.0]),
        },
        "right",
    )
    assert result["action.end_effector_position"].tolist() == [0.0, 1.0, 2.0]
    assert result["action.end_effector_rotation"].tolist() == [3.0, 4.0, 5.0]
    assert result["action.gripper_close"].tolist() == [1.0]
    assert np.allclose(result["action.base_motion"], [0.1, 0.2, 0.3, 0.4])
    assert result["action.control_mode"].tolist() == [0.0]


def test_camera_names_accept_rgb_only_observations():
    observation = {
        "video.robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        "state.base_position": np.zeros(3),
    }
    assert _camera_names(observation) == [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]


def test_attempt_requires_policy_human_and_native_success():
    frames = [
        {"action_source": {"recovery_mode": "policy"}, "success": False},
        {"action_source": {"recovery_mode": "human"}, "success": False},
        {"action_source": {"recovery_mode": "human"}, "success": True},
    ]
    assert summarize_attempt(frames, 1) == {
        "success": True,
        "accepted": True,
        "frames": 3,
        "policy_frames": 1,
        "human_correction_frames": 2,
        "switch_frame": 1,
    }
    frames[-1]["success"] = False
    assert not summarize_attempt(frames, 1)["accepted"]
