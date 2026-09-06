import numpy as np
import pytest

from persistent_scene_memory.robocasa_closed_loop import (
    OracleRecoveryPolicy,
    action12_to_dict,
    live_state16,
    policy_observation_images,
    rollout_diagnostics,
)


def test_oracle_recovery_policy_switches_at_requested_frame():
    class Source:
        def __init__(self, name):
            self.name = name
            self.last_info = {}
            self.closed = False

        def act(self, env, frame_index, observation):
            self.last_info = {"name": self.name}
            return {"source": self.name}

        def close(self):
            self.closed = True

    remote, oracle = Source("policy"), Source("oracle")
    recovery = OracleRecoveryPolicy(remote, oracle, switch_frame=2)

    assert recovery.act(None, 1, {}) == {"source": "policy"}
    assert recovery.last_info["recovery_mode"] == "policy"
    assert recovery.act(None, 2, {}) == {"source": "oracle"}
    assert recovery.last_info["recovery_mode"] == "oracle"
    recovery.close()
    assert remote.closed is True
from persistent_scene_memory.robocasa_smolvla_replay import ACTION_FIELDS


class Box:
    def __init__(self, shape):
        self.shape = shape
        self.low = np.full(shape, -1.0)
        self.high = np.full(shape, 1.0)
        self.dtype = np.dtype(np.float32)


class DictSpace:
    spaces = {
        "action.end_effector_position": Box((3,)),
        "action.end_effector_rotation": Box((3,)),
        "action.gripper_close": Box((1,)),
        "action.base_motion": Box((4,)),
        "action.control_mode": Box((1,)),
    }


def test_live_state_has_native_field_order():
    observation = {
        "state.base_position": np.array([1, 2, 3]),
        "state.base_rotation": np.array([4, 5, 6, 7]),
        "state.end_effector_position_relative": np.array([8, 9, 10]),
        "state.end_effector_rotation_relative": np.array([11, 12, 13, 14]),
        "state.gripper_qpos": np.array([15, 16]),
    }
    assert live_state16(observation).tolist() == list(range(1, 17))


def test_policy_images_support_rgb_only_observations():
    observation = {
        f"video.camera_{index}": np.full((2, 2, 3), index, dtype=np.uint8)
        for index in range(3)
    }
    images = policy_observation_images(observation)
    assert [int(image[0, 0, 0]) for image in images] == [0, 1, 2]


def test_action_unflatten_clips_and_scales_motion_fields():
    result = action12_to_dict(np.full(12, 2.0), DictSpace(), scale=0.25)

    assert tuple(result) == ACTION_FIELDS
    assert np.allclose(result["action.end_effector_position"], 0.5)
    assert np.allclose(result["action.base_motion"], 0.5)
    assert np.allclose(result["action.gripper_close"], 1.0)
    assert np.allclose(result["action.control_mode"], 1.0)


def test_action_unflatten_rejects_invalid_scale():
    with pytest.raises(ValueError, match="scale"):
        action12_to_dict(np.zeros(12), DictSpace(), scale=0)


def test_rollout_diagnostics_measure_motion_and_modes():
    states = np.zeros((2, 16), dtype=np.float32)
    states[1, :3] = [3, 4, 0]
    states[1, 7:10] = [0, 0, 2]
    actions = np.zeros((2, 12), dtype=np.float32)
    actions[0, 6] = 1
    actions[:, 11] = 1

    result = rollout_diagnostics(states, actions)

    assert result["base_displacement"] == pytest.approx(5)
    assert result["eef_relative_displacement"] == pytest.approx(2)
    assert result["gripper_close_fraction"] == pytest.approx(0.5)
    assert result["base_control_fraction"] == pytest.approx(1)
