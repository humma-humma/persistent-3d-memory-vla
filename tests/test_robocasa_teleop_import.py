import numpy as np

from persistent_scene_memory.robocasa_teleop_import import (
    reorder_teleop_actions,
    parse_env_config,
    state16_from_observation,
    teleop_phase,
)


def test_reorder_teleop_actions_matches_live_binary_and_field_convention():
    raw = np.arange(12, dtype=np.float32)[None]
    raw[0, 6] = -1
    raw[0, 11] = 1

    action = reorder_teleop_actions(raw)

    assert action.shape == (1, 12)
    assert np.array_equal(action[0, :6], raw[0, :6])
    assert action[0, 6] == 0
    assert np.array_equal(action[0, 7:11], raw[0, 7:11])
    assert action[0, 11] == 1


def test_state16_from_observation_uses_native_field_order():
    observation = {
        "robot0_base_pos": np.arange(3),
        "robot0_base_quat": np.arange(3, 7),
        "robot0_base_to_eef_pos": np.arange(7, 10),
        "robot0_base_to_eef_quat": np.arange(10, 14),
        "robot0_gripper_qpos": np.arange(14, 16),
    }

    assert np.array_equal(state16_from_observation(observation), np.arange(16))


def test_teleop_phase_prioritizes_base_then_manipulation_then_hold():
    actions = np.zeros((8, 12), dtype=np.float32)
    actions[2, 0] = 1
    actions[5, 7] = 1

    assert teleop_phase(actions, 0, 8) == "human_base_motion"
    assert teleop_phase(actions, 0, 4) == "human_manipulation"
    assert teleop_phase(actions, 6, 2) == "human_hold"


def test_parse_env_config_accepts_single_and_double_encoded_metadata():
    config = {"env_name": "PickPlaceCounterToCabinet"}

    assert parse_env_config(config) == config
    assert parse_env_config('{"env_name": "PickPlaceCounterToCabinet"}') == config
    assert parse_env_config('"{\\"env_name\\": \\"PickPlaceCounterToCabinet\\"}"') == config
