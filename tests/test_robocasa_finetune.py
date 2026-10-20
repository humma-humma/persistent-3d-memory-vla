import numpy as np
import pytest

import persistent_scene_memory.robocasa_finetune as finetune

from persistent_scene_memory.robocasa_finetune import (
    action_event_label,
    available_episode_anchors,
    dataset_statistics,
    episode_anchors,
    episode_geometry,
    episode_training_arrays,
    phase_balanced_example_order,
    selected_action_indices,
    validate_demonstrations,
)
from persistent_scene_memory.robocasa_smolvla_replay import robocasa_state16


def test_episode_anchors_keep_full_horizon_and_include_last_valid_frame():
    assert episode_anchors(180, horizon=50, stride=50) == [0, 50, 100, 130]
    assert episode_anchors(20, horizon=50) == []
    with pytest.raises(ValueError):
        episode_anchors(0)


def test_available_episode_anchors_respects_sparse_rgb_cache():
    cache = {"stride": 20, "horizon": 50}
    anchors = available_episode_anchors(112, horizon=10, stride=20, training_cache=cache)
    assert anchors == [0, 20, 40, 60]
    assert available_episode_anchors(112, horizon=10, stride=20) == [0, 20, 40, 60, 80, 100, 102]


def test_phase_balanced_order_equalizes_updates_and_is_deterministic():
    phases = ["approach"] * 5 + ["grasp"] + ["transport"] * 2

    first = phase_balanced_example_order(phases, steps=12, seed=7)
    second = phase_balanced_example_order(phases, steps=12, seed=7)
    counts = {
        phase: sum(phases[index] == phase for index in first)
        for phase in set(phases)
    }

    assert first == second
    assert counts == {"approach": 4, "grasp": 4, "transport": 4}
    with pytest.raises(ValueError, match="nonempty"):
        phase_balanced_example_order([], steps=1, seed=7)


def test_dataset_statistics_preserve_native_dimensions():
    states = np.arange(48, dtype=np.float32).reshape(3, 16)
    actions = np.arange(36, dtype=np.float32).reshape(3, 12)
    stats = dataset_statistics(states, actions)
    assert stats["observation.state"]["mean"].shape == (16,)
    assert stats["action"]["std"].shape == (12,)
    assert np.allclose(stats["action"]["min"], actions[0])


def test_native_state_has_documented_field_order():
    arrays = {
        "state__state__base_position": np.array([1, 2, 3]),
        "state__state__base_rotation": np.array([4, 5, 6, 7]),
        "state__state__end_effector_position_relative": np.array([8, 9, 10]),
        "state__state__end_effector_rotation_relative": np.array([11, 12, 13, 14]),
        "state__state__gripper_qpos": np.array([15, 16]),
    }
    assert robocasa_state16(arrays).tolist() == list(range(1, 17))


def test_selected_action_indices_maps_fields_and_rejects_invalid_input():
    assert selected_action_indices(["base_motion", "end_effector_position"]) == [
        0,
        1,
        2,
        7,
        8,
        9,
        10,
    ]
    with pytest.raises(ValueError, match="unknown action fields"):
        selected_action_indices(["unknown"])
    with pytest.raises(ValueError, match="unique"):
        selected_action_indices(["gripper_close", "gripper_close"])


def test_action_event_label_prioritizes_gripper_opening_and_transitions():
    actions = np.zeros((4, 12), dtype=np.float32)
    assert action_event_label(actions) == "gripper_open"
    actions[:, 6] = 1
    assert action_event_label(actions) == "steady_closed"
    actions[2:, 11] = 1
    assert action_event_label(actions) == "control_transition"
    actions[2:, 6] = 0
    assert action_event_label(actions) == "gripper_transition"
    with pytest.raises(ValueError, match="shape"):
        action_event_label(np.zeros((2, 6)))


def test_demonstration_validation_rejects_failure_and_duplicate_seed(tmp_path):
    class Episode:
        def __init__(self, seed, success, name):
            self.root = tmp_path / name
            self.manifest = {
                "seed": seed,
                "frames": [{"frame_index": 7, "success": success}],
            }

    accepted = validate_demonstrations([Episode(1, True, "one")])
    assert accepted[0]["first_success_frame"] == 7
    with pytest.raises(ValueError, match="no successful frame"):
        validate_demonstrations([Episode(2, False, "failed")])
    with pytest.raises(ValueError, match="distinct identifiers"):
        validate_demonstrations([Episode(3, True, "a"), Episode(3, True, "b")])
    labelled = validate_demonstrations([Episode(4, False, "dagger")], expert_labelled=[True])
    assert labelled[0]["first_success_frame"] is None and labelled[0]["expert_labelled"]


def test_demonstration_validation_accepts_official_source_ids(tmp_path):
    class Episode:
        root = tmp_path / "official"
        manifest = {
            "demonstration_id": "official:episode_000007",
            "frames": [{"frame_index": 10, "success": True}],
        }

    record = validate_demonstrations([Episode()])[0]

    assert record["seed"] is None
    assert record["demonstration_id"] == "official:episode_000007"


def test_episode_geometry_is_object_minus_gripper_with_valid_flag():
    class Episode:
        root = "fake"
        manifest = {
            "frames": [
                {"action_source": {"eef_world": [1.0, 0.0, 1.0], "object_world": [1.5, -0.25, 0.75]}},
                {"action_source": {"eef_world": [0.0, 0.0, 0.0], "object_world": [0.0, 0.0, 0.0]}},
            ]
        }

    geometry = episode_geometry(Episode())
    assert geometry.shape == (2, 4)
    assert np.allclose(geometry[0], [0.5, -0.25, -0.25, 1.0])
    assert dataset_statistics(np.zeros((2, 20)), np.zeros((2, 12)))["observation.state"]["mean"].shape == (20,)
    Episode.manifest = {"frames": [{"action_source": {}}]}
    with pytest.raises(ValueError):
        episode_geometry(Episode())


def test_episode_training_arrays_cache_anchor_rgb_and_reuse_it(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune, "robocasa_state16", lambda arrays: arrays["state"])
    monkeypatch.setattr(finetune, "robocasa_action12", lambda arrays: arrays["action"])

    class Episode:
        root = tmp_path
        manifest = {"cameras": ["a", "b"]}
        reads = 0

        def __len__(self):
            return 4

        def frame(self, index):
            Episode.reads += 1
            value = np.uint8(index)
            return {
                "arrays": {
                    "state": np.full(16, index, dtype=np.float32),
                    "action": np.full(12, index, dtype=np.float32),
                    "rgb__a": np.full((2, 2, 3), value),
                    "rgb__b": np.full((2, 2, 3), value + 10),
                }
            }

    states, actions, rgb = episode_training_arrays(Episode(), [0, 2])
    assert states.shape == (4, 16) and actions[3, 0] == 3
    assert rgb.shape == (2, 2, 2, 2, 3)
    assert rgb[1, 0, 0, 0, 0] == 2 and rgb[1, 1, 0, 0, 0] == 12
    assert Episode.reads == 4

    _, _, cached = episode_training_arrays(Episode(), [0, 2])
    assert Episode.reads == 4 and np.array_equal(cached, rgb)
    del rgb, cached  # Windows cannot replace a file that is still memory-mapped.
    episode_training_arrays(Episode(), [0, 1])
    assert Episode.reads == 8
