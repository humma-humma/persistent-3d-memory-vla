import json

import numpy as np
import pytest

from persistent_scene_memory.robocasa_rollout import RoboCasaEpisode
from persistent_scene_memory.robocasa_smolvla_replay import (
    ACTION_FIELDS,
    EmbodimentMismatchError,
    load_replay_sample,
    policy_action_to_robocasa,
    quaternion_xyzw_to_rotation_vector,
    robocasa_action12,
    robocasa_state6,
)


def frame_arrays():
    arrays = {
        "state__state__end_effector_position_relative": np.array([1, 2, 3], dtype=np.float32),
        "state__state__end_effector_rotation_relative": np.array([0, 0, 0, 1], dtype=np.float32),
    }
    sizes = (3, 3, 1, 4, 1)
    for field, size in zip(ACTION_FIELDS, sizes):
        arrays["action__" + field.replace(".", "__")] = np.arange(size, dtype=np.float32)
    for camera in ("video.left", "video.right", "video.wrist"):
        arrays["rgb__" + camera.replace(".", "__")] = np.zeros((4, 5, 3), dtype=np.uint8)
    return arrays


def test_state6_uses_cartesian_pose_and_stable_quaternion_conversion():
    arrays = frame_arrays()
    assert robocasa_state6(arrays).tolist() == [1, 2, 3, 0, 0, 0]
    half = np.sqrt(0.5)
    rotation = quaternion_xyzw_to_rotation_vector(np.array([0, 0, half, half]))
    assert np.allclose(rotation, [0, 0, np.pi / 2])


def test_action12_has_documented_field_order():
    action = robocasa_action12(frame_arrays())
    assert action.shape == (12,)
    assert action.tolist() == [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 3, 0]


def test_direct_policy_action_execution_is_blocked():
    with pytest.raises(EmbodimentMismatchError, match="Direct control is disabled"):
        policy_action_to_robocasa(np.zeros(6))


def test_loads_aligned_replay_sample(tmp_path):
    episode_dir = tmp_path / "episode"
    memory_dir = tmp_path / "memory"
    episode_dir.mkdir()
    memory_dir.mkdir()
    arrays = frame_arrays()
    np.savez_compressed(episode_dir / "frame_000000.npz", **arrays)
    manifest = {
        "format": "robocasa-rgbd-rollout-v1",
        "task": "place the object",
        "cameras": ["video.left", "video.right", "video.wrist"],
        "frames": [{"frame_index": 0, "file": "frame_000000.npz", "action_source": {"phase": "approach"}}],
    }
    (episode_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez_compressed(
        memory_dir / "adapter_frame_000000.npz",
        values=np.ones((8, 7), dtype=np.float32),
        mask=np.array([True, False, False, False, False, False, False, False]),
    )

    sample = load_replay_sample(RoboCasaEpisode(episode_dir), memory_dir, 0)

    assert sample["task"] == "place the object"
    assert sample["phase"] == "approach"
    assert len(sample["images"]) == 3
    assert sample["geometry_values"].shape == (8, 7)
    assert sample["oracle_action12"].shape == (12,)
