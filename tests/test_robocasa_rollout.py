import json

import numpy as np

from persistent_scene_memory.robocasa_rollout import (
    RoboCasaEpisode,
    episode_to_memory,
    record_episode,
)
from persistent_scene_memory.robocasa_rollout_viewer import build_rollout_viewer


class FakeSpace:
    def __init__(self, *, shape=None, dtype=np.float32, spaces=None):
        self.shape = shape
        self.dtype = dtype
        if spaces is not None:
            self.spaces = spaces


class FakeEnv:
    action_space = FakeSpace(spaces={"action.x": FakeSpace(shape=(1,))})

    def _observation(self, value):
        rgb = np.full((4, 4, 3), value, dtype=np.uint8)
        depth = np.ones((4, 4, 1), dtype=np.float32)
        return {
            "video.cam_image": rgb,
            "video.cam_depth": depth,
            "state.arm": np.array([value], dtype=np.float32),
            "annotation.human.task_description": "pick the object",
        }

    def reset(self, seed=None):
        self.value = 1
        return self._observation(self.value), {}

    def step(self, action):
        self.value += 1
        return self._observation(self.value), 0.0, False, False, {"success": False}


def calibration_fn(env, pairs, height, width):
    return {mapped: {"intrinsics": np.eye(3), "camera_to_world": np.eye(4)} for mapped, _ in pairs}


def test_record_and_read_portable_episode(tmp_path):
    manifest = record_episode(
        FakeEnv(), tmp_path / "episode", steps=2,
        calibration_fn=calibration_fn, depth_fn=lambda env, depth: depth,
    )
    episode = RoboCasaEpisode(tmp_path / "episode")

    assert manifest["task"] == "pick the object"
    assert len(episode) == 2
    assert episode.frame(0)["arrays"]["rgb__video__cam"][0, 0, 0] == 1
    assert episode.frame(1)["arrays"]["state__state__arm"].tolist() == [2]


def test_world_points_use_intrinsics_and_camera_pose(tmp_path):
    record_episode(
        FakeEnv(), tmp_path / "episode", steps=1,
        calibration_fn=calibration_fn, depth_fn=lambda env, depth: depth,
    )
    points, colors, cameras = RoboCasaEpisode(tmp_path / "episode").world_points(0, stride=2)

    assert points.shape == (4, 3)
    assert np.allclose(points[0], [0, 0, 1])
    assert np.allclose(colors, 1 / 255)
    assert cameras.tolist() == [0, 0, 0, 0]


def test_episode_exports_memory_and_fixed_policy_tokens(tmp_path):
    record_episode(
        FakeEnv(), tmp_path / "episode", steps=2,
        calibration_fn=calibration_fn, depth_fn=lambda env, depth: depth,
    )
    summary = episode_to_memory(
        RoboCasaEpisode(tmp_path / "episode"), tmp_path / "memory",
        stride=2, voxel_size=1.0, max_tokens=3,
    )

    assert len(summary["frames"]) == 2
    assert summary["frames"][-1]["policy_tokens"] == 3
    assert json.loads((tmp_path / "memory" / "memory_frame_000001.json").read_text())["tokens"]
    with np.load(tmp_path / "memory" / "adapter_frame_000001.npz") as tokens:
        assert tokens["values"].shape == (3, 8)
        assert tokens["mask"].tolist() == [True, True, True]


def test_action_source_metadata_and_standalone_viewer(tmp_path):
    class Source:
        last_info = {}

        def act(self, env, frame_index, observation):
            self.last_info = {"name": "test", "phase": f"phase-{frame_index}"}
            return {"action.x": np.array([0.5], dtype=np.float32)}

    record_episode(
        FakeEnv(), tmp_path / "episode", steps=2, action_fn=Source(),
        calibration_fn=calibration_fn, depth_fn=lambda env, depth: depth,
    )
    episode_to_memory(
        RoboCasaEpisode(tmp_path / "episode"), tmp_path / "memory",
        stride=2, voxel_size=1.0, max_tokens=3,
    )
    html = build_rollout_viewer(tmp_path / "episode", tmp_path / "memory")

    assert "RoboCasa rollout" in html
    assert "pick the object" in html
    assert "phase-1" in html
    assert "data:image/jpeg;base64," in html
    assert '<canvas id="cloud">' in html


def test_record_episode_can_stop_on_first_success(tmp_path):
    class SuccessfulEnv(FakeEnv):
        def step(self, action):
            self.value += 1
            return self._observation(self.value), 1.0, False, False, {"success": True}

    manifest = record_episode(
        SuccessfulEnv(), tmp_path / "episode", steps=5, stop_on_success=True,
        calibration_fn=calibration_fn, depth_fn=lambda env, depth: depth,
    )

    assert len(manifest["frames"]) == 1
    assert manifest["frames"][0]["success"] is True
