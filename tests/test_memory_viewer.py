import json
from types import SimpleNamespace

import numpy as np
import pytest

from persistent_scene_memory.memory_viewer import (
    colors_for_memory_frame,
    load_memory_frame,
    load_memory_timeline,
    open_memory_viewer,
)
import persistent_scene_memory.memory_viewer as memory_viewer


def write_frame(path, frame_index, tokens):
    path.write_text(
        json.dumps(
            {
                "frame_index": frame_index,
                "feature_size": 3,
                "tokens": tokens,
            }
        ),
        encoding="utf-8",
    )


def token(key, last_seen, confidence, color):
    return {
        "key": key,
        "position": [1, 2, 3],
        "feature": color,
        "confidence": confidence,
        "last_seen": last_seen,
    }


def test_load_memory_frame_computes_token_ages(tmp_path):
    path = tmp_path / "memory_frame_000003.json"
    write_frame(path, 3, [token("current", 3, 0.9, [1, 0, 0]), token("old", 1, 0.4, [0, 0, 1])])

    frame = load_memory_frame(path)

    assert frame.keys == ("current", "old")
    assert frame.positions.shape == (2, 3)
    assert frame.ages.tolist() == [0, 2]


def test_memory_color_modes_distinguish_status_and_preserve_rgb(tmp_path):
    path = tmp_path / "memory_frame_000003.json"
    write_frame(path, 3, [token("current", 3, 1.0, [1, 0, 0]), token("old", 1, 1.0, [0, 0, 1])])
    frame = load_memory_frame(path)

    status = colors_for_memory_frame(frame, "status")

    assert not np.allclose(status[0], status[1])
    assert colors_for_memory_frame(frame, "rgb").tolist() == [[1, 0, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="unknown"):
        colors_for_memory_frame(frame, "missing")


def test_load_memory_timeline_sorts_snapshot_paths(tmp_path):
    write_frame(tmp_path / "memory_frame_000002.json", 2, [])
    write_frame(tmp_path / "memory_frame_000001.json", 1, [])

    frames = load_memory_timeline(tmp_path)

    assert [frame.frame_index for frame in frames] == [1, 2]


def test_viewer_adds_populated_cloud_before_camera_fit(tmp_path, monkeypatch):
    write_frame(
        tmp_path / "memory_frame_000000.json",
        0,
        [token("point", 0, 1.0, [1, 0, 0])],
    )
    added_sizes = []

    class FakePointCloud:
        pass

    class FakeVisualizer:
        def create_window(self, **_kwargs):
            pass

        def add_geometry(self, cloud, **_kwargs):
            added_sizes.append(len(cloud.points))

        def register_key_callback(self, *_args):
            pass

        def get_render_option(self):
            return SimpleNamespace()

        def update_geometry(self, _cloud):
            pass

        def run(self):
            pass

        def destroy_window(self):
            pass

    fake_open3d = SimpleNamespace(
        geometry=SimpleNamespace(PointCloud=FakePointCloud),
        utility=SimpleNamespace(Vector3dVector=np.asarray),
        visualization=SimpleNamespace(VisualizerWithKeyCallback=FakeVisualizer),
    )
    monkeypatch.setattr(memory_viewer, "_require_open3d", lambda: fake_open3d)

    open_memory_viewer(tmp_path)

    assert added_sizes == [1]
