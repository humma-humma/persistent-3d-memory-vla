import numpy as np
import cv2

from sfm_reconstruction.stage3 import _trajectory_pose_from_camera_to_world
from sfm_reconstruction.stage3_open3d_viewer import (
    build_stage3_scene_points,
    _line_indices,
    load_scene_cloud_ply,
    stage3_trajectory_geometries,
    smooth_points,
    write_stage3_scene_ply,
)
from sfm_reconstruction.stage3_visualize import NamedTrajectory


def _pose(timestamp, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return _trajectory_pose_from_camera_to_world(timestamp, transform)


def test_line_indices_connect_adjacent_points():
    assert _line_indices(1).shape == (0, 2)
    np.testing.assert_array_equal(
        _line_indices(4),
        np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
    )


def test_smooth_points_keeps_endpoints_and_smooths_interior():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    smoothed = smooth_points(points, 3)

    np.testing.assert_allclose(smoothed[0], points[0])
    np.testing.assert_allclose(smoothed[-1], points[-1])
    np.testing.assert_allclose(smoothed[1], points[:3].mean(axis=0))


def test_stage3_trajectory_geometries_align_estimate_to_ground_truth():
    ground_truth = [
        _pose(0.0, [0.0, 0.0, 0.0]),
        _pose(1.0, [1.0, 0.0, 0.0]),
        _pose(2.0, [2.0, 0.0, 0.0]),
        _pose(3.0, [3.0, 0.0, 0.0]),
    ]
    estimate = NamedTrajectory(
        "Estimate",
        [
            _pose(0.0, [10.0, 0.0, 0.0]),
            _pose(1.0, [12.0, 0.0, 0.0]),
            _pose(2.0, [14.0, 0.0, 0.0]),
            _pose(3.0, [16.0, 0.0, 0.0]),
        ],
    )

    geometries = stage3_trajectory_geometries(ground_truth, [estimate])

    assert [geometry.label for geometry in geometries] == ["Ground truth", "Estimate"]
    np.testing.assert_allclose(geometries[1].points, geometries[0].points, atol=1e-9)


def test_build_stage3_scene_points_backprojects_rgbd(tmp_path):
    from sfm_reconstruction.stage3 import load_stage3_dataset

    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir()
    depth_dir.mkdir()
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = [30, 20, 10]  # BGR on disk -> RGB in output
    depth = np.zeros((2, 2), dtype=np.uint16)
    depth[0, 0] = 1000
    cv2.imwrite(str(rgb_dir / "0.000000.png"), rgb)
    cv2.imwrite(str(depth_dir / "0.000000.png"), depth)
    (tmp_path / "camera_parameters.json").write_text(
        '{"intrinsics": [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]}',
        encoding="utf-8",
    )
    dataset = load_stage3_dataset(tmp_path)

    points, colors = build_stage3_scene_points(
        dataset,
        [_pose(0.0, [1.0, 2.0, 3.0])],
        frame_stride=1,
        pixel_stride=1,
    )

    np.testing.assert_allclose(points, [[1.0, 2.0, 4.0]], atol=1e-9)
    np.testing.assert_allclose(colors, [[10 / 255.0, 20 / 255.0, 30 / 255.0]])


def test_write_stage3_scene_ply(tmp_path):
    output = tmp_path / "scene.ply"

    write_stage3_scene_ply(
        output,
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.asarray([[1.0, 0.5, 0.0]], dtype=np.float64),
    )

    text = output.read_text(encoding="ascii")
    assert "element vertex 1" in text
    assert text.splitlines()[-1] == "1 2 3 255 128 0"


def test_load_scene_cloud_ply_can_override_color(tmp_path):
    output = tmp_path / "scene.ply"
    write_stage3_scene_ply(
        output,
        np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.asarray([[1.0, 0.5, 0.0]], dtype=np.float64),
    )

    cloud = load_scene_cloud_ply(
        output,
        label="GT pose",
        uniform_color=(0.0, 0.2, 1.0),
    )

    assert cloud.label == "GT pose"
    np.testing.assert_allclose(cloud.points, [[1.0, 2.0, 3.0]])
    assert cloud.uniform_color == (0.0, 0.2, 1.0)
