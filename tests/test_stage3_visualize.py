import numpy as np

from sfm_reconstruction.stage3 import _trajectory_pose_from_camera_to_world
from sfm_reconstruction.stage3_visualize import (
    NamedTrajectory,
    aligned_trajectory_points,
    plot_stage3_trajectories,
    similarity_align_points,
)


def _pose(timestamp, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return _trajectory_pose_from_camera_to_world(timestamp, transform)


def test_similarity_align_points_recovers_scaled_translation():
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    estimated = reference * 2.0 + np.asarray([5.0, -1.0, 3.0])

    aligned = similarity_align_points(reference, estimated)

    np.testing.assert_allclose(aligned, reference, atol=1e-9)


def test_aligned_trajectory_points_keeps_all_estimated_poses():
    reference = [
        _pose(0.0, [0.0, 0.0, 0.0]),
        _pose(1.0, [1.0, 0.0, 0.0]),
        _pose(2.0, [2.0, 0.0, 0.0]),
        _pose(3.0, [3.0, 0.0, 0.0]),
    ]
    estimated = [
        _pose(0.0, [10.0, 0.0, 0.0]),
        _pose(1.0, [12.0, 0.0, 0.0]),
        _pose(2.0, [14.0, 0.0, 0.0]),
        _pose(3.0, [16.0, 0.0, 0.0]),
    ]

    ref_matched, aligned_matched, aligned_all = aligned_trajectory_points(
        reference, estimated
    )

    assert aligned_all.shape == (4, 3)
    np.testing.assert_allclose(aligned_matched, ref_matched, atol=1e-9)


def test_plot_stage3_trajectories_writes_png(tmp_path):
    reference = NamedTrajectory(
        "Ground truth",
        [
            _pose(0.0, [0.0, 0.0, 0.0]),
            _pose(1.0, [1.0, 0.0, 0.0]),
            _pose(2.0, [1.0, 0.0, 1.0]),
        ],
    )
    estimate = NamedTrajectory(
        "Estimate",
        [
            _pose(0.0, [0.1, 0.0, 0.0]),
            _pose(1.0, [1.1, 0.0, 0.0]),
            _pose(2.0, [1.1, 0.0, 1.0]),
        ],
    )
    output = tmp_path / "trajectory.png"

    plot_stage3_trajectories(reference, [estimate], output)

    assert output.is_file()
    assert output.stat().st_size > 0
