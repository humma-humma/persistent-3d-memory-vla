import json

import numpy as np
import pytest

from sfm_reconstruction.stage3 import (
    Stage3OdometryReport,
    TrajectoryPose,
    apply_loop_closure,
    evaluate_translation_trajectory,
    identity_trajectory_template,
    load_stage3_dataset,
    load_trajectory,
    main,
    write_odometry_report,
    write_loop_closure_summary,
    write_stage3_manifest,
    write_trajectory,
    _quaternion_xyzw_from_rotation,
    _estimate_rigid_transform_ransac,
    _rigid_transform,
    _trajectory_pose_from_camera_to_world,
    write_trajectory_metrics,
)


def _write_stage3_dataset(root):
    rgb = root / "rgb"
    depth = root / "depth"
    rgb.mkdir()
    depth.mkdir()
    for timestamp in ("0.000000", "0.033333"):
        (rgb / f"{timestamp}.png").touch()
        (depth / f"{timestamp}.png").touch()
    (root / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": [
                    [525.0, 0.0, 319.5],
                    [0.0, 525.0, 239.5],
                    [0.0, 0.0, 1.0],
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "gt_camera_trajectory.txt").write_text(
        "0.000000 0 0 0 0 0 0 1\n"
        "0.033333 0.1 0 0 0 0 0.01 0.99995\n",
        encoding="utf-8",
    )


def test_load_stage3_dataset_and_trajectory(tmp_path):
    _write_stage3_dataset(tmp_path)

    dataset = load_stage3_dataset(tmp_path)
    trajectory = load_trajectory(tmp_path / "gt_camera_trajectory.txt")

    assert len(dataset.frames) == 2
    assert dataset.frames[0].timestamp == 0.0
    assert dataset.frames[0].depth_path is not None
    assert dataset.intrinsics.shape == (3, 3)
    assert len(trajectory) == 2
    np.testing.assert_allclose(trajectory[1].translation, [0.1, 0.0, 0.0])


def test_stage3_loader_rejects_missing_depth_by_default(tmp_path):
    _write_stage3_dataset(tmp_path)
    (tmp_path / "depth" / "0.033333.png").unlink()

    with pytest.raises(ValueError, match="Missing depth maps"):
        load_stage3_dataset(tmp_path)

    dataset = load_stage3_dataset(tmp_path, allow_missing_depth=True)
    assert dataset.frames[1].depth_path is None


def test_write_stage3_manifest_and_identity_template(tmp_path):
    _write_stage3_dataset(tmp_path)
    dataset = load_stage3_dataset(tmp_path)
    manifest_path = tmp_path / "out" / "stage3_manifest.json"
    trajectory_path = tmp_path / "out" / "estimated_camera_trajectory.txt"

    write_stage3_manifest(manifest_path, dataset)
    write_trajectory(trajectory_path, identity_trajectory_template(dataset))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert manifest["matched_depth_count"] == 2
    assert trajectory_path.read_text(encoding="utf-8").splitlines()[0].endswith(
        "0 0 0 0 0 0 1"
    )


def test_stage3_setup_cli_writes_manifest_and_template(tmp_path):
    _write_stage3_dataset(tmp_path)
    output_dir = tmp_path / "setup"

    main(
        [
            "--dataset",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--write-trajectory-template",
        ]
    )

    assert (output_dir / "stage3_manifest.json").is_file()
    assert (output_dir / "estimated_camera_trajectory.txt").is_file()


def test_camera_to_world_pose_uses_expected_trajectory_convention():
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]

    pose = _trajectory_pose_from_camera_to_world(1.25, camera_to_world)

    assert pose.timestamp == 1.25
    np.testing.assert_allclose(pose.translation, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose.quaternion_xyzw, [0.0, 0.0, 0.0, 1.0])


def test_quaternion_conversion_for_z_rotation():
    angle = np.deg2rad(90.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    quaternion = _quaternion_xyzw_from_rotation(rotation)

    np.testing.assert_allclose(
        quaternion,
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        atol=1e-12,
    )


def test_write_odometry_report(tmp_path):
    report_path = tmp_path / "reports" / "rgbd_odometry_report.csv"

    write_odometry_report(
        report_path,
        [
            Stage3OdometryReport(
                timestamp=0.0,
                reference_timestamp=None,
                matched_features=0,
                valid_depth_points=0,
                pnp_inliers=0,
                status="origin",
            ),
            Stage3OdometryReport(
                timestamp=0.1,
                reference_timestamp=0.0,
                matched_features=100,
                valid_depth_points=90,
                pnp_inliers=80,
                status="tracked",
            ),
        ],
    )

    lines = report_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "timestamp,reference_timestamp,matched_features,"
        "valid_depth_points,pnp_inliers,status"
    )
    assert lines[1] == "0,,0,0,0,origin"
    assert lines[2] == "0.1,0,100,90,80,tracked"


def test_evaluate_translation_trajectory_and_write_metrics(tmp_path):
    ground_truth = [
        TrajectoryPose(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(0.1, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
    ]
    estimated = [
        TrajectoryPose(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(0.1, [1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
    ]

    metrics = evaluate_translation_trajectory(ground_truth, estimated)
    output_path = tmp_path / "trajectory_metrics.json"
    write_trajectory_metrics(output_path, metrics)

    assert metrics.matched_poses == 2
    assert metrics.translation_max == 1.0
    assert json.loads(output_path.read_text(encoding="utf-8"))["matched_poses"] == 2


def test_rigid_transform_estimates_known_motion():
    source = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    translation = np.array([0.1, -0.2, 0.3])
    target = source + translation

    transform = _rigid_transform(source, target)

    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], translation, atol=1e-12)


def test_rigid_transform_ransac_rejects_outlier():
    source = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 2.0],
        ]
    )
    translation = np.array([0.02, -0.01, 0.03])
    target = source + translation
    target[-1] += np.array([1.0, 1.0, 1.0])

    transform, inliers = _estimate_rigid_transform_ransac(
        source,
        target,
        min_inliers=4,
        iterations=50,
        threshold=0.05,
    )

    assert np.count_nonzero(inliers) == 4
    np.testing.assert_allclose(transform[:3, 3], translation, atol=1e-12)


def test_apply_loop_closure_closes_final_pose_to_first():
    poses = [
        TrajectoryPose(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(2.0, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
    ]

    closed, summary = apply_loop_closure(poses, window_fraction=1.0)

    np.testing.assert_allclose(closed[0].translation, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(closed[-1].translation, [0.0, 0.0, 0.0], atol=1e-12)
    assert summary.original_end_translation_error == 2.0
    assert summary.corrected_end_translation_error < 1e-12


def test_apply_loop_closure_can_leave_early_window_unchanged():
    poses = [
        TrajectoryPose(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(1.0, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(2.0, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        TrajectoryPose(3.0, [3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
    ]

    closed, _ = apply_loop_closure(poses, window_fraction=0.5)

    np.testing.assert_allclose(closed[0].translation, poses[0].translation)
    np.testing.assert_allclose(closed[1].translation, poses[1].translation)
    np.testing.assert_allclose(closed[-1].translation, poses[0].translation)


def test_loop_closure_cli_post_processes_existing_trajectory(tmp_path):
    _write_stage3_dataset(tmp_path)
    input_path = tmp_path / "estimated_camera_trajectory.txt"
    write_trajectory(
        input_path,
        [
            TrajectoryPose(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            TrajectoryPose(0.033333, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    output_dir = tmp_path / "closed"

    main(
        [
            "--dataset",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--input-trajectory",
            str(input_path),
            "--apply-loop-closure",
        ]
    )

    assert (output_dir / "estimated_camera_trajectory_loop_closed.txt").is_file()
    assert (output_dir / "loop_closure_summary.json").is_file()


def test_write_loop_closure_summary(tmp_path):
    _, summary = apply_loop_closure(
        [
            TrajectoryPose(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            TrajectoryPose(1.0, [0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ]
    )
    output_path = tmp_path / "loop_closure_summary.json"

    write_loop_closure_summary(output_path, summary)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["poses"] == 2
    assert data["target"] == "first"
