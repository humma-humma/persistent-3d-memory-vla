import json

import numpy as np
import pytest

from persistent_scene_memory.causal_mapping import (
    StableTrackRegistry,
    _points_observed_in,
    _quality_summary,
    run_causal_mapping,
)
from sfm_reconstruction.dataset import load_stage1_dataset
from sfm_reconstruction.geometry import project_points
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.open3d_viewer import load_ascii_ply_table
from sfm_reconstruction.reconstruction import ReconstructionConfig, ReconstructionResult


def test_stable_registry_survives_internal_track_reordering():
    first = ReconstructionResult(
        poses={},
        points={0: np.array([0, 0, 1]), 1: np.array([1, 0, 1])},
        tracks=[
            Track({0: np.array([10.0, 10.0]), 1: np.array([11.0, 10.0])}),
            Track({0: np.array([20.0, 20.0]), 1: np.array([21.0, 20.0])}),
        ],
        initial_pair=(0, 1),
        skipped_track_conflicts=0,
    )
    second = ReconstructionResult(
        poses={},
        points={0: np.array([1, 0, 1]), 1: np.array([0, 0, 1])},
        tracks=[
            Track({0: np.array([20.0, 20.0]), 1: np.array([21.0, 20.0])}),
            Track({0: np.array([10.0, 10.0]), 1: np.array([11.0, 10.0])}),
        ],
        initial_pair=(0, 1),
        skipped_track_conflicts=0,
    )
    registry = StableTrackRegistry()

    first_ids = registry.assign(first)
    second_ids = registry.assign(second)

    assert first_ids == {0: 0, 1: 1}
    assert second_ids == {0: 1, 1: 0}


def test_quality_summary_reports_calibration_and_brier():
    summary = _quality_summary(
        np.array([0.9, 0.2]),
        np.array([0.05, 0.5]),
        0.15,
    )

    assert summary["fraction_correct"] == 0.5
    assert summary["brier"] == pytest.approx(0.025)
    assert sum(item["count"] for item in summary["calibration_bins"]) == 2


def test_unregistered_camera_cannot_update_world_memory():
    result = ReconstructionResult(
        poses={0: Pose.identity(), 1: Pose.identity()},
        points={0: np.array([0, 0, 1])},
        tracks=[Track({0: np.array([1, 1]), 2: np.array([2, 2])})],
        initial_pair=(0, 1),
        skipped_track_conflicts=0,
    )

    assert _points_observed_in(result, (2,)).points == {}


def test_causal_prefix_pipeline_uses_only_available_images(tmp_path):
    rng = np.random.default_rng(23)
    intrinsics = np.array(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points = np.column_stack(
        (rng.uniform(-1, 1, 40), rng.uniform(-0.7, 0.7, 40), rng.uniform(4, 7, 40))
    )
    poses = {
        image_id: Pose(np.eye(3), [-0.4 * image_id, 0.0, 0.0])
        for image_id in range(3)
    }
    images = tmp_path / "images"
    correspondences = tmp_path / "correspondences"
    images.mkdir()
    correspondences.mkdir()
    projections = {}
    for image_id, pose in poses.items():
        (images / f"{image_id:05d}.jpg").touch()
        projections[image_id] = project_points(points, pose, intrinsics)
    np.savetxt(
        correspondences / "0_1.txt",
        np.hstack((projections[0], projections[1])),
    )
    np.savetxt(
        correspondences / "1_2.txt",
        np.hstack((projections[1][:30], projections[2][:30])),
    )
    (tmp_path / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": intrinsics.tolist(),
                "extrinsics": {
                    f"{image_id:05d}.jpg": pose.matrix().tolist()
                    for image_id, pose in poses.items()
                },
            }
        )
    )
    output = tmp_path / "causal"

    report = run_causal_mapping(
        load_stage1_dataset(tmp_path),
        output,
        ReconstructionConfig(
            bundle_adjustment=False,
            min_pnp_points=8,
            min_pnp_inliers=8,
            max_reprojection_error=1.0,
            pnp_reprojection_error=1.0,
            min_triangulation_angle=0.25,
        ),
    )

    assert [row["available_image_ids"] for row in report["frames"]] == [[0, 1], [0, 1, 2]]
    first = load_ascii_ply_table(
        output / "causal_snapshots" / "frame_000000" / "estimated_points_rich.ply"
    )
    assert np.max(first.property("registered_observations")) == 2
    memory = json.loads(
        (output / "causal_memory" / "memory_frame_000001.json").read_text()
    )
    assert sum(memory["frame_index"] - token["last_seen"] == 1 for token in memory["tokens"]) == 10
    assert report["stable_identity"]["split_identifiers"] == 0
    assert (output / "causal_validation.json").is_file()
