import cv2
import numpy as np

from sfm_reconstruction.dataset import Stage1Dataset
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.reconstruction import ReconstructionResult
from sfm_reconstruction.reprojection_diagnostics import (
    per_camera_reprojection_rows,
    reprojection_observation_rows,
    write_reprojection_diagnostics,
)


def test_reprojection_diagnostics_write_residuals_and_overlay(tmp_path):
    image_path = tmp_path / "00000.jpg"
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    dataset = Stage1Dataset(
        root=tmp_path,
        intrinsics=np.eye(3),
        image_paths={0: image_path},
        image_names={0: image_path.name},
        correspondence_paths={},
        ground_truth_extrinsics={},
    )
    result = ReconstructionResult(
        poses={0: Pose.identity()},
        points={0: np.array([5.0, 6.0, 1.0])},
        tracks=[Track(observations={0: np.array([5.0, 6.0])})],
        initial_pair=(0, 0),
        skipped_track_conflicts=0,
    )

    rows = reprojection_observation_rows(dataset, result)
    camera_rows = per_camera_reprojection_rows(rows)
    summary = write_reprojection_diagnostics(tmp_path / "diag", dataset, result)

    assert rows[0]["error"] == 0.0
    assert camera_rows[0]["median_error"] == 0.0
    assert summary["observations"] == 1
    assert (tmp_path / "diag" / "observations.csv").is_file()
    assert (tmp_path / "diag" / "per_camera.csv").is_file()
    assert (tmp_path / "diag" / "overlays" / "00000_reprojection.png").is_file()
