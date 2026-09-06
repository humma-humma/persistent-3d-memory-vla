import json

import numpy as np

from sfm_reconstruction.dataset import load_stage1_dataset
from sfm_reconstruction.tracks import build_tracks


def test_tracks_merge_observations_across_pairs(tmp_path):
    images = tmp_path / "images"
    correspondences = tmp_path / "correspondences"
    images.mkdir()
    correspondences.mkdir()
    for image_id in range(3):
        (images / f"{image_id:05d}.jpg").touch()
    np.savetxt(correspondences / "0_1.txt", [[10, 20, 11, 20]])
    np.savetxt(correspondences / "1_2.txt", [[11, 20, 12, 20]])
    (tmp_path / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0, 0, 1]],
                "extrinsics": {},
            }
        )
    )

    result = build_tracks(load_stage1_dataset(tmp_path))

    assert len(result.tracks) == 1
    assert set(result.tracks[0].observations) == {0, 1, 2}
    assert result.skipped_conflicts == 0
    assert result.skipped_conflicts_by_pair == {}


def test_tracks_can_require_three_observations(tmp_path):
    images = tmp_path / "images"
    correspondences = tmp_path / "correspondences"
    images.mkdir()
    correspondences.mkdir()
    for image_id in range(3):
        (images / f"{image_id:05d}.jpg").touch()
    np.savetxt(correspondences / "0_1.txt", [[10, 20, 11, 20]])
    (tmp_path / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
                "extrinsics": {},
            }
        )
    )

    result = build_tracks(
        load_stage1_dataset(tmp_path),
        min_observations=3,
    )

    assert result.tracks == []
