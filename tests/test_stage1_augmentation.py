import numpy as np

from sfm_reconstruction.geometry import project_points
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.stage1_augmentation import (
    Stage1AugmentationConfig,
    triangulate_learned_track,
)


def test_triangulate_learned_track_uses_fixed_multiview_poses():
    intrinsics = np.asarray(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
    )
    poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), [-1.0, 0.0, 0.0]),
        2: Pose(np.eye(3), [0.0, -1.0, 0.0]),
    }
    expected = np.asarray([0.2, -0.1, 5.0])
    track = Track(
        observations={
            image_id: project_points(expected.reshape(1, 3), pose, intrinsics)[0]
            for image_id, pose in poses.items()
        }
    )

    result = triangulate_learned_track(
        track, poses, intrinsics, Stage1AugmentationConfig()
    )

    assert result.reject_reason == ""
    np.testing.assert_allclose(result.point, expected, atol=1e-6)
