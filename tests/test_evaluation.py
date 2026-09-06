import numpy as np

from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.models import Pose


def test_pose_evaluation_removes_global_translation_scale():
    estimated = {
        0: Pose.identity(),
        1: Pose(np.eye(3), [-2.0, 0.0, 0.0]),
        2: Pose(np.eye(3), [-4.0, 0.0, 0.0]),
    }
    ground_truth = {
        0: np.eye(4),
        1: np.array(
            [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        ),
        2: np.array(
            [[1, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        ),
    }

    metrics = evaluate_poses(estimated, ground_truth)

    assert metrics is not None
    assert metrics.scale == 2.0
    assert metrics.mean_rotation_degrees == 0.0
    assert metrics.mean_translation == 0.0
