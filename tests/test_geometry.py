import numpy as np

from sfm_reconstruction.geometry import (
    project_points,
    solve_pnp,
    triangulate_points,
)
from sfm_reconstruction.models import Pose


def test_triangulation_and_pnp_recover_synthetic_geometry():
    rng = np.random.default_rng(3)
    intrinsics = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 80),
            rng.uniform(-0.7, 0.7, 80),
            rng.uniform(4.0, 8.0, 80),
        )
    )
    first_pose = Pose.identity()
    second_pose = Pose(np.eye(3), np.array([-0.8, 0.0, 0.0]))
    first_pixels = project_points(points, first_pose, intrinsics)
    second_pixels = project_points(points, second_pose, intrinsics)

    triangulated = triangulate_points(
        first_pixels, second_pixels, first_pose, second_pose, intrinsics
    )
    pnp = solve_pnp(points, second_pixels, intrinsics, reprojection_error=1.0)

    assert np.max(np.linalg.norm(triangulated - points, axis=1)) < 1e-8
    assert np.linalg.norm(pnp.pose.translation.ravel() - [-0.8, 0.0, 0.0]) < 1e-5
    assert pnp.inliers.sum() == len(points)
