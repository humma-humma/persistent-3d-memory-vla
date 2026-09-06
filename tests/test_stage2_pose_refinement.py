from types import SimpleNamespace

import numpy as np

from sfm_reconstruction.models import Pose
from sfm_reconstruction.stage2_pose_refinement import (
    Stage2PoseRefinementConfig,
    build_stage2_pose_graph,
)


def test_build_stage2_pose_graph_adds_consistent_wide_constraint(
    tmp_path, monkeypatch
):
    path = tmp_path / "0_2.txt"
    np.savetxt(path, np.zeros((6, 4)))
    poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), [-1.0, 0.0, 0.0]),
        2: Pose(np.eye(3), [-2.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(
        "sfm_reconstruction.stage2_pose_refinement.estimate_relative_pose",
        lambda *args, **kwargs: SimpleNamespace(
            pose=Pose(np.eye(3), [-1.0, 0.0, 0.0]),
            inliers=np.ones(6, dtype=bool),
        ),
    )

    graph, diagnostics = build_stage2_pose_graph(
        poses,
        np.eye(3),
        [path],
        Stage2PoseRefinementConfig(min_inliers=5),
    )

    assert len(graph.odometry_edges) == 2
    assert len(graph.loop_edges) == 1
    assert diagnostics[0]["accepted"] == 1
    assert np.linalg.norm(
        graph.loop_edges[0].measured_target_from_source[:3, 3]
    ) == 2.0


def test_build_stage2_pose_graph_rejects_opposite_translation(tmp_path, monkeypatch):
    path = tmp_path / "0_1.txt"
    np.savetxt(path, np.zeros((6, 4)))
    poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), [-1.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(
        "sfm_reconstruction.stage2_pose_refinement.estimate_relative_pose",
        lambda *args, **kwargs: SimpleNamespace(
            pose=Pose(np.eye(3), [1.0, 0.0, 0.0]),
            inliers=np.ones(6, dtype=bool),
        ),
    )

    graph, diagnostics = build_stage2_pose_graph(
        poses,
        np.eye(3),
        [path],
        Stage2PoseRefinementConfig(min_inliers=5),
    )

    assert not graph.loop_edges
    assert diagnostics[0]["accepted"] == 0
    assert "translation direction" in diagnostics[0]["reject_reason"]
