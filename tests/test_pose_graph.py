import json

import cv2
import numpy as np
import pytest

from sfm_reconstruction.stage3 import (
    Stage3OdometryConfig,
    TrajectoryPose,
    _transform_from_trajectory_pose,
    load_stage3_dataset,
)
from sfm_reconstruction.pose_graph import (
    LoopEdgeCandidate,
    PoseGraphConfig,
    PoseGraphEdge,
    PoseGraphNode,
    PoseGraph,
    _apply_acceptance,
    _loop_correspondences,
    _loop_candidate_pairs,
    _matrix_from_vec6,
    _vec6_from_matrix,
    build_keyframe_pose_graph,
    build_pose_graph,
    expand_keyframe_optimization,
    estimate_loop_edges,
    optimize_pose_graph,
    pose_graph_config_from_args,
    run_pose_graph_pipeline,
    select_loop_edges,
    write_loop_candidates_csv,
    write_pose_graph_edges_csv,
    write_pose_graph_summary,
)
from sfm_reconstruction.place_recognition import (
    LoopProposal,
    refine_loop_proposals,
    select_global_loop_candidates,
)


def _pose(translation, rotation_vector=(0.0, 0.0, 0.0)):
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def _trajectory_pose(timestamp, transform):
    from sfm_reconstruction.stage3 import _trajectory_pose_from_camera_to_world

    return _trajectory_pose_from_camera_to_world(timestamp, transform)


# --------------------------------------------------------------------------- #
# SE(3) helpers
# --------------------------------------------------------------------------- #
def test_vec6_matrix_roundtrip():
    transform = _pose([0.1, -0.2, 0.3], rotation_vector=[0.05, -0.1, 0.2])
    recovered = _matrix_from_vec6(_vec6_from_matrix(transform))
    np.testing.assert_allclose(recovered, transform, atol=1e-12)


# --------------------------------------------------------------------------- #
# Loop candidate selection
# --------------------------------------------------------------------------- #
def test_loop_candidate_pairs_respect_separation_and_windows():
    config = PoseGraphConfig(
        loop_source_window=40,
        loop_target_window=40,
        loop_source_stride=8,
        loop_target_stride=8,
        loop_min_separation=50,
    )
    pairs = _loop_candidate_pairs(200, config)

    assert pairs
    for source, target in pairs:
        assert source - target >= 50
        assert source >= 200 - 40
        assert target < 40


def test_learned_loop_correspondences_reuses_cached_features():
    class FakeMatcher:
        def __init__(self):
            self.extracted = []

        def extract(self, path):
            self.extracted.append(path)
            return {"path": path}

        def match(self, features0, features1):
            assert features0["path"] == "target.png"
            assert features1["path"] == "source.png"
            return (
                np.asarray([[10.0, 20.0], [30.0, 40.0]]),
                np.asarray([[11.0, 21.0], [31.0, 41.0]]),
            )

    matcher = FakeMatcher()
    config = PoseGraphConfig(loop_matcher="superpoint-lightglue")
    cache = {}

    first = _loop_correspondences(
        0, 11, "target.png", "source.png",
        Stage3OdometryConfig(), config, cache, matcher,
    )
    second = _loop_correspondences(
        0, 11, "target.png", "source.png",
        Stage3OdometryConfig(), config, cache, matcher,
    )

    np.testing.assert_array_equal(first[0], [[10.0, 20.0], [30.0, 40.0]])
    np.testing.assert_array_equal(first[1], [[11.0, 21.0], [31.0, 41.0]])
    np.testing.assert_array_equal(second[0], first[0])
    assert matcher.extracted == ["target.png", "source.png"]


def test_pose_graph_cli_accepts_learned_loop_matcher(tmp_path):
    from sfm_reconstruction.stage3 import build_parser

    args = build_parser().parse_args(
        [
            "--dataset", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--loop-matcher", "superpoint-lightglue",
            "--learned-device", "cpu",
            "--learned-filter-threshold", "0.2",
            "--loop-candidate-mode", "dinov2",
            "--retrieval-stride", "24",
            "--retrieval-min-separation", "600",
            "--retrieval-max-candidates", "12",
            "--retrieval-proposal-spacing", "32",
        ]
    )
    config = pose_graph_config_from_args(args)

    assert config.loop_matcher == "superpoint-lightglue"
    assert config.learned_device == "cpu"
    assert config.learned_filter_threshold == 0.2
    assert config.loop_candidate_mode == "dinov2"
    assert config.retrieval_stride == 24
    assert config.retrieval_min_separation == 600
    assert config.retrieval_max_candidates == 12
    assert config.retrieval_proposal_spacing == 32


def test_global_retrieval_ranks_and_diversifies_loop_candidates():
    descriptors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 0.9],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    proposals = select_global_loop_candidates(
        descriptors,
        [0, 100, 200, 300, 400],
        min_separation=150,
        max_candidates=3,
        endpoint_spacing=80,
        min_similarity=0.8,
    )

    assert [(item.source_index, item.target_index) for item in proposals] == [
        (300, 0),
        (400, 100),
    ]
    assert [item.rank for item in proposals] == [1, 2]
    assert all(item.similarity > 0.99 for item in proposals)


def test_retrieval_refinement_adds_nearby_endpoints_without_duplicates():
    proposals = [LoopProposal(100, 10, 0.9, 1), LoopProposal(200, 20, 0.8, 2)]

    refined = refine_loop_proposals(
        proposals, frame_count=250, radius=2, top_k=1, min_separation=50
    )

    pairs = {(item.source_index, item.target_index) for item in refined}
    assert (103, 10) not in pairs
    assert {(98, 10), (99, 10), (101, 10), (102, 10)} <= pairs
    assert {(100, 8), (100, 9), (100, 11), (100, 12)} <= pairs
    assert len(pairs) == len(refined)


def test_apply_acceptance_rejects_unsafe_edges():
    base = LoopEdgeCandidate(
        source_index=199,
        target_index=0,
        source_timestamp=1.0,
        target_timestamp=0.0,
        method="pnp",
        matched_features=100,
        valid_depth_points=80,
        inliers=40,
        median_reprojection_error=1.0,
        translation_norm=0.5,
        rotation_degrees=5.0,
        accepted=False,
        reject_reason="",
    )
    config = PoseGraphConfig(
        loop_min_inliers=15,
        loop_max_translation=3.0,
        loop_max_rotation_degrees=60.0,
        loop_max_reprojection_error=4.0,
    )

    assert _apply_acceptance(base, config).accepted

    too_few = _replace(base, inliers=5)
    assert not _apply_acceptance(too_few, config).accepted
    assert "inliers" in _apply_acceptance(too_few, config).reject_reason

    big_translation = _replace(base, translation_norm=10.0)
    assert not _apply_acceptance(big_translation, config).accepted

    big_rotation = _replace(base, rotation_degrees=120.0)
    assert not _apply_acceptance(big_rotation, config).accepted

    high_error = _replace(base, median_reprojection_error=9.0)
    assert not _apply_acceptance(high_error, config).accepted


def test_select_loop_edges_prefers_more_inliers_and_caps_count():
    candidates = [
        _accepted_candidate(target_index=0, inliers=20, error=2.0),
        _accepted_candidate(target_index=1, inliers=40, error=1.0),
        _accepted_candidate(target_index=2, inliers=30, error=0.5),
        LoopEdgeCandidate(
            source_index=190,
            target_index=3,
            source_timestamp=1.0,
            target_timestamp=0.0,
            method="pnp",
            matched_features=10,
            valid_depth_points=5,
            inliers=2,
            median_reprojection_error=1.0,
            translation_norm=0.1,
            rotation_degrees=1.0,
            accepted=False,
            reject_reason="too few inliers: 2",
        ),
    ]
    config = PoseGraphConfig(max_loop_edges=2)

    selected = select_loop_edges(candidates, config)

    assert len(selected) == 2
    assert selected[0].inliers == 40  # best first
    assert all(candidate.accepted for candidate in selected)


def test_select_loop_edges_can_reject_near_duplicate_endpoints():
    candidates = [
        _replace(_accepted_candidate(target_index=24, inliers=113, error=0.46), source_index=4395),
        _replace(_accepted_candidate(target_index=32, inliers=112, error=0.43), source_index=4395),
        _replace(_accepted_candidate(target_index=220, inliers=80, error=0.50), source_index=4100),
    ]
    config = PoseGraphConfig(max_loop_edges=3, loop_endpoint_spacing=100)

    selected = select_loop_edges(candidates, config)

    assert [(edge.source_index, edge.target_index) for edge in selected] == [
        (4395, 24),
        (4100, 220),
    ]


# --------------------------------------------------------------------------- #
# Pose graph construction
# --------------------------------------------------------------------------- #
def test_build_pose_graph_recovers_relative_odometry():
    poses = [
        _trajectory_pose(0.0, _pose([0.0, 0.0, 0.0])),
        _trajectory_pose(1.0, _pose([1.0, 0.0, 0.0])),
        _trajectory_pose(2.0, _pose([2.0, 0.0, 0.0])),
    ]
    loop = LoopEdgeCandidate(
        source_index=2,
        target_index=0,
        source_timestamp=2.0,
        target_timestamp=0.0,
        method="pnp",
        matched_features=50,
        valid_depth_points=40,
        inliers=30,
        median_reprojection_error=1.0,
        translation_norm=0.0,
        rotation_degrees=0.0,
        accepted=True,
        reject_reason="",
        source_from_target=_pose([0.0, 0.0, 0.0]),
    )

    graph = build_pose_graph(poses, None, [loop], PoseGraphConfig())

    assert len(graph.nodes) == 3
    assert len(graph.odometry_edges) == 2
    assert len(graph.loop_edges) == 1
    first = graph.odometry_edges[0]
    # measured target_from_source maps node0 -> node1: translation -1 in x.
    np.testing.assert_allclose(
        first.measured_target_from_source[:3, 3], [-1.0, 0.0, 0.0], atol=1e-9
    )
    loop_edge = graph.loop_edges[0]
    assert loop_edge.source_index == 0
    assert loop_edge.target_index == 2


# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #
def _manual_graph(node_transforms, edges):
    nodes = [
        PoseGraphNode(index=i, timestamp=float(i), camera_to_world=t)
        for i, t in enumerate(node_transforms)
    ]
    return PoseGraph(nodes=nodes, edges=edges)


def test_optimize_recovers_true_poses_from_perturbed_init():
    true = [
        _pose([0.0, 0.0, 0.0]),
        _pose([1.0, 0.0, 0.0], rotation_vector=[0.0, 0.0, 0.1]),
        _pose([1.0, 1.0, 0.0], rotation_vector=[0.0, 0.0, 0.2]),
        _pose([0.0, 1.0, 0.0], rotation_vector=[0.0, 0.0, 0.1]),
    ]

    def measured(p, q):
        return np.linalg.inv(true[q]) @ true[p]

    edges = [
        PoseGraphEdge(0, 1, measured(0, 1), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(1, 2, measured(1, 2), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(2, 3, measured(2, 3), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(0, 3, measured(0, 3), 0.5, 0.5, 25, "loop"),
    ]

    rng = np.random.default_rng(0)
    init = [true[0]] + [
        _pose(
            true[i][:3, 3] + rng.normal(scale=0.1, size=3),
            rotation_vector=cv2.Rodrigues(true[i][:3, :3])[0].ravel()
            + rng.normal(scale=0.05, size=3),
        )
        for i in range(1, 4)
    ]
    graph = _manual_graph(init, edges)

    result = optimize_pose_graph(graph, PoseGraphConfig())

    assert result.success
    assert result.final_cost < result.initial_cost
    for node, expected in zip(result.poses, true):
        np.testing.assert_allclose(node.translation, expected[:3, 3], atol=1e-4)


def test_optimize_loop_edge_pulls_endpoint_toward_start():
    # VO straight line: nodes drift away to x=3; a loop edge says the last node
    # coincides with the first. Odometry alone keeps the drift; the loop edge
    # must pull the endpoint back toward the origin.
    init = [_pose([float(i), 0.0, 0.0]) for i in range(4)]

    def measured(p, q):
        return np.linalg.inv(init[q]) @ init[p]

    odometry = [
        PoseGraphEdge(0, 1, measured(0, 1), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(1, 2, measured(1, 2), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(2, 3, measured(2, 3), 1.0, 1.0, 30, "odometry"),
    ]
    loop = PoseGraphEdge(0, 3, _pose([0.0, 0.0, 0.0]), 1.0, 1.0, 30, "loop")

    without_loop = optimize_pose_graph(_manual_graph(init, odometry), PoseGraphConfig())
    with_loop = optimize_pose_graph(
        _manual_graph(init, odometry + [loop]), PoseGraphConfig()
    )

    end_without = np.linalg.norm(without_loop.poses[-1].translation)
    end_with = np.linalg.norm(with_loop.poses[-1].translation)
    assert end_without == pytest.approx(3.0, abs=1e-3)
    assert end_with < end_without


def test_optimize_keeps_first_pose_fixed():
    init = [_pose([0.0, 0.0, 0.0]), _pose([1.0, 0.0, 0.0]), _pose([2.0, 0.0, 0.0])]

    def measured(p, q):
        return np.linalg.inv(init[q]) @ init[p]

    edges = [
        PoseGraphEdge(0, 1, measured(0, 1), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(1, 2, measured(1, 2), 1.0, 1.0, 30, "odometry"),
        PoseGraphEdge(0, 2, _pose([0.0, 0.0, 0.0]), 1.0, 1.0, 30, "loop"),
    ]
    result = optimize_pose_graph(_manual_graph(init, edges), PoseGraphConfig())

    np.testing.assert_allclose(result.poses[0].translation, [0.0, 0.0, 0.0], atol=1e-12)


def test_keyframe_graph_expands_back_to_full_trajectory():
    poses = [
        _trajectory_pose(float(i), _pose([float(i), 0.0, 0.0]))
        for i in range(6)
    ]
    loop = LoopEdgeCandidate(
        source_index=5,
        target_index=0,
        source_timestamp=5.0,
        target_timestamp=0.0,
        method="pnp",
        matched_features=80,
        valid_depth_points=60,
        inliers=40,
        median_reprojection_error=0.5,
        translation_norm=0.0,
        rotation_degrees=0.0,
        accepted=True,
        reject_reason="",
        source_from_target=np.eye(4),
    )
    graph = build_keyframe_pose_graph(
        poses, [loop], PoseGraphConfig(keyframe_stride=2)
    )

    assert [node.index for node in graph.nodes] == [0, 2, 4, 5]

    optimized_keyframes = [
        _trajectory_pose(0.0, _pose([0.0, 0.0, 0.0])),
        _trajectory_pose(2.0, _pose([1.8, 0.0, 0.0])),
        _trajectory_pose(4.0, _pose([3.2, 0.0, 0.0])),
        _trajectory_pose(5.0, _pose([3.5, 0.0, 0.0])),
    ]
    expanded = expand_keyframe_optimization(poses, optimized_keyframes)

    assert len(expanded) == len(poses)
    np.testing.assert_allclose(expanded[2].translation, [1.8, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(expanded[5].translation, [3.5, 0.0, 0.0], atol=1e-9)
    assert 0.0 < expanded[1].translation[0] < 1.8


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def test_write_loop_candidates_and_graph_outputs(tmp_path):
    candidate = _accepted_candidate(target_index=0, inliers=30, error=1.0)
    rejected = _replace(candidate, accepted=False, reject_reason="too few inliers: 2")
    write_loop_candidates_csv(tmp_path / "loop_candidates.csv", [candidate, rejected])

    lines = (tmp_path / "loop_candidates.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("source_index,target_index")
    assert lines[1].endswith(",1,")  # accepted -> 1, empty reject reason
    assert "too few inliers: 2" in lines[2]

    poses = [
        _trajectory_pose(0.0, _pose([0.0, 0.0, 0.0])),
        _trajectory_pose(1.0, _pose([1.0, 0.0, 0.0])),
    ]
    graph = build_pose_graph(poses, None, [], PoseGraphConfig())
    write_pose_graph_edges_csv(tmp_path / "edges.csv", graph)
    write_pose_graph_summary(tmp_path / "summary.json", graph)

    edge_lines = (tmp_path / "edges.csv").read_text(encoding="utf-8").splitlines()
    assert edge_lines[0].startswith("source_index,target_index,edge_type")
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["nodes"] == 2
    assert summary["odometry_edges"] == 1


# --------------------------------------------------------------------------- #
# Integration: estimate a real loop edge from a synthetic RGB-D sequence
# --------------------------------------------------------------------------- #
def _write_synthetic_rgbd_dataset(root, frame_count=12):
    rgb = root / "rgb"
    depth = root / "depth"
    rgb.mkdir()
    depth.mkdir()

    rng = np.random.default_rng(0)
    height, width = 120, 160
    low = rng.integers(0, 255, size=(height // 8, width // 8), dtype=np.uint8)
    texture = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    for center in [(40, 30), (110, 40), (70, 90), (130, 95)]:
        cv2.circle(image, center, 9, (255, 255, 255), -1)
        cv2.rectangle(image, (center[0] - 6, center[1] - 6), (center[0] + 6, center[1] + 6), (0, 0, 0), 1)

    # Depth varies with column so back-projected points are not coplanar.
    columns = np.arange(width, dtype=np.uint16)
    depth_row = (900 + columns * 4).astype(np.uint16)
    depth_map = np.tile(depth_row, (height, 1))

    for index in range(frame_count):
        timestamp = f"{index / 30.0:.6f}"
        cv2.imwrite(str(rgb / f"{timestamp}.png"), image)
        cv2.imwrite(str(depth / f"{timestamp}.png"), depth_map)

    (root / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": [
                    [120.0, 0.0, 80.0],
                    [0.0, 120.0, 60.0],
                    [0.0, 0.0, 1.0],
                ]
            }
        ),
        encoding="utf-8",
    )


def test_estimate_loop_edges_on_synthetic_sequence(tmp_path):
    _write_synthetic_rgbd_dataset(tmp_path, frame_count=12)
    dataset = load_stage3_dataset(tmp_path)
    odometry_config = Stage3OdometryConfig(min_matches=8, min_pnp_points=8, min_pnp_inliers=6)
    config = PoseGraphConfig(
        loop_source_window=4,
        loop_target_window=4,
        loop_source_stride=2,
        loop_target_stride=2,
        loop_min_separation=6,
        loop_min_matches=8,
        loop_min_depth_points=8,
        loop_min_inliers=8,
        keyframe_stride=3,
    )

    candidates = estimate_loop_edges(dataset, odometry_config, config)
    accepted = select_loop_edges(candidates, config)

    assert candidates
    assert accepted, "identical first/last frames should yield an accepted loop edge"
    best = accepted[0]
    # Identical frames -> near-identity relative transform.
    assert best.translation_norm < 0.2
    assert best.rotation_degrees < 5.0


def test_build_pose_graph_tolerates_empty_reports():
    poses = [
        _trajectory_pose(0.0, _pose([0.0, 0.0, 0.0])),
        _trajectory_pose(1.0, _pose([1.0, 0.0, 0.0])),
    ]
    graph = build_pose_graph(poses, [], [], PoseGraphConfig())
    assert len(graph.odometry_edges) == 1
    assert graph.odometry_edges[0].inlier_count == 0


def test_main_reuses_input_trajectory_for_pose_graph(tmp_path):
    from sfm_reconstruction.stage3 import main, write_trajectory

    _write_synthetic_rgbd_dataset(tmp_path, frame_count=12)
    # Pre-existing VO trajectory (all origin poses) to be reused, not recomputed.
    vo_path = tmp_path / "vo.txt"
    write_trajectory(
        vo_path,
        [
            _trajectory_pose(i / 30.0, _pose([0.0, 0.0, 0.0]))
            for i in range(12)
        ],
    )
    output_dir = tmp_path / "out"

    main(
        [
            "--dataset", str(tmp_path),
            "--output-dir", str(output_dir),
            "--input-trajectory", str(vo_path),
            "--run-pose-graph",
            "--loop-source-window", "4", "--loop-target-window", "4",
            "--loop-source-stride", "2", "--loop-target-stride", "2",
            "--loop-min-separation", "6", "--loop-min-matches", "8",
            "--loop-min-depth-points", "8", "--loop-min-inliers", "8",
        ]
    )

    assert (output_dir / "estimated_camera_trajectory_pose_graph.txt").is_file()
    assert (output_dir / "pose_graph_summary.json").is_file()


def test_run_pose_graph_pipeline_writes_all_outputs(tmp_path):
    _write_synthetic_rgbd_dataset(tmp_path, frame_count=12)
    dataset = load_stage3_dataset(tmp_path)
    odometry_config = Stage3OdometryConfig(min_matches=8, min_pnp_points=8, min_pnp_inliers=6)
    config = PoseGraphConfig(
        loop_source_window=4,
        loop_target_window=4,
        loop_source_stride=2,
        loop_target_stride=2,
        loop_min_separation=6,
        loop_min_matches=8,
        loop_min_depth_points=8,
        loop_min_inliers=8,
        keyframe_stride=3,
    )
    output_dir = tmp_path / "out"

    pipeline = run_pose_graph_pipeline(
        dataset, odometry_config, config, output_dir
    )

    for name in (
        "loop_candidates.csv",
        "loop_candidate_summary.json",
        "pose_graph_edges.csv",
        "pose_graph_summary.json",
        "estimated_camera_trajectory_pose_graph.txt",
        "pose_graph_optimization_summary.json",
    ):
        assert (output_dir / name).is_file(), name
    assert len(pipeline.optimization.poses) == len(dataset.frames)
    assert len(pipeline.graph.nodes) < len(dataset.frames)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _replace(candidate, **changes):
    from sfm_reconstruction.pose_graph import _replace_candidate

    return _replace_candidate(candidate, **changes)


def _accepted_candidate(target_index, inliers, error):
    return LoopEdgeCandidate(
        source_index=190 + target_index,
        target_index=target_index,
        source_timestamp=1.0,
        target_timestamp=0.0,
        method="pnp",
        matched_features=100,
        valid_depth_points=80,
        inliers=inliers,
        median_reprojection_error=error,
        translation_norm=0.2,
        rotation_degrees=2.0,
        accepted=True,
        reject_reason="",
        source_from_target=np.eye(4),
    )
