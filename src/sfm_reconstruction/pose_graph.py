"""Stage 3 pose-graph SLAM.

This module turns the RGB-D visual-odometry baseline in :mod:`stage3` into a
real pose-graph optimization:

1. Estimate end-to-start loop edges from RGB-D data (PnP or 3D-3D RANSAC).
2. Build a pose graph: nodes are camera-to-world poses, edges are relative
   transforms (frame-to-frame odometry plus accepted loop edges).
3. Optimize all poses with :func:`scipy.optimize.least_squares`, fixing the
   first pose at the origin, reusing the sparsity pattern from
   :mod:`bundle_adjustment`.
4. Export the optimized trajectory and metrics in assignment format.

Edge convention
---------------
A trajectory pose is a ``camera_to_world`` 4x4 transform.  An edge between a
``source`` node ``p`` and a ``target`` node ``q`` stores the measured relative
transform ``q_from_p`` which maps points from camera ``p`` coordinates to camera
``q`` coordinates.  The constraint between the two camera-to-world poses is::

    pose_p = pose_q @ q_from_p
    predicted q_from_p = inv(pose_q) @ pose_p

This matches the odometry update in :mod:`stage3`, where for a frame transition
``previous -> current`` the estimated ``current_from_previous`` satisfies
``current_camera_to_world = previous_camera_to_world @ inv(current_from_previous)``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import reprojection_errors, solve_pnp
from .learned_matching import create_superpoint_lightglue_matcher
from .matching import match_descriptors
from .place_recognition import (
    create_dinov2_retriever,
    refine_loop_proposals,
    select_global_loop_candidates,
)
from .stage3 import (
    Stage3Dataset,
    Stage3OdometryConfig,
    Stage3OdometryReport,
    Stage3OdometryResult,
    Stage3TrajectoryMetrics,
    TrajectoryPose,
    _build_rgbd_3d_correspondences,
    _build_rgbd_pnp_correspondences,
    _estimate_rigid_transform_ransac,
    _extract_stage3_features,
    _format_float,
    _rigid_residuals,
    _slerp_rotation,
    _trajectory_pose_from_camera_to_world,
    _transform_from_pose,
    _transform_from_trajectory_pose,
    estimate_rgbd_trajectory,
    evaluate_translation_trajectory,
    write_trajectory,
    write_trajectory_metrics,
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PoseGraphConfig:
    """Tuning for loop-edge estimation and pose-graph optimization."""

    # Loop-candidate selection.
    loop_method: str = "pnp"  # "pnp" or "rgbd"
    loop_source_window: int = 40  # last N frames considered as loop sources
    loop_target_window: int = 40  # first N frames considered as loop targets
    loop_source_stride: int = 8
    loop_target_stride: int = 8
    loop_min_separation: int = 50  # source/target index gap to count as a loop
    max_loop_edges: int = 1
    loop_endpoint_spacing: int = 0

    # Candidate generation: fixed end/start windows or global DINOv2 retrieval.
    loop_candidate_mode: str = "window"  # "window" or "dinov2"
    retrieval_stride: int = 16
    retrieval_min_separation: int = 500
    retrieval_max_candidates: int = 50
    retrieval_proposal_spacing: int = 16
    retrieval_refine_radius: int = 4
    retrieval_refine_top_k: int = 10
    retrieval_min_similarity: float = 0.6
    retrieval_batch_size: int = 16
    retrieval_device: str = "auto"

    # Loop correspondence frontend. Odometry remains classical.
    loop_matcher: str = "classical"  # "classical" or "superpoint-lightglue"
    learned_device: str = "auto"
    learned_filter_threshold: float = 0.1

    # Loop-edge acceptance thresholds.
    loop_min_matches: int = 30
    loop_min_depth_points: int = 12
    loop_min_inliers: int = 15
    loop_max_translation: float = 3.0
    loop_max_rotation_degrees: float = 60.0
    loop_max_reprojection_error: float = 4.0

    # Pose-graph edge weights.
    odometry_translation_weight: float = 1.0
    odometry_rotation_weight: float = 1.0
    loop_translation_weight: float = 0.5
    loop_rotation_weight: float = 0.5

    # Optimization.
    keyframe_stride: int = 1
    optimizer_loss: str = "soft_l1"
    optimizer_f_scale: float = 1.0
    optimizer_max_nfev: int = 200


@dataclass(frozen=True)
class LoopEdgeCandidate:
    """A scored end-to-start loop-closure candidate."""

    source_index: int
    target_index: int
    source_timestamp: float
    target_timestamp: float
    method: str
    matched_features: int
    valid_depth_points: int
    inliers: int
    median_reprojection_error: float
    translation_norm: float
    rotation_degrees: float
    accepted: bool
    reject_reason: str
    matcher: str = "classical"
    retrieval_similarity: float = float("nan")
    proposal_rank: int = -1
    # measured transform q_from_p with p = target (early), q = source (late):
    # source_from_target maps target-camera points into source-camera points.
    source_from_target: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass(frozen=True)
class PoseGraphNode:
    index: int
    timestamp: float
    camera_to_world: np.ndarray  # 4x4


@dataclass(frozen=True)
class PoseGraphEdge:
    source_index: int  # p
    target_index: int  # q
    measured_target_from_source: np.ndarray  # q_from_p, 4x4
    translation_weight: float
    rotation_weight: float
    inlier_count: int
    edge_type: str  # "odometry" or "loop"


@dataclass(frozen=True)
class PoseGraph:
    nodes: list[PoseGraphNode]
    edges: list[PoseGraphEdge]

    @property
    def loop_edges(self) -> list[PoseGraphEdge]:
        return [edge for edge in self.edges if edge.edge_type == "loop"]

    @property
    def odometry_edges(self) -> list[PoseGraphEdge]:
        return [edge for edge in self.edges if edge.edge_type == "odometry"]


@dataclass(frozen=True)
class PoseGraphOptimizationResult:
    poses: list[TrajectoryPose]
    iterations: int
    initial_cost: float
    final_cost: float
    success: bool
    message: str


# --------------------------------------------------------------------------- #
# Small SE(3) helpers
# --------------------------------------------------------------------------- #
def _matrix_from_vec6(vec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(vec[:3], dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(vec[3:6], dtype=np.float64)
    return transform


def _vec6_from_matrix(transform: np.ndarray) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(np.asarray(transform[:3, :3], dtype=np.float64))
    return np.concatenate(
        [rotation_vector.ravel(), np.asarray(transform[:3, 3], dtype=np.float64)]
    )


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    trace = float(np.trace(np.asarray(rotation, dtype=np.float64)[:3, :3]))
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# Priority 1: visual loop-edge estimation
# --------------------------------------------------------------------------- #
def _loop_candidate_pairs(
    frame_count: int, config: PoseGraphConfig
) -> list[tuple[int, int]]:
    """Pairs of (source late-index, target early-index) to evaluate."""
    if frame_count < 2:
        return []
    source_window = min(config.loop_source_window, frame_count)
    target_window = min(config.loop_target_window, frame_count)
    source_stride = max(config.loop_source_stride, 1)
    target_stride = max(config.loop_target_stride, 1)

    source_indices = list(
        range(frame_count - 1, frame_count - 1 - source_window, -source_stride)
    )
    target_indices = list(range(0, target_window, target_stride))

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for source in source_indices:
        for target in target_indices:
            if source - target < config.loop_min_separation:
                continue
            key = (source, target)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _estimate_loop_candidate(
    dataset: Stage3Dataset,
    source_index: int,
    target_index: int,
    odometry_config: Stage3OdometryConfig,
    config: PoseGraphConfig,
    feature_cache: dict[int, object],
    learned_matcher=None,
    retrieval_similarity: float = float("nan"),
    proposal_rank: int = -1,
) -> LoopEdgeCandidate:
    source_frame = dataset.frames[source_index]
    target_frame = dataset.frames[target_index]

    base = LoopEdgeCandidate(
        source_index=source_index,
        target_index=target_index,
        source_timestamp=source_frame.timestamp,
        target_timestamp=target_frame.timestamp,
        method=config.loop_method,
        matched_features=0,
        valid_depth_points=0,
        inliers=0,
        median_reprojection_error=float("inf"),
        translation_norm=0.0,
        rotation_degrees=0.0,
        accepted=False,
        reject_reason="not evaluated",
        matcher=config.loop_matcher,
        retrieval_similarity=retrieval_similarity,
        proposal_rank=proposal_rank,
        source_from_target=np.eye(4, dtype=np.float64),
    )

    try:
        if target_frame.depth_path is None:
            return _reject(base, "target frame has no depth map")
        if config.loop_method == "rgbd" and source_frame.depth_path is None:
            return _reject(base, "source frame has no depth map")

        target_points, source_points = _loop_correspondences(
            target_index,
            source_index,
            target_frame.rgb_path,
            source_frame.rgb_path,
            odometry_config,
            config,
            feature_cache,
            learned_matcher,
        )
        matched_features = len(target_points)
        if matched_features < config.loop_min_matches:
            return _reject(
                base,
                f"too few matches: {matched_features}",
                matched_features=matched_features,
            )
        if config.loop_method == "rgbd":
            points_target, points_source = _build_rgbd_3d_correspondences(
                target_points,
                source_points,
                target_frame.depth_path,
                source_frame.depth_path,
                dataset.intrinsics,
                odometry_config,
            )
            valid_depth_points = len(points_target)
            if valid_depth_points < config.loop_min_depth_points:
                return _reject(
                    base,
                    f"too few depth pairs: {valid_depth_points}",
                    matched_features=matched_features,
                    valid_depth_points=valid_depth_points,
                )
            source_from_target, inlier_mask = _estimate_rigid_transform_ransac(
                points_target,
                points_source,
                min_inliers=config.loop_min_inliers,
                iterations=odometry_config.ransac_iterations,
                threshold=odometry_config.ransac_3d_threshold,
            )
            inliers = int(np.count_nonzero(inlier_mask))
            residuals = _rigid_residuals(
                source_from_target, points_target, points_source
            )
            median_error = float(np.median(residuals[inlier_mask])) if inliers else float("inf")
        else:
            points_3d, points_2d = _build_rgbd_pnp_correspondences(
                target_points,
                source_points,
                target_frame.depth_path,
                dataset.intrinsics,
                odometry_config,
            )
            valid_depth_points = len(points_3d)
            if valid_depth_points < config.loop_min_depth_points:
                return _reject(
                    base,
                    f"too few depth points: {valid_depth_points}",
                    matched_features=matched_features,
                    valid_depth_points=valid_depth_points,
                )
            estimate = solve_pnp(
                points_3d,
                points_2d,
                dataset.intrinsics,
                reprojection_error=odometry_config.pnp_reprojection_error,
            )
            inlier_mask = estimate.inliers
            inliers = int(np.count_nonzero(inlier_mask))
            source_from_target = _transform_from_pose(estimate.pose)
            if inliers:
                errors = reprojection_errors(
                    points_3d[inlier_mask],
                    points_2d[inlier_mask],
                    estimate.pose,
                    dataset.intrinsics,
                )
                median_error = float(np.median(errors))
            else:
                median_error = float("inf")

        translation_norm = float(np.linalg.norm(source_from_target[:3, 3]))
        rotation_degrees = _rotation_angle_degrees(source_from_target)

        scored = LoopEdgeCandidate(
            source_index=source_index,
            target_index=target_index,
            source_timestamp=source_frame.timestamp,
            target_timestamp=target_frame.timestamp,
            method=config.loop_method,
            matched_features=matched_features,
            valid_depth_points=valid_depth_points,
            inliers=inliers,
            median_reprojection_error=median_error,
            translation_norm=translation_norm,
            rotation_degrees=rotation_degrees,
            accepted=False,
            reject_reason="",
            matcher=config.loop_matcher,
            retrieval_similarity=retrieval_similarity,
            proposal_rank=proposal_rank,
            source_from_target=source_from_target,
        )
        return _apply_acceptance(scored, config)
    except (RuntimeError, ValueError, cv2.error) as exc:
        return _reject(base, f"estimation error: {exc}")


def _apply_acceptance(
    candidate: LoopEdgeCandidate, config: PoseGraphConfig
) -> LoopEdgeCandidate:
    if candidate.inliers < config.loop_min_inliers:
        return _reject(candidate, f"too few inliers: {candidate.inliers}")
    if candidate.translation_norm > config.loop_max_translation:
        return _reject(
            candidate,
            f"translation {candidate.translation_norm:.3f} exceeds "
            f"{config.loop_max_translation:.3f}",
        )
    if candidate.rotation_degrees > config.loop_max_rotation_degrees:
        return _reject(
            candidate,
            f"rotation {candidate.rotation_degrees:.3f} deg exceeds "
            f"{config.loop_max_rotation_degrees:.3f} deg",
        )
    if candidate.median_reprojection_error > config.loop_max_reprojection_error:
        return _reject(
            candidate,
            f"median error {candidate.median_reprojection_error:.3f} exceeds "
            f"{config.loop_max_reprojection_error:.3f}",
        )
    return _replace_candidate(candidate, accepted=True, reject_reason="")


def _reject(
    candidate: LoopEdgeCandidate,
    reason: str,
    *,
    matched_features: int | None = None,
    valid_depth_points: int | None = None,
) -> LoopEdgeCandidate:
    return _replace_candidate(
        candidate,
        accepted=False,
        reject_reason=reason,
        matched_features=(
            candidate.matched_features
            if matched_features is None
            else matched_features
        ),
        valid_depth_points=(
            candidate.valid_depth_points
            if valid_depth_points is None
            else valid_depth_points
        ),
    )


def _replace_candidate(candidate: LoopEdgeCandidate, **changes) -> LoopEdgeCandidate:
    data = {
        "source_index": candidate.source_index,
        "target_index": candidate.target_index,
        "source_timestamp": candidate.source_timestamp,
        "target_timestamp": candidate.target_timestamp,
        "method": candidate.method,
        "matched_features": candidate.matched_features,
        "valid_depth_points": candidate.valid_depth_points,
        "inliers": candidate.inliers,
        "median_reprojection_error": candidate.median_reprojection_error,
        "translation_norm": candidate.translation_norm,
        "rotation_degrees": candidate.rotation_degrees,
        "accepted": candidate.accepted,
        "reject_reason": candidate.reject_reason,
        "matcher": candidate.matcher,
        "retrieval_similarity": candidate.retrieval_similarity,
        "proposal_rank": candidate.proposal_rank,
        "source_from_target": candidate.source_from_target,
    }
    data.update(changes)
    return LoopEdgeCandidate(**data)


def _cached_features(cache, index, rgb_path, odometry_config):
    features = cache.get(index)
    if features is None:
        features = _extract_stage3_features(rgb_path, odometry_config)
        cache[index] = features
    return features


def _loop_correspondences(
    target_index,
    source_index,
    target_rgb_path,
    source_rgb_path,
    odometry_config,
    config,
    feature_cache,
    learned_matcher,
):
    """Return full-resolution target/source pixels for one loop pair."""
    if config.loop_matcher == "superpoint-lightglue":
        target_features = feature_cache.get(target_index)
        if target_features is None:
            target_features = learned_matcher.extract(target_rgb_path)
            feature_cache[target_index] = target_features
        source_features = feature_cache.get(source_index)
        if source_features is None:
            source_features = learned_matcher.extract(source_rgb_path)
            feature_cache[source_index] = source_features
        return learned_matcher.match(target_features, source_features)

    target_features = _cached_features(
        feature_cache, target_index, target_rgb_path, odometry_config
    )
    source_features = _cached_features(
        feature_cache, source_index, source_rgb_path, odometry_config
    )
    matches = match_descriptors(
        target_features.descriptors,
        source_features.descriptors,
        odometry_config.ratio_threshold,
        target_features.descriptor_type,
    )
    return (
        target_features.points[matches[:, 0]],
        source_features.points[matches[:, 1]],
    )


def estimate_loop_edges(
    dataset: Stage3Dataset,
    odometry_config: Stage3OdometryConfig,
    config: PoseGraphConfig | None = None,
) -> list[LoopEdgeCandidate]:
    """Estimate and score end-to-start loop-closure candidates."""
    if config is None:
        config = PoseGraphConfig()
    if config.loop_method not in {"pnp", "rgbd"}:
        raise ValueError("loop_method must be one of: pnp, rgbd")
    if config.loop_matcher not in {"classical", "superpoint-lightglue"}:
        raise ValueError(
            "loop_matcher must be one of: classical, superpoint-lightglue"
        )
    if not 0.0 <= config.learned_filter_threshold <= 1.0:
        raise ValueError("learned_filter_threshold must be between 0 and 1")

    if config.loop_candidate_mode == "window":
        proposals = [
            (source, target, float("nan"), -1)
            for source, target in _loop_candidate_pairs(len(dataset.frames), config)
        ]
    elif config.loop_candidate_mode == "dinov2":
        stride = max(config.retrieval_stride, 1)
        frame_indices = list(range(0, len(dataset.frames), stride))
        if dataset.frames and frame_indices[-1] != len(dataset.frames) - 1:
            frame_indices.append(len(dataset.frames) - 1)
        retriever = create_dinov2_retriever(
            device=config.retrieval_device,
            batch_size=config.retrieval_batch_size,
        )
        descriptors = retriever.describe(
            [dataset.frames[index].rgb_path for index in frame_indices]
        )
        retrieved = select_global_loop_candidates(
            descriptors,
            frame_indices,
            min_separation=config.retrieval_min_separation,
            max_candidates=config.retrieval_max_candidates,
            endpoint_spacing=config.retrieval_proposal_spacing,
            min_similarity=config.retrieval_min_similarity,
        )
        retrieved = refine_loop_proposals(
            retrieved,
            frame_count=len(dataset.frames),
            radius=config.retrieval_refine_radius,
            top_k=config.retrieval_refine_top_k,
            min_separation=config.retrieval_min_separation,
        )
        proposals = [
            (
                proposal.source_index,
                proposal.target_index,
                proposal.similarity,
                proposal.rank,
            )
            for proposal in retrieved
        ]
    else:
        raise ValueError("loop_candidate_mode must be one of: window, dinov2")

    feature_cache: dict[int, object] = {}
    learned_matcher = None
    if config.loop_matcher == "superpoint-lightglue":
        learned_matcher = create_superpoint_lightglue_matcher(
            max_keypoints=odometry_config.max_features,
            device=config.learned_device,
            filter_threshold=config.learned_filter_threshold,
            image_scale=odometry_config.image_scale,
        )
    candidates: list[LoopEdgeCandidate] = []
    for source_index, target_index, similarity, proposal_rank in proposals:
        candidates.append(
            _estimate_loop_candidate(
                dataset,
                source_index,
                target_index,
                odometry_config,
                config,
                feature_cache,
                learned_matcher,
                similarity,
                proposal_rank,
            )
        )
    return candidates


def select_loop_edges(
    candidates: list[LoopEdgeCandidate], config: PoseGraphConfig | None = None
) -> list[LoopEdgeCandidate]:
    """Pick the best accepted loop edges (most inliers, lowest error)."""
    if config is None:
        config = PoseGraphConfig()
    accepted = [candidate for candidate in candidates if candidate.accepted]
    accepted.sort(
        key=lambda candidate: (
            -candidate.inliers,
            candidate.median_reprojection_error,
        )
    )
    # Keep selected endpoints separated when requested; dense end-to-start
    # windows often produce many copies of the same physical loop constraint.
    selected: list[LoopEdgeCandidate] = []
    endpoint_spacing = max(config.loop_endpoint_spacing, 0)
    for candidate in accepted:
        if any(
            abs(candidate.source_index - chosen.source_index) < endpoint_spacing
            or abs(candidate.target_index - chosen.target_index) < endpoint_spacing
            for chosen in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max(config.max_loop_edges, 1):
            break
    return selected


# --------------------------------------------------------------------------- #
# Priority 2: pose-graph representation
# --------------------------------------------------------------------------- #
def build_pose_graph(
    poses: list[TrajectoryPose],
    reports: list[Stage3OdometryReport] | None,
    loop_edges: list[LoopEdgeCandidate],
    config: PoseGraphConfig | None = None,
) -> PoseGraph:
    """Build nodes plus odometry and loop edges from a VO trajectory."""
    if config is None:
        config = PoseGraphConfig()
    if len(poses) < 2:
        raise ValueError("At least two poses are required to build a pose graph")

    transforms = [_transform_from_trajectory_pose(pose) for pose in poses]
    nodes = [
        PoseGraphNode(index=index, timestamp=pose.timestamp, camera_to_world=transform)
        for index, (pose, transform) in enumerate(zip(poses, transforms))
    ]

    edges: list[PoseGraphEdge] = []
    for index in range(len(poses) - 1):
        # Odometry measurement target_from_source with p=index, q=index+1.
        measured = np.linalg.inv(transforms[index + 1]) @ transforms[index]
        inlier_count = 0
        if reports is not None and index + 1 < len(reports):
            inlier_count = reports[index + 1].pnp_inliers
        edges.append(
            PoseGraphEdge(
                source_index=index,
                target_index=index + 1,
                measured_target_from_source=measured,
                translation_weight=config.odometry_translation_weight,
                rotation_weight=config.odometry_rotation_weight,
                inlier_count=inlier_count,
                edge_type="odometry",
            )
        )

    for loop in loop_edges:
        # p = target (early), q = source (late); measured q_from_p.
        edges.append(
            PoseGraphEdge(
                source_index=loop.target_index,
                target_index=loop.source_index,
                measured_target_from_source=np.asarray(
                    loop.source_from_target, dtype=np.float64
                ),
                translation_weight=config.loop_translation_weight,
                rotation_weight=config.loop_rotation_weight,
                inlier_count=loop.inliers,
                edge_type="loop",
            )
        )
    return PoseGraph(nodes=nodes, edges=edges)


def _keyframe_indices(
    frame_count: int,
    loop_edges: list[LoopEdgeCandidate],
    config: PoseGraphConfig,
) -> list[int]:
    """Select optimized frame indices while always keeping loop endpoints."""
    stride = max(config.keyframe_stride, 1)
    indices = set(range(0, frame_count, stride))
    indices.add(frame_count - 1)
    for loop in loop_edges:
        indices.add(loop.source_index)
        indices.add(loop.target_index)
    return sorted(index for index in indices if 0 <= index < frame_count)


def build_keyframe_pose_graph(
    poses: list[TrajectoryPose],
    loop_edges: list[LoopEdgeCandidate],
    config: PoseGraphConfig | None = None,
) -> PoseGraph:
    """Build a reduced pose graph from keyframes plus loop endpoints."""
    if config is None:
        config = PoseGraphConfig()
    if len(poses) < 2:
        raise ValueError("At least two poses are required to build a pose graph")

    keyframes = _keyframe_indices(len(poses), loop_edges, config)
    transforms = [_transform_from_trajectory_pose(pose) for pose in poses]
    nodes = [
        PoseGraphNode(
            index=index,
            timestamp=poses[index].timestamp,
            camera_to_world=transforms[index],
        )
        for index in keyframes
    ]

    edges: list[PoseGraphEdge] = []
    for source, target in zip(keyframes, keyframes[1:]):
        measured = np.linalg.inv(transforms[target]) @ transforms[source]
        edges.append(
            PoseGraphEdge(
                source_index=source,
                target_index=target,
                measured_target_from_source=measured,
                translation_weight=config.odometry_translation_weight,
                rotation_weight=config.odometry_rotation_weight,
                inlier_count=0,
                edge_type="odometry",
            )
        )

    keyframe_set = set(keyframes)
    for loop in loop_edges:
        if loop.source_index not in keyframe_set or loop.target_index not in keyframe_set:
            continue
        edges.append(
            PoseGraphEdge(
                source_index=loop.target_index,
                target_index=loop.source_index,
                measured_target_from_source=np.asarray(
                    loop.source_from_target, dtype=np.float64
                ),
                translation_weight=config.loop_translation_weight,
                rotation_weight=config.loop_rotation_weight,
                inlier_count=loop.inliers,
                edge_type="loop",
            )
        )
    return PoseGraph(nodes=nodes, edges=edges)


def _interpolate_correction(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
) -> np.ndarray:
    correction = np.eye(4, dtype=np.float64)
    correction[:3, :3] = _slerp_rotation(first[:3, :3], second[:3, :3], alpha)
    correction[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
    return correction


def expand_keyframe_optimization(
    original_poses: list[TrajectoryPose],
    optimized_keyframes: list[TrajectoryPose],
) -> list[TrajectoryPose]:
    """Apply interpolated keyframe corrections to every original pose."""
    if not optimized_keyframes:
        return original_poses

    original_transforms = [
        _transform_from_trajectory_pose(pose) for pose in original_poses
    ]
    # TrajectoryPose does not carry the source frame index, so recover the
    # optimized keyframe index by timestamp against the original sequence.
    index_by_timestamp = {
        f"{pose.timestamp:.9f}": index for index, pose in enumerate(original_poses)
    }
    optimized_indices: list[int] = []
    corrections: dict[int, np.ndarray] = {}
    for pose in optimized_keyframes:
        index = index_by_timestamp[f"{pose.timestamp:.9f}"]
        optimized_indices.append(index)
        optimized = _transform_from_trajectory_pose(pose)
        corrections[index] = optimized @ np.linalg.inv(original_transforms[index])

    optimized_indices = sorted(set(optimized_indices))
    expanded: list[TrajectoryPose] = []
    for index, original in enumerate(original_poses):
        if index <= optimized_indices[0]:
            correction = corrections[optimized_indices[0]]
        elif index >= optimized_indices[-1]:
            correction = corrections[optimized_indices[-1]]
        elif index in corrections:
            correction = corrections[index]
        else:
            right_pos = int(np.searchsorted(optimized_indices, index))
            right = optimized_indices[right_pos]
            left = optimized_indices[right_pos - 1]
            alpha = (index - left) / float(right - left)
            correction = _interpolate_correction(
                corrections[left], corrections[right], alpha
            )
        expanded.append(
            _trajectory_pose_from_camera_to_world(
                original.timestamp, correction @ original_transforms[index]
            )
        )
    return expanded


# --------------------------------------------------------------------------- #
# Priority 3: pose-graph optimization
# --------------------------------------------------------------------------- #
def optimize_pose_graph(
    graph: PoseGraph, config: PoseGraphConfig | None = None
) -> PoseGraphOptimizationResult:
    """Optimize all camera poses, fixing node 0 at its initial pose."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    if config is None:
        config = PoseGraphConfig()
    if len(graph.nodes) < 2:
        raise ValueError("At least two nodes are required for optimization")
    if not graph.edges:
        raise ValueError("Pose graph has no edges to optimize")

    node_by_index = {node.index: node for node in graph.nodes}
    fixed_index = graph.nodes[0].index
    fixed_transform = node_by_index[fixed_index].camera_to_world
    variable_indices = [
        node.index for node in graph.nodes if node.index != fixed_index
    ]
    offset_of = {index: position * 6 for position, index in enumerate(variable_indices)}

    initial = np.concatenate(
        [_vec6_from_matrix(node_by_index[index].camera_to_world) for index in variable_indices]
    ) if variable_indices else np.zeros(0)

    edges = graph.edges

    def transforms_from(values: np.ndarray) -> dict[int, np.ndarray]:
        current = {fixed_index: fixed_transform}
        for index in variable_indices:
            offset = offset_of[index]
            current[index] = _matrix_from_vec6(values[offset : offset + 6])
        return current

    def residuals(values: np.ndarray) -> np.ndarray:
        current = transforms_from(values)
        out = np.empty(6 * len(edges), dtype=np.float64)
        for row, edge in enumerate(edges):
            pose_p = current[edge.source_index]
            pose_q = current[edge.target_index]
            predicted = np.linalg.inv(pose_q) @ pose_p
            error = np.linalg.inv(edge.measured_target_from_source) @ predicted
            error_vec = _vec6_from_matrix(error)
            base = 6 * row
            out[base : base + 3] = edge.rotation_weight * error_vec[:3]
            out[base + 3 : base + 6] = edge.translation_weight * error_vec[3:6]
        return out

    sparsity = lil_matrix((6 * len(edges), max(len(initial), 1)), dtype=np.int8)
    for row, edge in enumerate(edges):
        rows = slice(6 * row, 6 * row + 6)
        for endpoint in (edge.source_index, edge.target_index):
            if endpoint in offset_of:
                offset = offset_of[endpoint]
                sparsity[rows, offset : offset + 6] = 1

    if len(initial) == 0:
        # Nothing to optimize (single free node); return the inputs unchanged.
        poses = [
            _trajectory_pose_from_camera_to_world(node.timestamp, node.camera_to_world)
            for node in graph.nodes
        ]
        return PoseGraphOptimizationResult(
            poses=poses,
            iterations=0,
            initial_cost=0.0,
            final_cost=0.0,
            success=True,
            message="no free poses",
        )

    initial_residuals = residuals(initial)
    result = least_squares(
        residuals,
        initial,
        jac_sparsity=sparsity,
        method="trf",
        loss=config.optimizer_loss,
        f_scale=config.optimizer_f_scale,
        x_scale="jac",
        max_nfev=config.optimizer_max_nfev,
        verbose=0,
    )

    optimized = transforms_from(result.x)
    poses = [
        _trajectory_pose_from_camera_to_world(
            node.timestamp, optimized[node.index]
        )
        for node in graph.nodes
    ]
    return PoseGraphOptimizationResult(
        poses=poses,
        iterations=int(result.nfev),
        initial_cost=float(0.5 * np.dot(initial_residuals, initial_residuals)),
        final_cost=float(result.cost),
        success=bool(result.success),
        message=str(result.message),
    )


# --------------------------------------------------------------------------- #
# Priority 1/2/3 diagnostics writers
# --------------------------------------------------------------------------- #
_LOOP_CANDIDATE_FIELDS = [
    "source_index",
    "target_index",
    "source_timestamp",
    "target_timestamp",
    "method",
    "matcher",
    "retrieval_similarity",
    "proposal_rank",
    "matched_features",
    "valid_depth_points",
    "inliers",
    "median_reprojection_error",
    "translation_norm",
    "rotation_degrees",
    "accepted",
    "reject_reason",
]


def write_loop_candidates_csv(
    path: str | Path, candidates: list[LoopEdgeCandidate]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LOOP_CANDIDATE_FIELDS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "source_index": candidate.source_index,
                    "target_index": candidate.target_index,
                    "source_timestamp": _format_float(candidate.source_timestamp),
                    "target_timestamp": _format_float(candidate.target_timestamp),
                    "method": candidate.method,
                    "matcher": candidate.matcher,
                    "retrieval_similarity": (
                        _format_float(candidate.retrieval_similarity)
                        if np.isfinite(candidate.retrieval_similarity)
                        else ""
                    ),
                    "proposal_rank": candidate.proposal_rank,
                    "matched_features": candidate.matched_features,
                    "valid_depth_points": candidate.valid_depth_points,
                    "inliers": candidate.inliers,
                    "median_reprojection_error": _format_float(
                        candidate.median_reprojection_error
                    ),
                    "translation_norm": _format_float(candidate.translation_norm),
                    "rotation_degrees": _format_float(candidate.rotation_degrees),
                    "accepted": int(candidate.accepted),
                    "reject_reason": candidate.reject_reason,
                }
            )


def write_loop_candidate_summary(
    path: str | Path,
    candidates: list[LoopEdgeCandidate],
    accepted: list[LoopEdgeCandidate],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluated_candidates": len(candidates),
        "accepted_candidates": len(accepted),
        "accepted_edges": [
            {
                "source_index": candidate.source_index,
                "target_index": candidate.target_index,
                "method": candidate.method,
                "matcher": candidate.matcher,
                "retrieval_similarity": (
                    candidate.retrieval_similarity
                    if np.isfinite(candidate.retrieval_similarity)
                    else None
                ),
                "proposal_rank": candidate.proposal_rank,
                "inliers": candidate.inliers,
                "median_reprojection_error": candidate.median_reprojection_error,
                "translation_norm": candidate.translation_norm,
                "rotation_degrees": candidate.rotation_degrees,
            }
            for candidate in accepted
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_pose_graph_edges_csv(path: str | Path, graph: PoseGraph) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_index",
                "target_index",
                "edge_type",
                "translation_weight",
                "rotation_weight",
                "inlier_count",
                "translation_norm",
                "rotation_degrees",
            ],
        )
        writer.writeheader()
        for edge in graph.edges:
            measured = edge.measured_target_from_source
            writer.writerow(
                {
                    "source_index": edge.source_index,
                    "target_index": edge.target_index,
                    "edge_type": edge.edge_type,
                    "translation_weight": _format_float(edge.translation_weight),
                    "rotation_weight": _format_float(edge.rotation_weight),
                    "inlier_count": edge.inlier_count,
                    "translation_norm": _format_float(
                        float(np.linalg.norm(measured[:3, 3]))
                    ),
                    "rotation_degrees": _format_float(
                        _rotation_angle_degrees(measured)
                    ),
                }
            )


def write_pose_graph_summary(path: str | Path, graph: PoseGraph) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "odometry_edges": len(graph.odometry_edges),
        "loop_edges": len(graph.loop_edges),
        "loop_edge_pairs": [
            {"source_index": edge.source_index, "target_index": edge.target_index}
            for edge in graph.loop_edges
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_optimization_summary(
    path: str | Path,
    graph: PoseGraph,
    result: PoseGraphOptimizationResult,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "odometry_edges": len(graph.odometry_edges),
        "loop_edges": len(graph.loop_edges),
        "iterations": result.iterations,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "success": result.success,
        "message": result.message,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_comparison_summary(
    path: str | Path,
    metrics_by_variant: dict[str, object],
    accepted_loop_edges: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"accepted_loop_edges": accepted_loop_edges}
    for variant, metrics in metrics_by_variant.items():
        if metrics is None:
            payload[variant] = None
            continue
        payload[variant] = {
            "matched_poses": metrics.matched_poses,
            "translation_rmse": metrics.translation_rmse,
            "translation_mean": metrics.translation_mean,
            "translation_median": metrics.translation_median,
            "translation_max": metrics.translation_max,
        }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def add_pose_graph_arguments(parser: argparse.ArgumentParser) -> None:
    """Register pose-graph CLI flags onto the Stage 3 argument parser."""
    group = parser.add_argument_group("pose graph")
    group.add_argument(
        "--run-pose-graph",
        action="store_true",
        help="Estimate loop edges, build a pose graph, and optimize it.",
    )
    group.add_argument("--loop-method", choices=("pnp", "rgbd"), default="pnp")
    group.add_argument(
        "--loop-candidate-mode", choices=("window", "dinov2"), default="window"
    )
    group.add_argument("--retrieval-stride", type=int, default=16)
    group.add_argument("--retrieval-min-separation", type=int, default=500)
    group.add_argument("--retrieval-max-candidates", type=int, default=50)
    group.add_argument("--retrieval-proposal-spacing", type=int, default=16)
    group.add_argument("--retrieval-refine-radius", type=int, default=4)
    group.add_argument("--retrieval-refine-top-k", type=int, default=10)
    group.add_argument("--retrieval-min-similarity", type=float, default=0.6)
    group.add_argument("--retrieval-batch-size", type=int, default=16)
    group.add_argument(
        "--retrieval-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    group.add_argument(
        "--loop-matcher",
        choices=("classical", "superpoint-lightglue"),
        default="classical",
        help="Correspondence frontend used only for loop-edge estimation.",
    )
    group.add_argument(
        "--learned-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    group.add_argument("--learned-filter-threshold", type=float, default=0.1)
    group.add_argument("--loop-source-window", type=int, default=40)
    group.add_argument("--loop-target-window", type=int, default=40)
    group.add_argument("--loop-source-stride", type=int, default=8)
    group.add_argument("--loop-target-stride", type=int, default=8)
    group.add_argument("--loop-min-separation", type=int, default=50)
    group.add_argument("--max-loop-edges", type=int, default=1)
    group.add_argument("--loop-endpoint-spacing", type=int, default=0)
    group.add_argument("--loop-min-matches", type=int, default=30)
    group.add_argument("--loop-min-depth-points", type=int, default=12)
    group.add_argument("--loop-min-inliers", type=int, default=15)
    group.add_argument("--loop-max-translation", type=float, default=3.0)
    group.add_argument("--loop-max-rotation-degrees", type=float, default=60.0)
    group.add_argument("--loop-max-reprojection-error", type=float, default=4.0)
    group.add_argument("--odometry-translation-weight", type=float, default=1.0)
    group.add_argument("--odometry-rotation-weight", type=float, default=1.0)
    group.add_argument("--loop-translation-weight", type=float, default=0.5)
    group.add_argument("--loop-rotation-weight", type=float, default=0.5)
    group.add_argument("--pose-graph-keyframe-stride", type=int, default=1)
    group.add_argument("--optimizer-loss", default="soft_l1")
    group.add_argument("--optimizer-f-scale", type=float, default=1.0)
    group.add_argument("--optimizer-max-nfev", type=int, default=200)


@dataclass(frozen=True)
class PoseGraphPipelineResult:
    candidates: list[LoopEdgeCandidate]
    accepted: list[LoopEdgeCandidate]
    graph: PoseGraph
    optimization: PoseGraphOptimizationResult
    trajectory_path: Path
    pose_graph_metrics: Stage3TrajectoryMetrics | None


def run_pose_graph_pipeline(
    dataset: Stage3Dataset,
    odometry_config: Stage3OdometryConfig,
    config: PoseGraphConfig,
    output_dir: str | Path,
    *,
    vo_result: Stage3OdometryResult | None = None,
    ground_truth: list[TrajectoryPose] | None = None,
    raw_metrics: Stage3TrajectoryMetrics | None = None,
    loop_closed_metrics: Stage3TrajectoryMetrics | None = None,
) -> PoseGraphPipelineResult:
    """End-to-end Priority 1-4: loop edges, graph, optimization, export."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if vo_result is None:
        vo_result = estimate_rgbd_trajectory(dataset, odometry_config)

    candidates = estimate_loop_edges(dataset, odometry_config, config)
    accepted = select_loop_edges(candidates, config)
    write_loop_candidates_csv(output_dir / "loop_candidates.csv", candidates)
    write_loop_candidate_summary(
        output_dir / "loop_candidate_summary.json", candidates, accepted
    )

    if config.keyframe_stride > 1:
        graph = build_keyframe_pose_graph(vo_result.poses, accepted, config)
    else:
        graph = build_pose_graph(vo_result.poses, vo_result.reports, accepted, config)
    write_pose_graph_edges_csv(output_dir / "pose_graph_edges.csv", graph)
    write_pose_graph_summary(output_dir / "pose_graph_summary.json", graph)

    optimization = optimize_pose_graph(graph, config)
    if config.keyframe_stride > 1:
        optimization = PoseGraphOptimizationResult(
            poses=expand_keyframe_optimization(vo_result.poses, optimization.poses),
            iterations=optimization.iterations,
            initial_cost=optimization.initial_cost,
            final_cost=optimization.final_cost,
            success=optimization.success,
            message=optimization.message,
        )
    trajectory_path = output_dir / "estimated_camera_trajectory_pose_graph.txt"
    write_trajectory(trajectory_path, optimization.poses)
    write_optimization_summary(
        output_dir / "pose_graph_optimization_summary.json", graph, optimization
    )

    pose_graph_metrics: Stage3TrajectoryMetrics | None = None
    if ground_truth is not None:
        pose_graph_metrics = evaluate_translation_trajectory(
            ground_truth, optimization.poses
        )
        write_trajectory_metrics(
            output_dir / "trajectory_metrics_pose_graph.json", pose_graph_metrics
        )
        write_comparison_summary(
            output_dir / "stage3_comparison_summary.json",
            {
                "raw_vo": raw_metrics,
                "loop_closed": loop_closed_metrics,
                "pose_graph": pose_graph_metrics,
            },
            accepted_loop_edges=len(accepted),
        )

    return PoseGraphPipelineResult(
        candidates=candidates,
        accepted=accepted,
        graph=graph,
        optimization=optimization,
        trajectory_path=trajectory_path,
        pose_graph_metrics=pose_graph_metrics,
    )


def pose_graph_config_from_args(args: argparse.Namespace) -> PoseGraphConfig:
    return PoseGraphConfig(
        loop_method=args.loop_method,
        loop_candidate_mode=args.loop_candidate_mode,
        retrieval_stride=args.retrieval_stride,
        retrieval_min_separation=args.retrieval_min_separation,
        retrieval_max_candidates=args.retrieval_max_candidates,
        retrieval_proposal_spacing=args.retrieval_proposal_spacing,
        retrieval_refine_radius=args.retrieval_refine_radius,
        retrieval_refine_top_k=args.retrieval_refine_top_k,
        retrieval_min_similarity=args.retrieval_min_similarity,
        retrieval_batch_size=args.retrieval_batch_size,
        retrieval_device=args.retrieval_device,
        loop_matcher=args.loop_matcher,
        learned_device=args.learned_device,
        learned_filter_threshold=args.learned_filter_threshold,
        loop_source_window=args.loop_source_window,
        loop_target_window=args.loop_target_window,
        loop_source_stride=args.loop_source_stride,
        loop_target_stride=args.loop_target_stride,
        loop_min_separation=args.loop_min_separation,
        max_loop_edges=args.max_loop_edges,
        loop_endpoint_spacing=args.loop_endpoint_spacing,
        loop_min_matches=args.loop_min_matches,
        loop_min_depth_points=args.loop_min_depth_points,
        loop_min_inliers=args.loop_min_inliers,
        loop_max_translation=args.loop_max_translation,
        loop_max_rotation_degrees=args.loop_max_rotation_degrees,
        loop_max_reprojection_error=args.loop_max_reprojection_error,
        odometry_translation_weight=args.odometry_translation_weight,
        odometry_rotation_weight=args.odometry_rotation_weight,
        loop_translation_weight=args.loop_translation_weight,
        loop_rotation_weight=args.loop_rotation_weight,
        keyframe_stride=args.pose_graph_keyframe_stride,
        optimizer_loss=args.optimizer_loss,
        optimizer_f_scale=args.optimizer_f_scale,
        optimizer_max_nfev=args.optimizer_max_nfev,
    )
