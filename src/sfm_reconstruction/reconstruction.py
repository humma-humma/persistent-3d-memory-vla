from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .bundle_adjustment import bundle_adjust
from .dataset import Stage1Dataset
from .geometry import (
    estimate_relative_pose,
    points_in_front,
    reprojection_errors,
    solve_pnp,
    triangulate_points,
    triangulation_angles,
)
from .models import Pose, Track
from .tracks import TrackBuildResult, build_tracks, observation_key


@dataclass(frozen=True)
class ReconstructionConfig:
    track_tolerance: float = 1e-3
    min_track_observations: int = 2
    essential_threshold: float = 1.0
    pnp_reprojection_error: float = 4.0
    max_reprojection_error: float = 5.0
    min_triangulation_angle: float = 1.0
    min_pnp_points: int = 12
    min_pnp_inliers: int = 10
    bundle_adjustment: bool = True
    bundle_adjustment_max_nfev: int = 40
    max_point_distance_factor: float | None = None
    two_view_max_reprojection_error: float | None = None
    two_view_min_triangulation_angle: float | None = None
    initial_pose_correspondence_dir: Path | None = None
    initial_pair: tuple[int, int] | None = None


def _filter_points(
    intrinsics: np.ndarray,
    tracks: list[Track],
    poses: dict[int, Pose],
    points: dict[int, np.ndarray],
    config: ReconstructionConfig,
) -> dict[int, np.ndarray]:
    distance_center = None
    maximum_distance = None
    if (
        config.two_view_max_reprojection_error is not None
        and config.two_view_max_reprojection_error <= 0.0
    ):
        raise ValueError("two_view_max_reprojection_error must be positive")
    if (
        config.two_view_min_triangulation_angle is not None
        and config.two_view_min_triangulation_angle <= 0.0
    ):
        raise ValueError("two_view_min_triangulation_angle must be positive")
    if config.max_point_distance_factor is not None:
        if config.max_point_distance_factor <= 0.0:
            raise ValueError("max_point_distance_factor must be positive")
        camera_centers = np.asarray(
            [pose.camera_center for pose in poses.values()],
            dtype=np.float64,
        )
        distance_center = np.median(camera_centers, axis=0)
        camera_radius = float(
            np.max(np.linalg.norm(camera_centers - distance_center, axis=1))
        )
        maximum_distance = config.max_point_distance_factor * camera_radius

    filtered = {}
    for track_id, point in points.items():
        if (
            distance_center is not None
            and maximum_distance is not None
            and np.linalg.norm(point - distance_center) > maximum_distance
        ):
            continue
        visible = sorted(set(tracks[track_id].observations) & set(poses))
        if len(visible) < config.min_track_observations:
            continue

        valid_cameras = []
        valid_errors = []
        for image_id in visible:
            pose = poses[image_id]
            camera_point = pose.rotation @ point.reshape(3, 1) + pose.translation
            if camera_point[2, 0] <= 0.0:
                continue
            error = reprojection_errors(
                point.reshape(1, 3),
                tracks[track_id].observations[image_id].reshape(1, 2),
                pose,
                intrinsics,
            )[0]
            if error <= config.max_reprojection_error:
                valid_cameras.append(image_id)
                valid_errors.append(float(error))
        if len(valid_cameras) < config.min_track_observations:
            continue

        maximum_angle = max(
            (
                triangulation_angles(
                    point.reshape(1, 3),
                    poses[first_id],
                    poses[second_id],
                )[0]
                for index, first_id in enumerate(valid_cameras)
                for second_id in valid_cameras[index + 1 :]
            ),
            default=0.0,
        )
        if len(valid_cameras) == 2:
            if (
                config.two_view_max_reprojection_error is not None
                and max(valid_errors) > config.two_view_max_reprojection_error
            ):
                continue
            if (
                config.two_view_min_triangulation_angle is not None
                and maximum_angle < config.two_view_min_triangulation_angle
            ):
                continue
        if maximum_angle >= config.min_triangulation_angle:
            filtered[track_id] = point
    return filtered


@dataclass(frozen=True)
class ReconstructionResult:
    poses: dict[int, Pose]
    points: dict[int, np.ndarray]
    tracks: list[Track]
    initial_pair: tuple[int, int]
    skipped_track_conflicts: int


SnapshotCallback = Callable[[tuple[int, ...], ReconstructionResult], None]


def _snapshot_result(
    dataset: Stage1Dataset,
    track_result: TrackBuildResult,
    poses: dict[int, Pose],
    points: dict[int, np.ndarray],
    initial_pair: tuple[int, int],
    config: ReconstructionConfig,
) -> ReconstructionResult:
    """Copy the currently registered map without future-frame observations."""
    visible_tracks = [
        Track(
            observations={
                image_id: observation.copy()
                for image_id, observation in track.observations.items()
                if image_id in poses
            }
        )
        for track in track_result.tracks
    ]
    visible_points = _filter_points(
        dataset.intrinsics,
        visible_tracks,
        poses,
        points,
        config,
    )
    return ReconstructionResult(
        poses=dict(poses),
        points={track_id: point.copy() for track_id, point in visible_points.items()},
        tracks=visible_tracks,
        initial_pair=initial_pair,
        skipped_track_conflicts=track_result.skipped_conflicts,
    )


def _track_ids_for_matches(
    pair: tuple[int, int],
    matches: np.ndarray,
    track_result: TrackBuildResult,
    tolerance: float,
) -> np.ndarray:
    result = np.full(len(matches), -1, dtype=np.int64)
    first_id, second_id = pair
    for index, match in enumerate(matches):
        first_track = track_result.observation_to_track.get(
            observation_key(first_id, match[:2], tolerance)
        )
        second_track = track_result.observation_to_track.get(
            observation_key(second_id, match[2:], tolerance)
        )
        if first_track is not None and first_track == second_track:
            result[index] = first_track
    return result


def _valid_triangulations(
    points_3d: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_pose: Pose,
    second_pose: Pose,
    intrinsics: np.ndarray,
    config: ReconstructionConfig,
) -> np.ndarray:
    finite = np.isfinite(points_3d).all(axis=1)
    cheirality = points_in_front(points_3d, first_pose) & points_in_front(
        points_3d, second_pose
    )
    angles = triangulation_angles(points_3d, first_pose, second_pose)
    first_error = reprojection_errors(
        points_3d, first_points, first_pose, intrinsics
    )
    second_error = reprojection_errors(
        points_3d, second_points, second_pose, intrinsics
    )
    return (
        finite
        & cheirality
        & (angles >= config.min_triangulation_angle)
        & (first_error <= config.max_reprojection_error)
        & (second_error <= config.max_reprojection_error)
    )


def _initialize(
    dataset: Stage1Dataset,
    track_result: TrackBuildResult,
    config: ReconstructionConfig,
) -> tuple[tuple[int, int], dict[int, Pose], dict[int, np.ndarray]]:
    image_ids = dataset.image_ids
    anchor_id = image_ids[0]
    if config.initial_pair is not None:
        if config.initial_pair[0] != anchor_id:
            raise ValueError("initial_pair must start with the anchor image")
        if config.initial_pair not in dataset.correspondence_paths:
            raise ValueError(
                f"initial_pair {config.initial_pair} has no correspondence file"
            )
    forward_half = set(image_ids[1 : len(image_ids) // 2 + 1])
    candidate_pairs = sorted(
        (
            (len(dataset.load_correspondences(pair)), pair)
            for pair in dataset.correspondence_paths
            if (
                pair == config.initial_pair
                if config.initial_pair is not None
                else anchor_id in pair
                and next(image_id for image_id in pair if image_id != anchor_id)
                in forward_half
            )
        ),
        reverse=True,
    )
    if not candidate_pairs:
        raise RuntimeError(f"No correspondence pair includes anchor image {anchor_id}")

    best: tuple[
        float,
        tuple[int, int],
        dict[int, Pose],
        dict[int, np.ndarray],
    ] | None = None
    for _, pair in candidate_pairs:
        matches = dataset.load_correspondences(pair)
        first_points = matches[:, :2]
        second_points = matches[:, 2:]
        pose_matches = matches
        if config.initial_pose_correspondence_dir is not None:
            pose_path = config.initial_pose_correspondence_dir / (
                f"{pair[0]}_{pair[1]}.txt"
            )
            if pose_path.is_file():
                pose_matches = np.loadtxt(pose_path, dtype=np.float64, ndmin=2)
        try:
            relative = estimate_relative_pose(
                pose_matches[:, :2],
                pose_matches[:, 2:],
                dataset.intrinsics,
                config.essential_threshold,
            )
        except (RuntimeError, ValueError):
            continue

        first_pose = Pose.identity()
        second_pose = relative.pose
        points_3d = triangulate_points(
            first_points,
            second_points,
            first_pose,
            second_pose,
            dataset.intrinsics,
        )
        valid = _valid_triangulations(
            points_3d,
            first_points,
            second_points,
            first_pose,
            second_pose,
            dataset.intrinsics,
            config,
        )
        if pose_matches is matches:
            valid &= relative.inliers
        track_ids = _track_ids_for_matches(
            pair, matches, track_result, config.track_tolerance
        )
        valid &= track_ids >= 0
        initialized_points = {
            int(track_id): point
            for track_id, point, keep in zip(track_ids, points_3d, valid)
            if keep
        }
        if len(initialized_points) < config.min_pnp_points:
            continue
        median_angle = float(np.median(
            triangulation_angles(
                points_3d[valid],
                first_pose,
                second_pose,
            )
        ))
        initialization_score = len(initialized_points) * min(median_angle, 15.0)
        poses = {pair[0]: first_pose, pair[1]: second_pose}
        candidate = (initialization_score, pair, poses, initialized_points)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("No valid initialization pair could be reconstructed")
    return best[1], best[2], best[3]


def _pnp_correspondences(
    image_id: int,
    tracks: list[Track],
    points: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    points_3d: list[np.ndarray] = []
    points_2d: list[np.ndarray] = []
    track_ids: list[int] = []
    for track_id, point_3d in points.items():
        observation = tracks[track_id].observations.get(image_id)
        if observation is None:
            continue
        points_3d.append(point_3d)
        points_2d.append(observation)
        track_ids.append(track_id)
    return np.asarray(points_3d), np.asarray(points_2d), track_ids


def _triangulate_new_tracks(
    intrinsics: np.ndarray,
    tracks: list[Track],
    poses: dict[int, Pose],
    points: dict[int, np.ndarray],
    config: ReconstructionConfig,
) -> int:
    added = 0
    registered = set(poses)
    for track_id, track in enumerate(tracks):
        if track_id in points:
            continue
        visible = sorted(registered & set(track.observations))
        if len(visible) < 2:
            continue
        pair = max(
            (
                (np.linalg.norm(poses[first].camera_center - poses[second].camera_center), first, second)
                for index, first in enumerate(visible)
                for second in visible[index + 1 :]
            ),
            default=None,
        )
        if pair is None or pair[0] <= 1e-8:
            continue
        first_id, second_id = pair[1], pair[2]
        first_point = track.observations[first_id].reshape(1, 2)
        second_point = track.observations[second_id].reshape(1, 2)
        point_3d = triangulate_points(
            first_point,
            second_point,
            poses[first_id],
            poses[second_id],
            intrinsics,
        )
        valid = _valid_triangulations(
            point_3d,
            first_point,
            second_point,
            poses[first_id],
            poses[second_id],
            intrinsics,
            config,
        )[0]
        if not valid:
            continue

        errors = [
            reprojection_errors(
                point_3d,
                track.observations[image_id].reshape(1, 2),
                poses[image_id],
                intrinsics,
            )[0]
            for image_id in visible
        ]
        if np.median(errors) > config.max_reprojection_error:
            continue
        points[track_id] = point_3d[0]
        added += 1
    return added


def reconstruct(
    dataset: Stage1Dataset,
    config: ReconstructionConfig | None = None,
    snapshot_callback: SnapshotCallback | None = None,
) -> ReconstructionResult:
    config = config or ReconstructionConfig()
    track_result = build_tracks(
        dataset,
        config.track_tolerance,
        config.min_track_observations,
    )
    initial_pair, poses, points = _initialize(dataset, track_result, config)
    _triangulate_new_tracks(
        dataset.intrinsics, track_result.tracks, poses, points, config
    )
    if snapshot_callback is not None:
        snapshot_callback(
            initial_pair,
            _snapshot_result(
                dataset, track_result, poses, points, initial_pair, config
            ),
        )

    unregistered = set(dataset.image_ids) - set(poses)
    while unregistered:
        candidates: list[tuple[int, int, Pose]] = []
        for image_id in sorted(unregistered):
            points_3d, points_2d, _ = _pnp_correspondences(
                image_id, track_result.tracks, points
            )
            if len(points_3d) < config.min_pnp_points:
                continue
            try:
                estimate = solve_pnp(
                    points_3d,
                    points_2d,
                    dataset.intrinsics,
                    config.pnp_reprojection_error,
                )
            except (RuntimeError, ValueError):
                continue
            inlier_count = int(estimate.inliers.sum())
            if inlier_count >= config.min_pnp_inliers:
                candidates.append((inlier_count, image_id, estimate.pose))

        if not candidates:
            break
        _, image_id, pose = max(candidates, key=lambda item: (item[0], -item[1]))
        poses[image_id] = pose
        unregistered.remove(image_id)
        _triangulate_new_tracks(
            dataset.intrinsics, track_result.tracks, poses, points, config
        )
        if snapshot_callback is not None:
            snapshot_callback(
                (image_id,),
                _snapshot_result(
                    dataset, track_result, poses, points, initial_pair, config
                ),
            )

    points = _filter_points(
        dataset.intrinsics,
        track_result.tracks,
        poses,
        points,
        config,
    )
    if config.bundle_adjustment and len(poses) >= 3 and points:
        poses, points = bundle_adjust(
            dataset.intrinsics,
            track_result.tracks,
            poses,
            points,
            fixed_camera_ids=set(initial_pair),
            max_nfev=config.bundle_adjustment_max_nfev,
            max_reprojection_error=config.max_reprojection_error,
        )
    points = _filter_points(
        dataset.intrinsics,
        track_result.tracks,
        poses,
        points,
        config,
    )

    return ReconstructionResult(
        poses=poses,
        points=points,
        tracks=track_result.tracks,
        initial_pair=initial_pair,
        skipped_track_conflicts=track_result.skipped_conflicts,
    )


def retriangulate_with_poses(
    dataset: Stage1Dataset,
    result: ReconstructionResult,
    poses: dict[int, Pose],
    config: ReconstructionConfig | None = None,
) -> ReconstructionResult:
    """Rebuild points on unchanged tracks using externally refined poses."""
    config = config or ReconstructionConfig()
    points: dict[int, np.ndarray] = {}
    _triangulate_new_tracks(
        dataset.intrinsics,
        result.tracks,
        poses,
        points,
        config,
    )
    points = _filter_points(
        dataset.intrinsics,
        result.tracks,
        poses,
        points,
        config,
    )
    if config.bundle_adjustment and len(poses) >= 3 and points:
        poses, points = bundle_adjust(
            dataset.intrinsics,
            result.tracks,
            poses,
            points,
            fixed_camera_ids=set(result.initial_pair),
            max_nfev=config.bundle_adjustment_max_nfev,
            max_reprojection_error=config.max_reprojection_error,
        )
        points = _filter_points(
            dataset.intrinsics,
            result.tracks,
            poses,
            points,
            config,
        )
    return ReconstructionResult(
        poses=poses,
        points=points,
        tracks=result.tracks,
        initial_pair=result.initial_pair,
        skipped_track_conflicts=result.skipped_track_conflicts,
    )
