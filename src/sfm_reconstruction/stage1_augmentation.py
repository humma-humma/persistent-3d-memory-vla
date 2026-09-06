from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .dataset import load_stage2_dataset
from .dense_fusion import load_camera_parameters
from .geometry import (
    points_in_front,
    project_points,
    triangulate_points,
    triangulation_angles,
)
from .models import Pose, Track
from .stage1_evaluation import evaluate_stage1_points
from .tracks import build_tracks


@dataclass(frozen=True)
class Stage1AugmentationConfig:
    min_observations: int = 3
    median_reprojection_error: float = 1.0
    max_reprojection_error: float = 2.5
    min_triangulation_angle: float = 2.0
    max_pair_dispersion: float = 0.05
    duplicate_distance: float = 0.01
    bounds_percentile: float = 1.0
    bounds_margin_factor: float = 0.15
    point_optimizer_max_nfev: int = 30


@dataclass(frozen=True)
class TrackPointResult:
    point: np.ndarray | None
    observations: int
    median_error: float
    max_error: float
    max_angle: float
    pair_dispersion: float
    reject_reason: str


def _linear_triangulation(
    track: Track, poses: dict[int, Pose], intrinsics: np.ndarray
) -> np.ndarray:
    rows = []
    for image_id, observed in track.observations.items():
        projection = intrinsics @ np.hstack(
            (poses[image_id].rotation, poses[image_id].translation)
        )
        rows.append(observed[0] * projection[2] - projection[0])
        rows.append(observed[1] * projection[2] - projection[1])
    _, _, right = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = right[-1]
    if abs(homogeneous[3]) <= 1e-12:
        return np.full(3, np.nan)
    return homogeneous[:3] / homogeneous[3]


def triangulate_learned_track(
    track: Track,
    poses: dict[int, Pose],
    intrinsics: np.ndarray,
    config: Stage1AugmentationConfig,
) -> TrackPointResult:
    visible = sorted(set(track.observations) & set(poses))
    empty = TrackPointResult(None, len(visible), np.inf, np.inf, 0.0, np.inf, "")
    if len(visible) < config.min_observations:
        return TrackPointResult(**{**asdict(empty), "reject_reason": "too few observations"})
    visible_track = Track(
        observations={image_id: track.observations[image_id] for image_id in visible}
    )
    initial = _linear_triangulation(visible_track, poses, intrinsics)
    if not np.isfinite(initial).all():
        return TrackPointResult(**{**asdict(empty), "reject_reason": "non-finite triangulation"})

    def residuals(point: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                project_points(point.reshape(1, 3), poses[image_id], intrinsics)[0]
                - visible_track.observations[image_id]
                for image_id in visible
            ]
        )

    optimized = least_squares(
        residuals,
        initial,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=config.point_optimizer_max_nfev,
    ).x
    if not all(
        points_in_front(optimized.reshape(1, 3), poses[image_id])[0]
        for image_id in visible
    ):
        return TrackPointResult(**{**asdict(empty), "reject_reason": "failed cheirality"})

    errors = np.asarray(
        [
            np.linalg.norm(
                project_points(
                    optimized.reshape(1, 3), poses[image_id], intrinsics
                )[0]
                - visible_track.observations[image_id]
            )
            for image_id in visible
        ]
    )
    angles = [
        float(
            triangulation_angles(
                optimized.reshape(1, 3), poses[first_id], poses[second_id]
            )[0]
        )
        for index, first_id in enumerate(visible)
        for second_id in visible[index + 1 :]
    ]
    max_angle = max(angles, default=0.0)
    pair_points = []
    for index, first_id in enumerate(visible):
        for second_id in visible[index + 1 :]:
            angle = triangulation_angles(
                optimized.reshape(1, 3), poses[first_id], poses[second_id]
            )[0]
            if angle < config.min_triangulation_angle:
                continue
            point = triangulate_points(
                visible_track.observations[first_id].reshape(1, 2),
                visible_track.observations[second_id].reshape(1, 2),
                poses[first_id],
                poses[second_id],
                intrinsics,
            )[0]
            if np.isfinite(point).all():
                pair_points.append(point)
    dispersion = (
        float(np.median(np.linalg.norm(np.asarray(pair_points) - optimized, axis=1)))
        if len(pair_points) >= 2
        else np.inf
    )
    result = TrackPointResult(
        point=optimized,
        observations=len(visible),
        median_error=float(np.median(errors)),
        max_error=float(np.max(errors)),
        max_angle=max_angle,
        pair_dispersion=dispersion,
        reject_reason="",
    )
    if result.median_error > config.median_reprojection_error:
        return TrackPointResult(**{**asdict(result), "point": None, "reject_reason": "high median reprojection"})
    if result.max_error > config.max_reprojection_error:
        return TrackPointResult(**{**asdict(result), "point": None, "reject_reason": "high maximum reprojection"})
    if result.max_angle < config.min_triangulation_angle:
        return TrackPointResult(**{**asdict(result), "point": None, "reject_reason": "low triangulation angle"})
    if result.pair_dispersion > config.max_pair_dispersion:
        return TrackPointResult(**{**asdict(result), "point": None, "reject_reason": "inconsistent pair triangulations"})
    return result


def _load_vertices(path: Path) -> np.ndarray:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("Stage 1 augmentation requires the evaluation extra") from error
    geometry = trimesh.load(path, process=False)
    return np.asarray(geometry.vertices, dtype=np.float64)


def _write_points(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for point in points:
            handle.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")


def augment_stage1_points(
    dataset_root: Path,
    baseline_result: Path,
    learned_correspondence_dir: Path,
    output_dir: Path,
    config: Stage1AugmentationConfig | None = None,
) -> dict[str, object]:
    config = config or Stage1AugmentationConfig()
    dataset = load_stage2_dataset(dataset_root, learned_correspondence_dir)
    intrinsics, named_poses = load_camera_parameters(
        baseline_result / "estimated_camera_parameters.json"
    )
    poses = {int(Path(name).stem): pose for name, pose in named_poses.items()}
    baseline_points = _load_vertices(baseline_result / "estimated_points.ply")
    track_result = build_tracks(dataset, min_observations=config.min_observations)

    low = np.percentile(baseline_points, config.bounds_percentile, axis=0)
    high = np.percentile(baseline_points, 100.0 - config.bounds_percentile, axis=0)
    margin = config.bounds_margin_factor * (high - low)
    low -= margin
    high += margin
    baseline_tree = cKDTree(baseline_points)
    accepted = []
    diagnostics = []
    for track_id, track in enumerate(track_result.tracks):
        result = triangulate_learned_track(track, poses, intrinsics, config)
        reason = result.reject_reason
        nearest = np.inf
        if result.point is not None:
            if np.any(result.point < low) or np.any(result.point > high):
                reason = "outside baseline bounds"
            else:
                nearest = float(baseline_tree.query(result.point)[0])
                if nearest <= config.duplicate_distance:
                    reason = "duplicate baseline point"
                else:
                    accepted.append(result.point)
        diagnostics.append(
            {
                "track_id": track_id,
                "observations": result.observations,
                "median_error": result.median_error,
                "max_error": result.max_error,
                "max_angle": result.max_angle,
                "pair_dispersion": result.pair_dispersion,
                "nearest_baseline_point": nearest,
                "accepted": int(not reason),
                "reject_reason": reason,
            }
        )

    learned_points = np.asarray(accepted, dtype=np.float64).reshape(-1, 3)
    merged = np.vstack((baseline_points, learned_points))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_points(output_dir / "learned_points.ply", learned_points)
    _write_points(output_dir / "estimated_points.ply", merged)
    shutil.copy2(
        baseline_result / "estimated_camera_parameters.json",
        output_dir / "estimated_camera_parameters.json",
    )
    with (output_dir / "learned_point_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    payload: dict[str, object] = {
        "dataset": str(dataset_root),
        "baseline_result": str(baseline_result),
        "learned_correspondence_dir": str(learned_correspondence_dir),
        "config": asdict(config),
        "baseline_points": len(baseline_points),
        "learned_tracks": len(track_result.tracks),
        "learned_track_conflicts": track_result.skipped_conflicts,
        "accepted_learned_points": len(learned_points),
        "merged_points": len(merged),
    }
    ground_truth_path = dataset_root / "gt_points.ply"
    if ground_truth_path.is_file():
        ground_truth = _load_vertices(ground_truth_path)
        payload["baseline_metrics"] = asdict(
            evaluate_stage1_points(ground_truth, baseline_points)
        )
        payload["augmented_metrics"] = asdict(
            evaluate_stage1_points(ground_truth, merged)
        )
    (output_dir / "stage1_augmentation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add validated learned points while freezing Stage 1 cameras."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline-result", required=True, type=Path)
    parser.add_argument("--learned-correspondence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = augment_stage1_points(
        args.dataset,
        args.baseline_result,
        args.learned_correspondence_dir,
        args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
