from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import solve_pnp
from .matching import match_descriptors, root_sift
from .models import Pose


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
TRAJECTORY_COLUMNS = 8


@dataclass(frozen=True)
class Stage3Frame:
    timestamp: float
    rgb_path: Path
    depth_path: Path | None


@dataclass(frozen=True)
class TrajectoryPose:
    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64).reshape(4)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)

    def row(self) -> list[float]:
        return [
            self.timestamp,
            *self.translation.tolist(),
            *self.quaternion_xyzw.tolist(),
        ]


@dataclass(frozen=True)
class Stage3Dataset:
    root: Path
    intrinsics: np.ndarray
    frames: list[Stage3Frame]
    ground_truth_trajectory_path: Path | None

    def subset(self, max_frames: int | None) -> "Stage3Dataset":
        if max_frames is None or max_frames >= len(self.frames):
            return self
        if max_frames < 1:
            raise ValueError("max_frames must be at least 1")
        return replace(self, frames=self.frames[:max_frames])

    @property
    def timestamps(self) -> list[float]:
        return [frame.timestamp for frame in self.frames]


@dataclass(frozen=True)
class Stage3OdometryConfig:
    motion_model: str = "pnp"
    feature_mode: str = "akaze"
    max_features: int = 2500
    image_scale: float = 1.0
    ratio_threshold: float = 0.8
    depth_scale: float = 1000.0
    min_depth: float = 0.1
    max_depth: float = 8.0
    min_matches: int = 30
    min_pnp_points: int = 12
    min_pnp_inliers: int = 10
    pnp_reprojection_error: float = 2.0
    ransac_iterations: int = 200
    ransac_3d_threshold: float = 0.05
    lk_quality_level: float = 0.01
    lk_min_distance: float = 7.0
    max_translation_step: float | None = 0.2
    max_rotation_step_degrees: float | None = 20.0


@dataclass(frozen=True)
class Stage3FrameFeatures:
    points: np.ndarray
    descriptors: np.ndarray
    descriptor_type: str


@dataclass(frozen=True)
class Stage3OdometryReport:
    timestamp: float
    reference_timestamp: float | None
    matched_features: int
    valid_depth_points: int
    pnp_inliers: int
    status: str


@dataclass(frozen=True)
class Stage3OdometryResult:
    poses: list[TrajectoryPose]
    reports: list[Stage3OdometryReport]


@dataclass(frozen=True)
class Stage3TrajectoryMetrics:
    matched_poses: int
    translation_rmse: float
    translation_mean: float
    translation_median: float
    translation_max: float


@dataclass(frozen=True)
class Stage3LoopClosureSummary:
    poses: int
    start_timestamp: float
    end_timestamp: float
    original_end_translation_error: float
    corrected_end_translation_error: float
    target: str


def load_stage3_dataset(
    root: str | Path,
    *,
    allow_missing_depth: bool = False,
) -> Stage3Dataset:
    root = Path(root).resolve()
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")

    intrinsics = _load_intrinsics(root / "camera_parameters.json")
    rgb_by_key = _timestamped_files(rgb_dir)
    depth_by_key = _timestamped_files(depth_dir)
    if not rgb_by_key:
        raise ValueError(f"No RGB images found under {rgb_dir}")
    if not depth_by_key:
        raise ValueError(f"No depth maps found under {depth_dir}")

    frames: list[Stage3Frame] = []
    missing_depth: list[str] = []
    for key, rgb_path in sorted(rgb_by_key.items(), key=lambda item: item[0]):
        depth_path = depth_by_key.get(key)
        if depth_path is None:
            missing_depth.append(rgb_path.stem)
        frames.append(Stage3Frame(timestamp=key, rgb_path=rgb_path, depth_path=depth_path))

    if missing_depth and not allow_missing_depth:
        preview = ", ".join(missing_depth[:5])
        if len(missing_depth) > 5:
            preview += ", ..."
        raise ValueError(
            f"Missing depth maps for {len(missing_depth)} RGB frames: {preview}"
        )

    gt_path = root / "gt_camera_trajectory.txt"
    return Stage3Dataset(
        root=root,
        intrinsics=intrinsics,
        frames=frames,
        ground_truth_trajectory_path=gt_path if gt_path.is_file() else None,
    )


def load_trajectory(path: str | Path) -> list[TrajectoryPose]:
    trajectory_path = Path(path)
    poses: list[TrajectoryPose] = []
    for line_number, raw_line in enumerate(
        trajectory_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != TRAJECTORY_COLUMNS:
            raise ValueError(
                f"{trajectory_path}:{line_number} must have "
                f"{TRAJECTORY_COLUMNS} columns, got {len(values)}"
            )
        numbers = [float(value) for value in values]
        poses.append(
            TrajectoryPose(
                timestamp=numbers[0],
                translation=np.asarray(numbers[1:4], dtype=np.float64),
                quaternion_xyzw=np.asarray(numbers[4:8], dtype=np.float64),
            )
        )
    if not poses:
        raise ValueError(f"No trajectory poses found in {trajectory_path}")
    return poses


def write_trajectory(path: str | Path, poses: list[TrajectoryPose]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pose in poses:
        lines.append(" ".join(_format_float(value) for value in pose.row()))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def identity_trajectory_template(dataset: Stage3Dataset) -> list[TrajectoryPose]:
    return [
        TrajectoryPose(
            timestamp=timestamp,
            translation=np.zeros(3, dtype=np.float64),
            quaternion_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
        for timestamp in dataset.timestamps
    ]


def estimate_rgbd_trajectory(
    dataset: Stage3Dataset,
    config: Stage3OdometryConfig | None = None,
) -> Stage3OdometryResult:
    if config is None:
        config = Stage3OdometryConfig()
    if not dataset.frames:
        raise ValueError("Stage 3 dataset contains no frames")
    if config.depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if config.image_scale <= 0:
        raise ValueError("image_scale must be positive")
    if config.motion_model not in {"lk_pnp", "pnp", "rgbd"}:
        raise ValueError("motion_model must be one of: lk_pnp, pnp, rgbd")
    if config.ransac_iterations < 1:
        raise ValueError("ransac_iterations must be positive")
    if config.ransac_3d_threshold <= 0:
        raise ValueError("ransac_3d_threshold must be positive")
    if not 0.0 < config.lk_quality_level < 1.0:
        raise ValueError("lk_quality_level must be between 0 and 1")
    if config.lk_min_distance <= 0:
        raise ValueError("lk_min_distance must be positive")
    if config.min_depth <= 0 or config.max_depth <= config.min_depth:
        raise ValueError("Expected 0 < min_depth < max_depth")

    first_pose = _trajectory_pose_from_camera_to_world(
        dataset.frames[0].timestamp,
        np.eye(4, dtype=np.float64),
    )
    poses = [first_pose]
    reports = [
        Stage3OdometryReport(
            timestamp=dataset.frames[0].timestamp,
            reference_timestamp=None,
            matched_features=0,
            valid_depth_points=0,
            pnp_inliers=0,
            status="origin",
        )
    ]

    previous_frame = dataset.frames[0]
    previous_features = _extract_stage3_features(previous_frame.rgb_path, config)
    previous_camera_to_world = np.eye(4, dtype=np.float64)

    for frame in dataset.frames[1:]:
        current_features = _extract_stage3_features(frame.rgb_path, config)
        current_camera_to_world = previous_camera_to_world.copy()
        matched_features = 0
        valid_depth_points = 0
        pnp_inliers = 0
        report = Stage3OdometryReport(
            timestamp=frame.timestamp,
            reference_timestamp=previous_frame.timestamp,
            matched_features=matched_features,
            valid_depth_points=valid_depth_points,
            pnp_inliers=pnp_inliers,
            status="failed",
        )
        try:
            if previous_frame.depth_path is None:
                raise RuntimeError("reference frame has no depth map")
            if config.motion_model == "lk_pnp":
                previous_points, current_points = _track_lk_correspondences(
                    previous_frame.rgb_path,
                    frame.rgb_path,
                    config,
                )
                matched_features = len(previous_points)
            else:
                matches = match_descriptors(
                    previous_features.descriptors,
                    current_features.descriptors,
                    config.ratio_threshold,
                    previous_features.descriptor_type,
                )
                matched_features = len(matches)
                previous_points = previous_features.points[matches[:, 0]]
                current_points = current_features.points[matches[:, 1]]
            if matched_features < config.min_matches:
                raise RuntimeError(f"not enough matches: {matched_features}")
            if config.motion_model == "rgbd":
                if frame.depth_path is None:
                    raise RuntimeError("current frame has no depth map")
                points_previous, points_current = _build_rgbd_3d_correspondences(
                    previous_points,
                    current_points,
                    previous_frame.depth_path,
                    frame.depth_path,
                    dataset.intrinsics,
                    config,
                )
                valid_depth_points = len(points_previous)
                if valid_depth_points < config.min_pnp_points:
                    raise RuntimeError(
                        f"not enough valid depth pairs: {valid_depth_points}"
                    )
                current_from_previous, inliers = _estimate_rigid_transform_ransac(
                    points_previous,
                    points_current,
                    min_inliers=config.min_pnp_inliers,
                    iterations=config.ransac_iterations,
                    threshold=config.ransac_3d_threshold,
                )
                pnp_inliers = int(np.count_nonzero(inliers))
            else:
                points_3d, points_2d = _build_rgbd_pnp_correspondences(
                    previous_points,
                    current_points,
                    previous_frame.depth_path,
                    dataset.intrinsics,
                    config,
                )
                valid_depth_points = len(points_3d)
                if len(points_3d) < config.min_pnp_points:
                    raise RuntimeError(
                        f"not enough valid depth points: {len(points_3d)}"
                    )
                estimate = solve_pnp(
                    points_3d,
                    points_2d,
                    dataset.intrinsics,
                    reprojection_error=config.pnp_reprojection_error,
                )
                pnp_inliers = int(np.count_nonzero(estimate.inliers))
                if pnp_inliers < config.min_pnp_inliers:
                    raise RuntimeError(f"not enough PnP inliers: {pnp_inliers}")
                current_from_previous = _transform_from_pose(estimate.pose)
            _validate_motion_step(current_from_previous, config)
            current_camera_to_world = (
                previous_camera_to_world @ np.linalg.inv(current_from_previous)
            )
            report = Stage3OdometryReport(
                timestamp=frame.timestamp,
                reference_timestamp=previous_frame.timestamp,
                matched_features=matched_features,
                valid_depth_points=valid_depth_points,
                pnp_inliers=pnp_inliers,
                status="tracked",
            )
        except (RuntimeError, ValueError, cv2.error) as exc:
            report = Stage3OdometryReport(
                timestamp=frame.timestamp,
                reference_timestamp=previous_frame.timestamp,
                matched_features=matched_features,
                valid_depth_points=valid_depth_points,
                pnp_inliers=pnp_inliers,
                status=f"failed: {exc}",
            )

        poses.append(
            _trajectory_pose_from_camera_to_world(
                frame.timestamp,
                current_camera_to_world,
            )
        )
        reports.append(report)
        previous_frame = frame
        previous_features = current_features
        previous_camera_to_world = current_camera_to_world

    return Stage3OdometryResult(poses=poses, reports=reports)


def write_stage3_manifest(path: str | Path, dataset: Stage3Dataset) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched_depth_count = sum(frame.depth_path is not None for frame in dataset.frames)
    manifest = {
        "root": str(dataset.root),
        "frame_count": len(dataset.frames),
        "rgb_count": len(dataset.frames),
        "matched_depth_count": matched_depth_count,
        "missing_depth_count": len(dataset.frames) - matched_depth_count,
        "first_timestamp": dataset.frames[0].timestamp if dataset.frames else None,
        "last_timestamp": dataset.frames[-1].timestamp if dataset.frames else None,
        "intrinsics": dataset.intrinsics.tolist(),
        "ground_truth_trajectory": (
            str(dataset.ground_truth_trajectory_path)
            if dataset.ground_truth_trajectory_path is not None
            else None
        ),
        "frames": [
            {
                "timestamp": frame.timestamp,
                "rgb": str(frame.rgb_path),
                "depth": str(frame.depth_path) if frame.depth_path is not None else None,
            }
            for frame in dataset.frames
        ],
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_odometry_report(path: str | Path, reports: list[Stage3OdometryReport]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "reference_timestamp",
                "matched_features",
                "valid_depth_points",
                "pnp_inliers",
                "status",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "timestamp": _format_float(report.timestamp),
                    "reference_timestamp": (
                        ""
                        if report.reference_timestamp is None
                        else _format_float(report.reference_timestamp)
                    ),
                    "matched_features": report.matched_features,
                    "valid_depth_points": report.valid_depth_points,
                    "pnp_inliers": report.pnp_inliers,
                    "status": report.status,
                }
            )


def evaluate_translation_trajectory(
    ground_truth: list[TrajectoryPose],
    estimated: list[TrajectoryPose],
    *,
    timestamp_decimals: int = 6,
) -> Stage3TrajectoryMetrics:
    ground_truth_by_timestamp = {
        round(pose.timestamp, timestamp_decimals): pose for pose in ground_truth
    }
    errors = []
    for pose in estimated:
        key = round(pose.timestamp, timestamp_decimals)
        reference = ground_truth_by_timestamp.get(key)
        if reference is None:
            continue
        errors.append(float(np.linalg.norm(reference.translation - pose.translation)))
    if not errors:
        raise ValueError("No matching timestamps between trajectories")
    values = np.asarray(errors, dtype=np.float64)
    return Stage3TrajectoryMetrics(
        matched_poses=len(values),
        translation_rmse=float(np.sqrt(np.mean(values * values))),
        translation_mean=float(np.mean(values)),
        translation_median=float(np.median(values)),
        translation_max=float(np.max(values)),
    )


def write_trajectory_metrics(path: str | Path, metrics: Stage3TrajectoryMetrics) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "matched_poses": metrics.matched_poses,
                "translation_rmse": metrics.translation_rmse,
                "translation_mean": metrics.translation_mean,
                "translation_median": metrics.translation_median,
                "translation_max": metrics.translation_max,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def apply_loop_closure(
    poses: list[TrajectoryPose],
    *,
    target: str = "first",
    window_fraction: float = 0.7,
    correct_rotation: bool = False,
) -> tuple[list[TrajectoryPose], Stage3LoopClosureSummary]:
    if len(poses) < 2:
        raise ValueError("At least two poses are required for loop closure")
    if target != "first":
        raise ValueError("target must be 'first'")
    if not 0.0 < window_fraction <= 1.0:
        raise ValueError("window_fraction must be in (0, 1]")

    transforms = [_transform_from_trajectory_pose(pose) for pose in poses]
    target_transform = transforms[0]
    final_transform = transforms[-1]
    correction_end = target_transform @ np.linalg.inv(final_transform)
    correction_rotation = correction_end[:3, :3]
    if correct_rotation:
        correction_translation = correction_end[:3, 3]
    else:
        correction_translation = target_transform[:3, 3] - final_transform[:3, 3]
    corrected: list[TrajectoryPose] = []
    start_index = int((1.0 - window_fraction) * (len(transforms) - 1))
    denominator = max(len(transforms) - 1 - start_index, 1)
    for index, (pose, transform) in enumerate(zip(poses, transforms)):
        alpha = 0.0 if index < start_index else (index - start_index) / denominator
        correction = np.eye(4, dtype=np.float64)
        if correct_rotation:
            correction[:3, :3] = _slerp_rotation(
                np.eye(3, dtype=np.float64),
                correction_rotation,
                alpha,
            )
        correction[:3, 3] = alpha * correction_translation
        corrected_transform = correction @ transform
        corrected.append(
            _trajectory_pose_from_camera_to_world(
                pose.timestamp,
                corrected_transform,
            )
        )

    target_translation = target_transform[:3, 3]
    summary = Stage3LoopClosureSummary(
        poses=len(poses),
        start_timestamp=poses[0].timestamp,
        end_timestamp=poses[-1].timestamp,
        original_end_translation_error=float(
            np.linalg.norm(final_transform[:3, 3] - target_translation)
        ),
        corrected_end_translation_error=float(
            np.linalg.norm(
                _transform_from_trajectory_pose(corrected[-1])[:3, 3]
                - target_translation
            )
        ),
        target=target,
    )
    return corrected, summary


def write_loop_closure_summary(
    path: str | Path,
    summary: Stage3LoopClosureSummary,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "poses": summary.poses,
                "start_timestamp": summary.start_timestamp,
                "end_timestamp": summary.end_timestamp,
                "original_end_translation_error": (
                    summary.original_end_translation_error
                ),
                "corrected_end_translation_error": (
                    summary.corrected_end_translation_error
                ),
                "target": summary.target,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Stage 3 RGB-D SLAM data.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-trajectory",
        type=Path,
        help="Existing Stage 3 trajectory to post-process.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--allow-missing-depth", action="store_true")
    parser.add_argument(
        "--write-trajectory-template",
        action="store_true",
        help="Write an identity-pose trajectory template in evaluator format.",
    )
    parser.add_argument(
        "--run-rgbd-odometry",
        action="store_true",
        help="Estimate a first-pass RGB-D visual odometry trajectory.",
    )
    parser.add_argument(
        "--motion-model",
        choices=("lk_pnp", "pnp", "rgbd"),
        default="pnp",
    )
    parser.add_argument("--feature-mode", choices=("akaze", "sift"), default="akaze")
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--ratio-threshold", type=float, default=0.8)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--min-pnp-points", type=int, default=12)
    parser.add_argument("--min-pnp-inliers", type=int, default=10)
    parser.add_argument("--pnp-reprojection-error", type=float, default=2.0)
    parser.add_argument("--ransac-iterations", type=int, default=200)
    parser.add_argument("--ransac-3d-threshold", type=float, default=0.05)
    parser.add_argument("--lk-quality-level", type=float, default=0.01)
    parser.add_argument("--lk-min-distance", type=float, default=7.0)
    parser.add_argument("--max-translation-step", type=float, default=0.2)
    parser.add_argument("--max-rotation-step-degrees", type=float, default=20.0)
    parser.add_argument(
        "--apply-loop-closure",
        action="store_true",
        help="Apply a first-to-last loop-closure correction to the trajectory.",
    )
    parser.add_argument("--loop-closure-window-fraction", type=float, default=0.7)
    parser.add_argument("--loop-closure-correct-rotation", action="store_true")
    # Lazy import avoids a circular import (pose_graph imports from stage3).
    from .pose_graph import add_pose_graph_arguments
    from .official_eval import add_official_eval_arguments

    add_pose_graph_arguments(parser)
    add_official_eval_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = load_stage3_dataset(
        args.dataset,
        allow_missing_depth=args.allow_missing_depth,
    ).subset(args.max_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "stage3_manifest.json"
    write_stage3_manifest(manifest_path, dataset)
    print(
        f"Validated {len(dataset.frames)} RGB-D frames; "
        f"manifest: {manifest_path.resolve()}"
    )
    if dataset.ground_truth_trajectory_path is not None:
        gt = load_trajectory(dataset.ground_truth_trajectory_path)
        print(f"Ground-truth trajectory poses: {len(gt)}")
    else:
        print("Ground-truth trajectory: not found")
    if args.write_trajectory_template:
        trajectory_path = args.output_dir / "estimated_camera_trajectory.txt"
        write_trajectory(trajectory_path, identity_trajectory_template(dataset))
        print(f"Trajectory template: {trajectory_path.resolve()}")
    odometry_config = Stage3OdometryConfig(
        motion_model=args.motion_model,
        feature_mode=args.feature_mode,
        max_features=args.max_features,
        image_scale=args.image_scale,
        ratio_threshold=args.ratio_threshold,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_matches=args.min_matches,
        min_pnp_points=args.min_pnp_points,
        min_pnp_inliers=args.min_pnp_inliers,
        pnp_reprojection_error=args.pnp_reprojection_error,
        ransac_iterations=args.ransac_iterations,
        ransac_3d_threshold=args.ransac_3d_threshold,
        lk_quality_level=args.lk_quality_level,
        lk_min_distance=args.lk_min_distance,
        max_translation_step=args.max_translation_step,
        max_rotation_step_degrees=args.max_rotation_step_degrees,
    )
    trajectory_for_loop_closure: list[TrajectoryPose] | None = None
    odometry_result: Stage3OdometryResult | None = None
    raw_metrics: Stage3TrajectoryMetrics | None = None
    loop_closed_metrics: Stage3TrajectoryMetrics | None = None
    produced_trajectories: list[tuple[str, Path]] = []
    if args.run_rgbd_odometry:
        result = estimate_rgbd_trajectory(dataset, odometry_config)
        odometry_result = result
        trajectory_path = args.output_dir / "estimated_camera_trajectory.txt"
        report_path = args.output_dir / "rgbd_odometry_report.csv"
        metrics_path = args.output_dir / "trajectory_metrics.json"
        write_trajectory(trajectory_path, result.poses)
        trajectory_for_loop_closure = result.poses
        produced_trajectories.append(("raw_vo", trajectory_path))
        write_odometry_report(report_path, result.reports)
        tracked = sum(report.status == "tracked" for report in result.reports)
        print(
            f"RGB-D odometry tracked {tracked}/{max(len(result.reports) - 1, 0)} "
            f"frame transitions."
        )
        print(f"Estimated trajectory: {trajectory_path.resolve()}")
        print(f"Odometry report: {report_path.resolve()}")
        if dataset.ground_truth_trajectory_path is not None:
            metrics = evaluate_translation_trajectory(
                load_trajectory(dataset.ground_truth_trajectory_path),
                result.poses,
            )
            raw_metrics = metrics
            write_trajectory_metrics(metrics_path, metrics)
            print(
                "Translation RMSE vs ground truth: "
                f"{metrics.translation_rmse:.6f} "
                f"over {metrics.matched_poses} poses."
            )
            print(f"Trajectory metrics: {metrics_path.resolve()}")
    if args.input_trajectory is not None:
        trajectory_for_loop_closure = load_trajectory(args.input_trajectory)
    if args.apply_loop_closure:
        if trajectory_for_loop_closure is None:
            raise ValueError(
                "--apply-loop-closure requires --run-rgbd-odometry "
                "or --input-trajectory"
            )
        closed, summary = apply_loop_closure(
            trajectory_for_loop_closure,
            window_fraction=args.loop_closure_window_fraction,
            correct_rotation=args.loop_closure_correct_rotation,
        )
        closed_path = args.output_dir / "estimated_camera_trajectory_loop_closed.txt"
        summary_path = args.output_dir / "loop_closure_summary.json"
        write_trajectory(closed_path, closed)
        produced_trajectories.append(("loop_closed", closed_path))
        write_loop_closure_summary(summary_path, summary)
        print(f"Loop-closed trajectory: {closed_path.resolve()}")
        print(f"Loop-closure summary: {summary_path.resolve()}")
        if dataset.ground_truth_trajectory_path is not None:
            closed_metrics = evaluate_translation_trajectory(
                load_trajectory(dataset.ground_truth_trajectory_path),
                closed,
            )
            loop_closed_metrics = closed_metrics
            closed_metrics_path = (
                args.output_dir / "trajectory_metrics_loop_closed.json"
            )
            write_trajectory_metrics(closed_metrics_path, closed_metrics)
            print(
                "Loop-closed translation RMSE vs ground truth: "
                f"{closed_metrics.translation_rmse:.6f} "
                f"over {closed_metrics.matched_poses} poses."
            )
            print(f"Loop-closed metrics: {closed_metrics_path.resolve()}")

    if args.run_pose_graph:
        from .pose_graph import pose_graph_config_from_args, run_pose_graph_pipeline

        pose_graph_config = pose_graph_config_from_args(args)
        ground_truth = (
            load_trajectory(dataset.ground_truth_trajectory_path)
            if dataset.ground_truth_trajectory_path is not None
            else None
        )
        # Reuse an existing VO trajectory (e.g. from a prior run) instead of
        # recomputing odometry, when one is supplied and VO was not run here.
        vo_result_for_graph = odometry_result
        if vo_result_for_graph is None and args.input_trajectory is not None:
            vo_result_for_graph = Stage3OdometryResult(
                poses=load_trajectory(args.input_trajectory),
                reports=[],
            )
            print(
                f"Pose graph reusing VO trajectory: "
                f"{Path(args.input_trajectory).resolve()}"
            )
        pipeline = run_pose_graph_pipeline(
            dataset,
            odometry_config,
            pose_graph_config,
            args.output_dir,
            vo_result=vo_result_for_graph,
            ground_truth=ground_truth,
            raw_metrics=raw_metrics,
            loop_closed_metrics=loop_closed_metrics,
        )
        print(
            f"Pose graph: {len(pipeline.graph.nodes)} nodes, "
            f"{len(pipeline.graph.odometry_edges)} odometry edges, "
            f"{len(pipeline.graph.loop_edges)} loop edges "
            f"({len(pipeline.accepted)} accepted of "
            f"{len(pipeline.candidates)} candidates)."
        )
        print(
            "Pose-graph optimization: "
            f"{pipeline.optimization.iterations} iterations, "
            f"cost {pipeline.optimization.initial_cost:.6g} -> "
            f"{pipeline.optimization.final_cost:.6g}."
        )
        print(f"Pose-graph trajectory: {pipeline.trajectory_path.resolve()}")
        produced_trajectories.append(("pose_graph", pipeline.trajectory_path))
        if pipeline.pose_graph_metrics is not None:
            print(
                "Pose-graph translation RMSE vs ground truth: "
                f"{pipeline.pose_graph_metrics.translation_rmse:.6f} "
                f"over {pipeline.pose_graph_metrics.matched_poses} poses."
            )

    if args.run_official_eval:
        from .official_eval import run_official_eval, write_official_eval_summary

        if dataset.ground_truth_trajectory_path is None:
            print("Official eval skipped: no ground-truth trajectory found.")
        elif not produced_trajectories:
            print(
                "Official eval skipped: no trajectory produced "
                "(use --run-rgbd-odometry / --run-pose-graph)."
            )
        else:
            results = []
            for label, trajectory_path in produced_trajectories:
                result = run_official_eval(
                    dataset.ground_truth_trajectory_path,
                    trajectory_path,
                    args.output_dir,
                    label=label,
                    evaluator_dir=args.evaluator_dir,
                    max_difference=args.official_eval_max_difference,
                )
                results.append(result)
                print(
                    f"Official ATE RMSE ({label}): "
                    f"{result.rmse if result.rmse is not None else 'n/a'} "
                    f"(scale {result.scale:.6f}, returncode {result.returncode})."
                )
            summary_path = args.output_dir / "official_eval_summary.json"
            write_official_eval_summary(summary_path, results)
            print(f"Official eval summary: {summary_path.resolve()}")


def _load_intrinsics(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing camera parameter JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{path} intrinsics must have shape (3, 3)")
    return intrinsics


def _timestamped_files(root: Path) -> dict[float, Path]:
    files: dict[float, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            timestamp = float(path.stem)
        except ValueError:
            continue
        if timestamp in files:
            raise ValueError(f"Duplicate timestamp {timestamp:g} under {root}")
        files[timestamp] = path
    return files


def _extract_stage3_features(
    image_path: Path,
    config: Stage3OdometryConfig,
) -> Stage3FrameFeatures:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read RGB image: {image_path}")
    point_scale = 1.0
    if config.image_scale != 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=config.image_scale,
            fy=config.image_scale,
            interpolation=cv2.INTER_AREA,
        )
        point_scale = 1.0 / config.image_scale
    if config.feature_mode == "akaze":
        detector = cv2.AKAZE_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        descriptor_type = "binary"
    elif config.feature_mode == "sift":
        detector = cv2.SIFT_create(nfeatures=config.max_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        descriptor_type = "float"
        if descriptors is not None:
            descriptors = root_sift(descriptors)
    else:
        raise ValueError("feature_mode must be one of: akaze, sift")

    if descriptors is None or not keypoints:
        raise RuntimeError(f"No features detected in {image_path}")
    if config.max_features > 0 and len(keypoints) > config.max_features:
        responses = np.asarray(
            [keypoint.response for keypoint in keypoints], dtype=np.float64
        )
        selected = np.argsort(responses)[-config.max_features :]
        keypoints = [keypoints[index] for index in selected]
        descriptors = descriptors[selected]
    return Stage3FrameFeatures(
        points=np.asarray(
            [
                (keypoint.pt[0] * point_scale, keypoint.pt[1] * point_scale)
                for keypoint in keypoints
            ],
            dtype=np.float64,
        ),
        descriptors=descriptors,
        descriptor_type=descriptor_type,
    )


def _build_rgbd_pnp_correspondences(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    reference_depth_path: Path,
    intrinsics: np.ndarray,
    config: Stage3OdometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    depth = cv2.imread(str(reference_depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth map: {reference_depth_path}")

    height, width = depth.shape[:2]
    rounded = np.rint(reference_points).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    rounded = rounded[in_bounds]
    reference_points = reference_points[in_bounds]
    current_points = current_points[in_bounds]
    raw_depth = depth[rounded[:, 1], rounded[:, 0]].astype(np.float64)
    z = raw_depth / config.depth_scale
    valid = (raw_depth > 0) & (z >= config.min_depth) & (z <= config.max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    reference_points = reference_points[valid]
    current_points = current_points[valid]
    z = z[valid]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (reference_points[:, 0] - cx) * z / fx
    y = (reference_points[:, 1] - cy) * z / fy
    points_3d = np.column_stack((x, y, z))
    return points_3d.astype(np.float64), current_points.astype(np.float64)


def _track_lk_correspondences(
    reference_image_path: Path,
    current_image_path: Path,
    config: Stage3OdometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    reference_gray = cv2.imread(str(reference_image_path), cv2.IMREAD_GRAYSCALE)
    current_gray = cv2.imread(str(current_image_path), cv2.IMREAD_GRAYSCALE)
    if reference_gray is None:
        raise RuntimeError(f"Could not read RGB image: {reference_image_path}")
    if current_gray is None:
        raise RuntimeError(f"Could not read RGB image: {current_image_path}")

    point_scale = 1.0
    if config.image_scale != 1.0:
        reference_gray = cv2.resize(
            reference_gray,
            None,
            fx=config.image_scale,
            fy=config.image_scale,
            interpolation=cv2.INTER_AREA,
        )
        current_gray = cv2.resize(
            current_gray,
            None,
            fx=config.image_scale,
            fy=config.image_scale,
            interpolation=cv2.INTER_AREA,
        )
        point_scale = 1.0 / config.image_scale

    corners = cv2.goodFeaturesToTrack(
        reference_gray,
        maxCorners=max(config.max_features, 1),
        qualityLevel=config.lk_quality_level,
        minDistance=config.lk_min_distance,
        blockSize=7,
    )
    if corners is None or len(corners) == 0:
        raise RuntimeError(f"No trackable corners detected in {reference_image_path}")

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        current_gray,
        corners.astype(np.float32),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )
    if next_points is None or status is None:
        raise RuntimeError("Lucas-Kanade optical flow failed")
    status = status.ravel().astype(bool)
    previous_points = corners.reshape(-1, 2)[status].astype(np.float64)
    current_points = next_points.reshape(-1, 2)[status].astype(np.float64)
    previous_points *= point_scale
    current_points *= point_scale
    return previous_points, current_points


def _build_rgbd_3d_correspondences(
    reference_points: np.ndarray,
    current_points: np.ndarray,
    reference_depth_path: Path,
    current_depth_path: Path,
    intrinsics: np.ndarray,
    config: Stage3OdometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    reference_3d, valid_reference = _backproject_points(
        reference_points,
        reference_depth_path,
        intrinsics,
        config,
    )
    current_3d, valid_current = _backproject_points(
        current_points,
        current_depth_path,
        intrinsics,
        config,
    )
    valid = valid_reference & valid_current
    return reference_3d[valid], current_3d[valid]


def _backproject_points(
    points: np.ndarray,
    depth_path: Path,
    intrinsics: np.ndarray,
    config: Stage3OdometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth map: {depth_path}")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    height, width = depth.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    raw_depth = np.zeros(len(points), dtype=np.float64)
    raw_depth[in_bounds] = depth[
        rounded[in_bounds, 1], rounded[in_bounds, 0]
    ].astype(np.float64)
    z = raw_depth / config.depth_scale
    valid = in_bounds & (raw_depth > 0) & (z >= config.min_depth) & (z <= config.max_depth)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = (points[:, 0] - cx) * z / fx
    y = (points[:, 1] - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float64), valid


def _estimate_rigid_transform_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    min_inliers: int,
    iterations: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_points = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(source_points) != len(target_points):
        raise ValueError("source_points and target_points must have the same length")
    if len(source_points) < 3:
        raise RuntimeError("At least three 3D correspondences are required")

    rng = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_error = np.inf
    sample_count = min(3, len(source_points))
    for _ in range(iterations):
        sample = rng.choice(len(source_points), size=sample_count, replace=False)
        try:
            candidate = _rigid_transform(source_points[sample], target_points[sample])
        except ValueError:
            continue
        residuals = _rigid_residuals(candidate, source_points, target_points)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        if count < min_inliers:
            continue
        error = float(np.mean(residuals[inliers]))
        if best_inliers is None or count > np.count_nonzero(best_inliers) or (
            count == np.count_nonzero(best_inliers) and error < best_error
        ):
            best_inliers = inliers
            best_error = error

    if best_inliers is None:
        raise RuntimeError("3D rigid-transform RANSAC failed")
    transform = _rigid_transform(source_points[best_inliers], target_points[best_inliers])
    residuals = _rigid_residuals(transform, source_points, target_points)
    refined_inliers = residuals <= threshold
    if np.count_nonzero(refined_inliers) >= min_inliers:
        transform = _rigid_transform(
            source_points[refined_inliers],
            target_points[refined_inliers],
        )
        best_inliers = refined_inliers
    return transform, best_inliers


def _rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source_points = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target_points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(source_points) != len(target_points) or len(source_points) < 3:
        raise ValueError("At least three paired 3D points are required")
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _rigid_residuals(
    transform: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    transformed = (
        transform[:3, :3] @ np.asarray(source_points, dtype=np.float64).T
        + transform[:3, 3:4]
    ).T
    return np.linalg.norm(transformed - target_points, axis=1)


def _transform_from_pose(pose: Pose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose.rotation
    transform[:3, 3] = pose.translation.ravel()
    return transform


def _validate_motion_step(
    current_from_previous: np.ndarray,
    config: Stage3OdometryConfig,
) -> None:
    if config.max_translation_step is not None:
        translation_step = float(np.linalg.norm(current_from_previous[:3, 3]))
        if translation_step > config.max_translation_step:
            raise RuntimeError(
                f"translation step {translation_step:.3f} exceeds "
                f"{config.max_translation_step:.3f}"
            )
    if config.max_rotation_step_degrees is not None:
        trace = float(np.trace(current_from_previous[:3, :3]))
        angle = np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        if angle > config.max_rotation_step_degrees:
            raise RuntimeError(
                f"rotation step {angle:.3f} deg exceeds "
                f"{config.max_rotation_step_degrees:.3f} deg"
            )


def _trajectory_pose_from_camera_to_world(
    timestamp: float,
    camera_to_world: np.ndarray,
) -> TrajectoryPose:
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape (4, 4)")
    return TrajectoryPose(
        timestamp=timestamp,
        translation=camera_to_world[:3, 3],
        quaternion_xyzw=_quaternion_xyzw_from_rotation(camera_to_world[:3, :3]),
    )


def _transform_from_trajectory_pose(pose: TrajectoryPose) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_quaternion_xyzw(pose.quaternion_xyzw)
    transform[:3, 3] = pose.translation
    return transform


def _rotation_from_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([qx, qy, qz, qw])
    if norm == 0.0:
        raise ValueError("quaternion must be nonzero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def _slerp_rotation(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first_q = _quaternion_xyzw_from_rotation(first)
    second_q = _quaternion_xyzw_from_rotation(second)
    quaternion = _slerp_quaternion(first_q, second_q, alpha)
    return _rotation_from_quaternion_xyzw(quaternion)


def _slerp_quaternion(
    first: np.ndarray,
    second: np.ndarray,
    alpha: float,
) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(4)
    second = np.asarray(second, dtype=np.float64).reshape(4)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + alpha * (second - first)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    scale_first = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale_second = sin_theta / sin_theta_0
    return scale_first * first + scale_second * second


def _quaternion_xyzw_from_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _format_float(value: float) -> str:
    return f"{value:.9g}"


if __name__ == "__main__":
    main()
