from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .stage3 import TrajectoryPose, load_trajectory


@dataclass(frozen=True)
class NamedTrajectory:
    label: str
    poses: list[TrajectoryPose]


def _trajectory_array(poses: list[TrajectoryPose]) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray([pose.timestamp for pose in poses], dtype=np.float64)
    points = np.asarray([pose.translation for pose in poses], dtype=np.float64)
    return timestamps, points


def _match_by_timestamp(
    reference: list[TrajectoryPose],
    estimated: list[TrajectoryPose],
) -> tuple[np.ndarray, np.ndarray]:
    reference_by_time = {f"{pose.timestamp:.9f}": pose.translation for pose in reference}
    ref_points: list[np.ndarray] = []
    est_points: list[np.ndarray] = []
    for pose in estimated:
        ref = reference_by_time.get(f"{pose.timestamp:.9f}")
        if ref is None:
            continue
        ref_points.append(ref)
        est_points.append(pose.translation)
    if not ref_points:
        raise ValueError("No matching timestamps between reference and estimate")
    return np.asarray(ref_points, dtype=np.float64), np.asarray(est_points, dtype=np.float64)


def similarity_align_points(
    reference: np.ndarray,
    estimated: np.ndarray,
) -> np.ndarray:
    """Align estimated points to reference with scale, rotation, translation."""
    if reference.shape != estimated.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and estimated must both have shape (N, 3)")
    if len(reference) < 3:
        return estimated.copy()

    ref_mean = reference.mean(axis=0)
    est_mean = estimated.mean(axis=0)
    ref_centered = reference - ref_mean
    est_centered = estimated - est_mean

    covariance = (est_centered.T @ ref_centered) / len(reference)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(est_centered * est_centered, axis=1)))
    if variance <= 1e-12:
        return estimated.copy()
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    return scale * (estimated - est_mean) @ rotation + ref_mean


def _alignment_parameters(
    reference: list[TrajectoryPose],
    estimated: list[TrajectoryPose],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    ref_matched, est_matched = _match_by_timestamp(reference, estimated)
    est_mean = est_matched.mean(axis=0)
    ref_mean = ref_matched.mean(axis=0)
    est_centered = est_matched - est_mean
    ref_centered = ref_matched - ref_mean
    covariance = (est_centered.T @ ref_centered) / len(ref_matched)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(est_centered * est_centered, axis=1)))
    if variance <= 1e-12:
        return ref_matched, est_matched, 1.0, np.eye(3, dtype=np.float64), np.zeros(3)
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    return ref_matched, est_matched, scale, rotation, ref_mean - scale * est_mean @ rotation


def aligned_trajectory_points(
    reference: list[TrajectoryPose],
    estimated: list[TrajectoryPose],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_matched, est_matched, scale, rotation, offset = _alignment_parameters(
        reference, estimated
    )
    _, all_estimated = _trajectory_array(estimated)
    aligned_matched = scale * est_matched @ rotation + offset
    aligned_all = scale * all_estimated @ rotation + offset
    return ref_matched, aligned_matched, aligned_all


def plot_stage3_trajectories(
    reference: NamedTrajectory,
    estimates: list[NamedTrajectory],
    output: str | Path,
    *,
    title: str = "Stage 3 Trajectory Comparison",
) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    _, reference_points = _trajectory_array(reference.poses)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    top_axis, error_axis = axes

    top_axis.plot(
        reference_points[:, 0],
        reference_points[:, 2],
        color="black",
        linewidth=2.0,
        label=reference.label,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, estimate in enumerate(estimates):
        ref_matched, aligned_matched, aligned_points = aligned_trajectory_points(
            reference.poses, estimate.poses
        )
        color = colors[index % len(colors)]
        top_axis.plot(
            aligned_points[:, 0],
            aligned_points[:, 2],
            linewidth=1.4,
            color=color,
            label=estimate.label,
        )
        errors = np.linalg.norm(aligned_matched - ref_matched, axis=1)
        error_axis.plot(
            np.arange(len(errors)),
            errors,
            linewidth=1.1,
            color=color,
            label=f"{estimate.label} RMSE {np.sqrt(np.mean(errors * errors)):.3f}",
        )

    top_axis.set_title("Top-down path (X/Z, similarity-aligned)")
    top_axis.set_xlabel("X")
    top_axis.set_ylabel("Z")
    top_axis.axis("equal")
    top_axis.grid(True, alpha=0.25)
    top_axis.legend(loc="best")

    error_axis.set_title("Aligned translation error")
    error_axis.set_xlabel("Matched pose index")
    error_axis.set_ylabel("Error")
    error_axis.grid(True, alpha=0.25)
    error_axis.legend(loc="best")

    figure.suptitle(title)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved Stage 3 trajectory plot: {output.resolve()}")


def _parse_trajectory_arg(value: str) -> NamedTrajectory:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trajectory must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("trajectory label must be non-empty")
    return NamedTrajectory(label=label, poses=load_trajectory(Path(path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Stage 3 ground-truth and estimated trajectories."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        type=_parse_trajectory_arg,
        help="Estimated trajectory as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="Stage 3 Trajectory Comparison")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    reference = NamedTrajectory(
        label="Ground truth",
        poses=load_trajectory(args.ground_truth),
    )
    plot_stage3_trajectories(
        reference,
        args.trajectory,
        args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
