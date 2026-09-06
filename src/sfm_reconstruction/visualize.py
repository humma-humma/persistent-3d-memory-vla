from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .models import Pose


def load_ascii_ply(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="ascii") as file:
        first_line = file.readline().strip()
        if first_line != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = None
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has no end_header marker")
            fields = line.strip().split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            if fields == ["end_header"]:
                break
        if vertex_count is None:
            raise ValueError(f"{path} has no vertex count")
        points = np.loadtxt(file, dtype=np.float64, max_rows=vertex_count, ndmin=2)
    if points.shape[1] < 3:
        raise ValueError(f"{path} vertices must contain x, y and z")
    return points[:, :3]


def load_camera_poses(path: str | Path) -> dict[str, Pose]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = {}
    for name, matrix in data["extrinsics"].items():
        transform = np.asarray(matrix, dtype=np.float64)
        poses[name] = Pose(transform[:3, :3], transform[:3, 3])
    return poses


def _set_equal_axes(axis, values: np.ndarray) -> None:
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_reconstruction(
    points: np.ndarray,
    poses: dict[str, Pose],
    output: str | Path | None = None,
    show: bool = False,
    max_points: int = 20_000,
    camera_scale: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    finite_points = points[np.isfinite(points).all(axis=1)]
    centers = np.asarray([pose.camera_center for pose in poses.values()])
    combined = np.vstack((finite_points, centers))

    lower, upper = np.percentile(combined, [1, 99], axis=0)
    in_bounds = np.all((finite_points >= lower) & (finite_points <= upper), axis=1)
    visible_points = finite_points[in_bounds]
    if len(visible_points) > max_points:
        indices = np.linspace(0, len(visible_points) - 1, max_points, dtype=int)
        visible_points = visible_points[indices]

    scene_scale = float(np.linalg.norm(upper - lower))
    camera_scale = camera_scale or 0.025 * scene_scale

    figure = plt.figure(figsize=(11, 9))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        visible_points[:, 0],
        visible_points[:, 1],
        visible_points[:, 2],
        s=1.0,
        c=visible_points[:, 2],
        cmap="viridis",
        alpha=0.55,
        label=f"Points ({len(points):,})",
    )

    ordered = sorted(poses.items())
    ordered_centers = np.asarray([pose.camera_center for _, pose in ordered])
    axis.plot(
        ordered_centers[:, 0],
        ordered_centers[:, 1],
        ordered_centers[:, 2],
        color="crimson",
        linewidth=1.0,
        alpha=0.7,
        label=f"Cameras ({len(poses)})",
    )
    axis.scatter(
        ordered_centers[:, 0],
        ordered_centers[:, 1],
        ordered_centers[:, 2],
        color="crimson",
        marker="^",
        s=18,
    )
    for _, pose in ordered:
        forward_world = pose.rotation.T @ np.array([0.0, 0.0, 1.0])
        axis.quiver(
            *pose.camera_center,
            *forward_world,
            length=camera_scale,
            normalize=True,
            color="darkorange",
            linewidth=0.6,
        )

    _set_equal_axes(axis, np.vstack((visible_points, ordered_centers)))
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("Sparse SfM Reconstruction")
    axis.legend(loc="upper right")
    figure.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved visualization: {output.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize an SfM point cloud and camera poses."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--camera-scale", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    points = load_ascii_ply(args.result_dir / "estimated_points.ply")
    poses = load_camera_poses(
        args.result_dir / "estimated_camera_parameters.json"
    )
    output = args.output
    if output is None and not args.show:
        output = args.result_dir / "reconstruction.png"
    plot_reconstruction(
        points,
        poses,
        output=output,
        show=args.show,
        max_points=args.max_points,
        camera_scale=args.camera_scale,
    )


if __name__ == "__main__":
    main()
