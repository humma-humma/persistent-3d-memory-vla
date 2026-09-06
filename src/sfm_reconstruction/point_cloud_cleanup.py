from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .open3d_viewer import load_ascii_ply_table


@dataclass(frozen=True)
class CleanupConfig:
    statistical_neighbors: int | None = 20
    statistical_std_ratio: float = 2.0
    radius: float | None = None
    radius_min_neighbors: int = 8
    voxel_size: float | None = None


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for point-cloud cleanup. Use the "
            "`mardm_open3d` environment or install `open3d>=0.19`."
        ) from exc
    return o3d


def _rgb_from_table(table) -> np.ndarray:
    channels = [table.property(name) for name in ("red", "green", "blue")]
    if all(channel is not None for channel in channels):
        colors = np.column_stack(channels)
        return np.clip(colors, 0, 255).astype(np.uint8)
    return np.full((len(table.values), 3), 255, dtype=np.uint8)


def write_rgb_point_cloud(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> int:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return len(points)


def clean_point_cloud(
    input_path: str | Path,
    output_path: str | Path,
    config: CleanupConfig | None = None,
) -> dict[str, object]:
    config = config or CleanupConfig()
    if config.statistical_neighbors is not None and config.statistical_neighbors < 2:
        raise ValueError("statistical_neighbors must be at least 2")
    if config.statistical_std_ratio <= 0.0:
        raise ValueError("statistical_std_ratio must be positive")
    if config.radius is not None and config.radius <= 0.0:
        raise ValueError("radius must be positive")
    if config.radius_min_neighbors < 1:
        raise ValueError("radius_min_neighbors must be positive")
    if config.voxel_size is not None and config.voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")

    o3d = _require_open3d()
    input_path = Path(input_path)
    output_path = Path(output_path)
    table = load_ascii_ply_table(input_path)
    colors = _rgb_from_table(table)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(table.points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    counts = {"input": len(table.points)}

    if config.statistical_neighbors is not None and len(pcd.points):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=config.statistical_neighbors,
            std_ratio=config.statistical_std_ratio,
        )
    counts["after_statistical"] = len(pcd.points)

    if config.radius is not None and len(pcd.points):
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=config.radius_min_neighbors,
            radius=config.radius,
        )
    counts["after_radius"] = len(pcd.points)

    if config.voxel_size is not None and len(pcd.points):
        pcd = pcd.voxel_down_sample(config.voxel_size)
    counts["after_voxel"] = len(pcd.points)

    points = np.asarray(pcd.points)
    output_colors = np.rint(np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
    written = write_rgb_point_cloud(output_path, points, output_colors)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "counts": counts,
        "points_written": int(written),
        "config": asdict(config),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean a dense SfM point cloud with Open3D filters."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-statistical", action="store_true")
    parser.add_argument("--statistical-neighbors", type=int, default=20)
    parser.add_argument("--statistical-std-ratio", type=float, default=2.0)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--radius-min-neighbors", type=int, default=8)
    parser.add_argument("--voxel-size", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = (
        args.output.resolve()
        if args.output is not None
        else args.input.resolve().with_name(f"{args.input.stem}_cleaned.ply")
    )
    config = CleanupConfig(
        statistical_neighbors=None
        if args.no_statistical
        else args.statistical_neighbors,
        statistical_std_ratio=args.statistical_std_ratio,
        radius=args.radius,
        radius_min_neighbors=args.radius_min_neighbors,
        voxel_size=args.voxel_size,
    )
    summary = clean_point_cloud(args.input.resolve(), output, config)
    print(f"Wrote {summary['points_written']} cleaned points: {output}")


if __name__ == "__main__":
    main()
