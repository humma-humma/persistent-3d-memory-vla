from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .models import Pose


@dataclass(frozen=True)
class PlyTable:
    property_names: list[str]
    values: np.ndarray

    @property
    def points(self) -> np.ndarray:
        indices = [self.property_names.index(name) for name in ("x", "y", "z")]
        return self.values[:, indices].astype(np.float64, copy=False)

    def property(self, name: str) -> np.ndarray | None:
        if name not in self.property_names:
            return None
        return self.values[:, self.property_names.index(name)]


def load_ascii_ply_table(path: str | Path) -> PlyTable:
    path = Path(path)
    with path.open("r", encoding="ascii") as file:
        if file.readline().strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        if file.readline().strip() != "format ascii 1.0":
            raise ValueError(f"{path} must be an ASCII PLY file")

        vertex_count: int | None = None
        current_element: str | None = None
        property_names: list[str] = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has no end_header marker")
            fields = line.strip().split()
            if fields == ["end_header"]:
                break
            if len(fields) >= 3 and fields[0] == "element":
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
                continue
            if fields and fields[0] == "property" and current_element == "vertex":
                property_names.append(fields[-1])

        if vertex_count is None:
            raise ValueError(f"{path} has no vertex count")
        for name in ("x", "y", "z"):
            if name not in property_names:
                raise ValueError(f"{path} is missing vertex property {name!r}")
        if vertex_count == 0:
            values = np.empty((0, len(property_names)), dtype=np.float64)
        else:
            values = np.loadtxt(
                file,
                dtype=np.float64,
                max_rows=vertex_count,
                ndmin=2,
            )
            if values.shape[1] != len(property_names):
                raise ValueError(
                    f"{path} has {values.shape[1]} columns but "
                    f"{len(property_names)} vertex properties"
                )
    return PlyTable(property_names, values)


def choose_point_cloud_path(result_dir: Path, point_cloud: Path | None) -> Path:
    if point_cloud is not None:
        return point_cloud
    rich_path = result_dir / "estimated_points_rich.ply"
    if rich_path.is_file():
        return rich_path
    return result_dir / "estimated_points.ply"


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float64)
    lower, upper = np.percentile(values[finite], [5, 95])
    if upper <= lower:
        normalized = np.full_like(values, 0.5, dtype=np.float64)
    else:
        normalized = (values - lower) / (upper - lower)
    normalized[~finite] = 0.0
    return np.clip(normalized, 0.0, 1.0)


def _scalar_colors(values: np.ndarray, palette: str) -> np.ndarray:
    t = _normalize(values)
    if palette == "error":
        return np.column_stack((t, 1.0 - 0.75 * t, 0.15 * (1.0 - t)))
    return np.column_stack((t, 0.35 + 0.5 * (1.0 - np.abs(2.0 * t - 1.0)), 1.0 - t))


def colors_for_mode(table: PlyTable, mode: str) -> np.ndarray:
    if mode == "rgb":
        channels = [table.property(name) for name in ("red", "green", "blue")]
        if all(channel is not None for channel in channels):
            return np.clip(np.column_stack(channels) / 255.0, 0.0, 1.0)
        mode = "height"

    if mode == "track_length":
        values = table.property("observations")
        if values is None:
            values = np.ones(len(table.values), dtype=np.float64)
        return _scalar_colors(values, "default")
    if mode == "reprojection_error":
        values = table.property("mean_reprojection_error")
        if values is None:
            values = np.zeros(len(table.values), dtype=np.float64)
        return _scalar_colors(values, "error")
    if mode == "triangulation_angle":
        values = table.property("max_triangulation_angle")
        if values is None:
            values = np.zeros(len(table.values), dtype=np.float64)
        return _scalar_colors(values, "default")
    if mode == "height":
        return _scalar_colors(table.points[:, 2], "default")
    raise ValueError(f"Unknown color mode: {mode}")


def load_camera_parameters(path: str | Path) -> tuple[np.ndarray, dict[str, Pose]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    poses = {}
    for name, matrix in data["extrinsics"].items():
        transform = np.asarray(matrix, dtype=np.float64)
        poses[name] = Pose(transform[:3, :3], transform[:3, 3])
    return intrinsics, poses


def camera_geometry_arrays(
    poses: dict[str, Pose],
    intrinsics: np.ndarray,
    scene_scale: float,
    frustum_scale: float | None = None,
    include_frustums: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = [pose for _, pose in sorted(poses.items())]
    centers = np.asarray([pose.camera_center for pose in ordered], dtype=np.float64)
    if len(centers) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )

    vertices: list[np.ndarray] = []
    lines: list[tuple[int, int]] = []
    colors: list[tuple[float, float, float]] = []
    for index, pose in enumerate(ordered):
        if index + 1 < len(ordered):
            lines.append((index, index + 1))
            colors.append((0.9, 0.15, 0.1))
    vertices.extend(centers)

    if not include_frustums:
        return (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(lines, dtype=np.int32),
            np.asarray(colors, dtype=np.float64),
        )

    scale = frustum_scale or max(scene_scale, 1e-6) * 0.035
    fx = max(float(intrinsics[0, 0]), 1e-6)
    fy = max(float(intrinsics[1, 1]), 1e-6)
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    width = max(2.0 * cx, 1.0)
    height = max(2.0 * cy, 1.0)
    pixel_corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float64,
    )

    for pose_index, pose in enumerate(ordered):
        center_index = pose_index
        corner_indices = []
        for pixel in pixel_corners:
            camera_point = np.array(
                [
                    (pixel[0] - cx) / fx * scale,
                    (pixel[1] - cy) / fy * scale,
                    scale,
                ],
                dtype=np.float64,
            )
            world_point = (
                pose.rotation.T @ (camera_point.reshape(3, 1) - pose.translation)
            ).ravel()
            corner_indices.append(len(vertices))
            vertices.append(world_point)
            lines.append((center_index, corner_indices[-1]))
            colors.append((0.95, 0.55, 0.05))
        for first, second in zip(corner_indices, corner_indices[1:] + corner_indices[:1]):
            lines.append((first, second))
            colors.append((0.95, 0.55, 0.05))

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(lines, dtype=np.int32),
        np.asarray(colors, dtype=np.float64),
    )


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is not installed. Install it with "
            "`python -m pip install -e .[open3d]` or `python -m pip install open3d`."
        ) from exc
    return o3d


def open_viewer(
    result_dir: Path,
    point_cloud: Path | None = None,
    color_mode: str = "rgb",
    point_size: float = 3.0,
    show_cameras: bool = True,
    show_frustums: bool = True,
    frustum_scale: float | None = None,
) -> None:
    o3d = _require_open3d()
    cloud_path = choose_point_cloud_path(result_dir, point_cloud)
    table = load_ascii_ply_table(cloud_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(table.points)
    pcd.colors = o3d.utility.Vector3dVector(colors_for_mode(table, color_mode))
    geometries = [pcd]

    finite_points = table.points[np.isfinite(table.points).all(axis=1)]
    scene_scale = 1.0
    if len(finite_points) > 0:
        scene_scale = float(np.linalg.norm(np.ptp(finite_points, axis=0)))

    camera_path = result_dir / "estimated_camera_parameters.json"
    if show_cameras and camera_path.is_file():
        intrinsics, poses = load_camera_parameters(camera_path)
        vertices, lines, colors = camera_geometry_arrays(
            poses,
            intrinsics,
            scene_scale,
            frustum_scale,
            include_frustums=show_frustums,
        )
        if len(vertices) and len(lines):
            camera_lines = o3d.geometry.LineSet()
            camera_lines.points = o3d.utility.Vector3dVector(vertices)
            camera_lines.lines = o3d.utility.Vector2iVector(lines)
            camera_lines.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(camera_lines)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"SfM Reconstruction - {result_dir.name}")
    for geometry in geometries:
        vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.asarray([0.02, 0.02, 0.025])
    vis.run()
    vis.destroy_window()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open an Open3D viewer for an SfM reconstruction."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--point-cloud", type=Path)
    parser.add_argument(
        "--color-mode",
        choices=(
            "rgb",
            "track_length",
            "reprojection_error",
            "triangulation_angle",
            "height",
        ),
        default="rgb",
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--frustum-scale", type=float)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-frustums", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    open_viewer(
        args.result_dir.resolve(),
        point_cloud=args.point_cloud,
        color_mode=args.color_mode,
        point_size=args.point_size,
        show_cameras=not args.no_cameras,
        show_frustums=not args.no_frustums,
        frustum_scale=args.frustum_scale,
    )


if __name__ == "__main__":
    main()
