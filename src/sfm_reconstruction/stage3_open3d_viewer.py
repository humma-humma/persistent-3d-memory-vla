from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .open3d_viewer import _require_open3d, load_ascii_ply_table
from .stage3 import (
    Stage3Dataset,
    TrajectoryPose,
    _transform_from_trajectory_pose,
    load_stage3_dataset,
    load_trajectory,
)
from .stage3_visualize import NamedTrajectory, aligned_trajectory_points


@dataclass(frozen=True)
class TrajectoryGeometry:
    label: str
    points: np.ndarray
    color: tuple[float, float, float]


@dataclass(frozen=True)
class SceneCloudGeometry:
    label: str
    points: np.ndarray
    colors: np.ndarray | None
    uniform_color: tuple[float, float, float] | None = None


_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.1, 0.45, 0.95),
    (0.95, 0.45, 0.05),
    (0.1, 0.7, 0.2),
    (0.8, 0.25, 0.85),
    (0.05, 0.7, 0.75),
)


def _parse_color(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B")
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise argparse.ArgumentTypeError("color components must be in [0, 1]")
    return (parts[0], parts[1], parts[2])


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("label must be non-empty")
    return label, Path(path)


def _points_from_poses(poses: list[TrajectoryPose]) -> np.ndarray:
    return np.asarray([pose.translation for pose in poses], dtype=np.float64)


def smooth_points(points: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered moving average to display-only trajectory points."""
    points = np.asarray(points, dtype=np.float64)
    if window <= 1 or len(points) < 3:
        return points.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    smoothed = points.copy()
    for index in range(1, len(points) - 1):
        start = max(0, index - radius)
        stop = min(len(points), index + radius + 1)
        smoothed[index] = points[start:stop].mean(axis=0)
    return smoothed


def _pose_by_timestamp(poses: list[TrajectoryPose]) -> dict[str, TrajectoryPose]:
    return {f"{pose.timestamp:.9f}": pose for pose in poses}


def build_stage3_scene_points(
    dataset: Stage3Dataset,
    trajectory: list[TrajectoryPose],
    *,
    frame_stride: int = 40,
    pixel_stride: int = 8,
    depth_scale: float = 1000.0,
    min_depth: float = 0.2,
    max_depth: float = 5.0,
    max_points: int = 400_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project sampled RGB-D frames into a world-coordinate point cloud."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be at least 1")
    if depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive")

    pose_lookup = _pose_by_timestamp(trajectory)
    fx = float(dataset.intrinsics[0, 0])
    fy = float(dataset.intrinsics[1, 1])
    cx = float(dataset.intrinsics[0, 2])
    cy = float(dataset.intrinsics[1, 2])

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for frame in dataset.frames[::frame_stride]:
        pose = pose_lookup.get(f"{frame.timestamp:.9f}")
        if pose is None or frame.depth_path is None:
            continue
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if depth is None or rgb is None:
            continue
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(
                rgb,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        ys = np.arange(0, depth.shape[0], pixel_stride, dtype=np.int32)
        xs = np.arange(0, depth.shape[1], pixel_stride, dtype=np.int32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        raw_depth = depth[grid_y, grid_x].astype(np.float64)
        z = raw_depth / depth_scale
        valid = (raw_depth > 0) & (z >= min_depth) & (z <= max_depth)
        if not np.any(valid):
            continue

        u = grid_x[valid].astype(np.float64)
        v = grid_y[valid].astype(np.float64)
        z = z[valid]
        points_camera = np.column_stack(
            (
                (u - cx) * z / fx,
                (v - cy) * z / fy,
                z,
            )
        )
        camera_to_world = _transform_from_trajectory_pose(pose)
        points_world = (
            camera_to_world[:3, :3] @ points_camera.T
        ).T + camera_to_world[:3, 3]
        colors = rgb[grid_y[valid], grid_x[valid], ::-1].astype(np.float64) / 255.0
        all_points.append(points_world)
        all_colors.append(colors)

    if not all_points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def write_stage3_scene_ply(
    path: str | Path, points: np.ndarray, colors: np.ndarray
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors_u8):
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def load_scene_cloud_ply(
    path: str | Path,
    *,
    label: str,
    uniform_color: tuple[float, float, float] | None = None,
) -> SceneCloudGeometry:
    table = load_ascii_ply_table(path)
    colors = None
    if uniform_color is None:
        channels = [table.property(name) for name in ("red", "green", "blue")]
        if all(channel is not None for channel in channels):
            colors = np.clip(np.column_stack(channels) / 255.0, 0.0, 1.0)
    return SceneCloudGeometry(
        label=label,
        points=table.points,
        colors=colors,
        uniform_color=uniform_color,
    )


def stage3_trajectory_geometries(
    ground_truth: list[TrajectoryPose] | None,
    estimates: list[NamedTrajectory],
    *,
    align_to_ground_truth: bool = True,
    smoothing_window: int = 1,
) -> list[TrajectoryGeometry]:
    geometries: list[TrajectoryGeometry] = []
    if ground_truth is not None:
        geometries.append(
            TrajectoryGeometry(
                label="Ground truth",
                points=smooth_points(_points_from_poses(ground_truth), smoothing_window),
                color=(0.0, 0.0, 0.0),
            )
        )

    for index, estimate in enumerate(estimates):
        points = _points_from_poses(estimate.poses)
        if align_to_ground_truth and ground_truth is not None:
            _, _, points = aligned_trajectory_points(ground_truth, estimate.poses)
        points = smooth_points(points, smoothing_window)
        geometries.append(
            TrajectoryGeometry(
                label=estimate.label,
                points=points,
                color=_COLORS[index % len(_COLORS)],
            )
        )
    return geometries


def _line_indices(point_count: int) -> np.ndarray:
    if point_count < 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack(
        (
            np.arange(0, point_count - 1, dtype=np.int32),
            np.arange(1, point_count, dtype=np.int32),
        )
    )


def _add_trajectory_line_set(o3d, geometries, trajectory: TrajectoryGeometry) -> None:
    lines = _line_indices(len(trajectory.points))
    if len(lines) == 0:
        return
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(trajectory.points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(trajectory.color, dtype=np.float64), (len(lines), 1))
    )
    geometries.append(line_set)

    markers = o3d.geometry.PointCloud()
    markers.points = o3d.utility.Vector3dVector(
        np.vstack((trajectory.points[0], trajectory.points[-1]))
    )
    markers.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(trajectory.color, dtype=np.float64), (2, 1))
    )
    geometries.append(markers)


def open_stage3_viewer(
    ground_truth: list[TrajectoryPose] | None,
    estimates: list[NamedTrajectory],
    *,
    scene_points: np.ndarray | None = None,
    scene_colors: np.ndarray | None = None,
    scene_uniform_color: tuple[float, float, float] | None = None,
    extra_scene_clouds: list[SceneCloudGeometry] | None = None,
    align_to_ground_truth: bool = True,
    point_size: float = 8.0,
    trajectory_smoothing_window: int = 1,
    scene_voxel_size: float | None = None,
    remove_outliers: bool = False,
    outlier_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    scene_as_mesh: bool = False,
    mesh_poisson_depth: int = 8,
    export_mesh: Path | None = None,
) -> None:
    o3d = _require_open3d()
    trajectory_geometries = stage3_trajectory_geometries(
        ground_truth,
        estimates,
        align_to_ground_truth=align_to_ground_truth,
        smoothing_window=trajectory_smoothing_window,
    )
    open3d_geometries = []
    scene_clouds: list[SceneCloudGeometry] = []
    if scene_points is not None and len(scene_points) > 0:
        scene_clouds.append(
            SceneCloudGeometry(
                label="Scene",
                points=scene_points,
                colors=scene_colors,
                uniform_color=scene_uniform_color,
            )
        )
    if extra_scene_clouds:
        scene_clouds.extend(extra_scene_clouds)

    for scene_cloud in scene_clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(scene_cloud.points)
        if scene_cloud.uniform_color is not None:
            pcd.paint_uniform_color(scene_cloud.uniform_color)
        elif scene_cloud.colors is not None and len(scene_cloud.colors) == len(scene_cloud.points):
            pcd.colors = o3d.utility.Vector3dVector(scene_cloud.colors)
        if scene_voxel_size is not None and scene_voxel_size > 0.0:
            pcd = pcd.voxel_down_sample(scene_voxel_size)
        if remove_outliers and len(pcd.points) > outlier_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=outlier_neighbors,
                std_ratio=outlier_std_ratio,
            )
        if scene_as_mesh and len(pcd.points) >= 100:
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(30)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd,
                depth=mesh_poisson_depth,
            )
            density_values = np.asarray(densities)
            if len(density_values):
                keep = density_values >= np.quantile(density_values, 0.05)
                mesh.remove_vertices_by_mask(~keep)
            mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color((0.72, 0.72, 0.70))
            if export_mesh is not None:
                export_mesh.parent.mkdir(parents=True, exist_ok=True)
                o3d.io.write_triangle_mesh(str(export_mesh), mesh)
                print(f"Saved Stage 3 scene mesh: {export_mesh.resolve()}")
            open3d_geometries.append(mesh)
        else:
            open3d_geometries.append(pcd)

    for trajectory in trajectory_geometries:
        _add_trajectory_line_set(o3d, open3d_geometries, trajectory)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    open3d_geometries.append(coordinate_frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Stage 3 Trajectory Viewer")
    for geometry in open3d_geometries:
        vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.line_width = 3.0
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.run()
    vis.destroy_window()


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
        description="Open an interactive Open3D viewer for Stage 3 trajectories."
    )
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        type=_parse_trajectory_arg,
        help="Estimated trajectory as LABEL=PATH. May be repeated.",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Show raw estimated coordinates instead of aligning to ground truth.",
    )
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--trajectory-smoothing-window", type=int, default=1)
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--scene-trajectory",
        type=Path,
        help="Trajectory used to back-project RGB-D frames into a scene cloud.",
    )
    parser.add_argument("--scene-frame-stride", type=int, default=40)
    parser.add_argument("--scene-pixel-stride", type=int, default=8)
    parser.add_argument("--scene-max-points", type=int, default=400_000)
    parser.add_argument("--scene-depth-scale", type=float, default=1000.0)
    parser.add_argument("--scene-min-depth", type=float, default=0.2)
    parser.add_argument("--scene-max-depth", type=float, default=5.0)
    parser.add_argument("--scene-uniform-color", type=_parse_color)
    parser.add_argument(
        "--extra-scene-ply",
        action="append",
        type=_parse_labeled_path,
        help="Extra scene cloud as LABEL=PATH. May be repeated.",
    )
    parser.add_argument(
        "--extra-scene-color",
        action="append",
        type=_parse_color,
        help="Uniform color for matching --extra-scene-ply, as R,G,B.",
    )
    parser.add_argument("--scene-voxel-size", type=float)
    parser.add_argument("--scene-remove-outliers", action="store_true")
    parser.add_argument("--scene-outlier-neighbors", type=int, default=20)
    parser.add_argument("--scene-outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--scene-as-mesh", action="store_true")
    parser.add_argument("--mesh-poisson-depth", type=int, default=8)
    parser.add_argument("--export-scene-mesh", type=Path)
    parser.add_argument("--export-scene-ply", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ground_truth = load_trajectory(args.ground_truth) if args.ground_truth else None
    scene_points = None
    scene_colors = None
    extra_scene_clouds: list[SceneCloudGeometry] = []
    if args.dataset is not None:
        scene_trajectory = (
            load_trajectory(args.scene_trajectory)
            if args.scene_trajectory is not None
            else args.trajectory[-1].poses
        )
        scene_points, scene_colors = build_stage3_scene_points(
            load_stage3_dataset(args.dataset),
            scene_trajectory,
            frame_stride=args.scene_frame_stride,
            pixel_stride=args.scene_pixel_stride,
            depth_scale=args.scene_depth_scale,
            min_depth=args.scene_min_depth,
            max_depth=args.scene_max_depth,
            max_points=args.scene_max_points,
        )
        if args.export_scene_ply is not None:
            write_stage3_scene_ply(args.export_scene_ply, scene_points, scene_colors)
            print(f"Saved Stage 3 scene cloud: {args.export_scene_ply.resolve()}")
    if args.extra_scene_ply:
        extra_colors = args.extra_scene_color or []
        for index, (label, path) in enumerate(args.extra_scene_ply):
            color = extra_colors[index] if index < len(extra_colors) else None
            extra_scene_clouds.append(
                load_scene_cloud_ply(path, label=label, uniform_color=color)
            )
    if args.no_view:
        return
    open_stage3_viewer(
        ground_truth,
        args.trajectory,
        scene_points=scene_points,
        scene_colors=scene_colors,
        scene_uniform_color=args.scene_uniform_color,
        extra_scene_clouds=extra_scene_clouds,
        align_to_ground_truth=not args.no_align,
        point_size=args.point_size,
        trajectory_smoothing_window=args.trajectory_smoothing_window,
        scene_voxel_size=args.scene_voxel_size,
        remove_outliers=args.scene_remove_outliers,
        outlier_neighbors=args.scene_outlier_neighbors,
        outlier_std_ratio=args.scene_outlier_std_ratio,
        scene_as_mesh=args.scene_as_mesh,
        mesh_poisson_depth=args.mesh_poisson_depth,
        export_mesh=args.export_scene_mesh,
    )


if __name__ == "__main__":
    main()
