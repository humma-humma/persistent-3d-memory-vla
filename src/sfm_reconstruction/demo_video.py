from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .open3d_viewer import (
    PlyTable,
    _require_open3d,
    camera_geometry_arrays,
    choose_point_cloud_path,
    colors_for_mode,
    load_ascii_ply_table,
    load_camera_parameters,
)


def trim_point_table(table: PlyTable, percentile: float) -> PlyTable:
    if percentile <= 0.0 or len(table.values) == 0:
        return table
    if percentile >= 50.0:
        raise ValueError("--trim-percentile must be less than 50")

    points = table.points
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return table
    lower, upper = np.percentile(points[finite], [percentile, 100.0 - percentile], axis=0)
    in_bounds = finite & np.all((points >= lower) & (points <= upper), axis=1)
    if not in_bounds.any():
        return table
    return PlyTable(table.property_names, table.values[in_bounds])


def orbit_front(frame_index: int, frame_count: int, base_front: np.ndarray) -> np.ndarray:
    front = np.asarray(base_front, dtype=np.float64)
    norm = float(np.linalg.norm(front))
    if norm <= 0.0:
        raise ValueError("base_front must be nonzero")
    front = front / norm

    xy_norm = float(np.linalg.norm(front[:2]))
    if xy_norm <= 1e-9:
        angle = 2.0 * np.pi * frame_index / max(frame_count, 1)
        rotated = np.array([np.cos(angle), np.sin(angle), front[2]], dtype=np.float64)
    else:
        base_angle = np.arctan2(front[1], front[0])
        angle = base_angle + 2.0 * np.pi * frame_index / max(frame_count, 1)
        rotated = np.array(
            [xy_norm * np.cos(angle), xy_norm * np.sin(angle), front[2]],
            dtype=np.float64,
        )
    return rotated / np.linalg.norm(rotated)


def _build_geometries(
    result_dir: Path,
    point_cloud: Path | None,
    color_mode: str,
    trim_percentile: float,
    show_cameras: bool,
    show_frustums: bool,
    frustum_scale: float | None,
):
    o3d = _require_open3d()
    cloud_path = choose_point_cloud_path(result_dir, point_cloud)
    table = trim_point_table(load_ascii_ply_table(cloud_path), trim_percentile)

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

    return o3d, pcd, geometries


def render_orbit_video(
    result_dir: Path,
    output: Path,
    point_cloud: Path | None = None,
    poster_output: Path | None = None,
    color_mode: str = "rgb",
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    seconds: float = 8.0,
    point_size: float = 2.0,
    zoom: float = 0.78,
    trim_percentile: float = 1.0,
    show_cameras: bool = True,
    show_frustums: bool = True,
    frustum_scale: float | None = None,
    visible: bool = False,
) -> None:
    import cv2

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")

    frame_count = max(1, int(round(fps * seconds)))
    result_dir = result_dir.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if poster_output is not None:
        poster_output = poster_output.resolve()
        poster_output.parent.mkdir(parents=True, exist_ok=True)

    o3d, pcd, geometries = _build_geometries(
        result_dir,
        point_cloud,
        color_mode,
        trim_percentile,
        show_cameras,
        show_frustums,
        frustum_scale,
    )

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(
        window_name=f"SfM Demo - {result_dir.name}",
        width=width,
        height=height,
        visible=visible,
    ):
        raise RuntimeError("Open3D could not create a render window")
    try:
        for geometry in geometries:
            vis.add_geometry(geometry)

        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.asarray([0.02, 0.02, 0.025])

        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        view_control = vis.get_view_control()
        view_control.set_lookat(center)
        view_control.set_up(np.array([0.0, 0.0, 1.0]))
        view_control.set_zoom(zoom)

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output}")
        try:
            base_front = np.array([0.45, -0.78, -0.43], dtype=np.float64)
            for frame_index in range(frame_count):
                view_control.set_front(orbit_front(frame_index, frame_count, base_front))
                vis.poll_events()
                vis.update_renderer()
                image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                image_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
                image_bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
                if frame_index == 0 and poster_output is not None:
                    cv2.imwrite(str(poster_output), image_bgr)
                writer.write(image_bgr)
        finally:
            writer.release()
    finally:
        vis.destroy_window()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a short Open3D orbit video for an SfM reconstruction."
    )
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--point-cloud", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poster-output", type=Path)
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--zoom", type=float, default=0.78)
    parser.add_argument("--trim-percentile", type=float, default=1.0)
    parser.add_argument("--frustum-scale", type=float)
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-frustums", action="store_true")
    parser.add_argument("--visible", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    render_orbit_video(
        result_dir=args.result_dir,
        output=args.output,
        point_cloud=args.point_cloud,
        poster_output=args.poster_output,
        color_mode=args.color_mode,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
        point_size=args.point_size,
        zoom=args.zoom,
        trim_percentile=args.trim_percentile,
        show_cameras=not args.no_cameras,
        show_frustums=not args.no_frustums,
        frustum_scale=args.frustum_scale,
        visible=args.visible,
    )


if __name__ == "__main__":
    main()
