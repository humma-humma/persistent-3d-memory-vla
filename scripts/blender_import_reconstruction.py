"""Build a Blender scene from the SfM PLY and camera JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trim-percentile", type=float, default=1.0)
    parser.add_argument("--point-size", type=float)
    return parser.parse_args(argv)


def load_ascii_ply(path: Path) -> list[tuple[float, float, float]]:
    with path.open("r", encoding="ascii") as file:
        if file.readline().strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = None
        is_ascii = False
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"{path} has no end_header marker")
            fields = line.strip().split()
            if fields[:2] == ["format", "ascii"]:
                is_ascii = True
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields == ["end_header"]:
                break
        if not is_ascii:
            raise ValueError(f"{path} must use ASCII PLY format")
        if vertex_count is None:
            raise ValueError(f"{path} has no vertex count")

        points = []
        for _ in range(vertex_count):
            values = file.readline().split()
            if len(values) < 3:
                raise ValueError(f"{path} contains an incomplete vertex")
            point = tuple(float(value) for value in values[:3])
            if all(math.isfinite(value) for value in point):
                points.append(point)
    return points


def percentile(values: list[float], amount: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * amount / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def trim_outliers(
    points: list[tuple[float, float, float]],
    amount: float,
) -> list[tuple[float, float, float]]:
    if amount <= 0.0:
        return points
    if amount >= 50.0:
        raise ValueError("--trim-percentile must be below 50")
    lower = [
        percentile([point[axis] for point in points], amount)
        for axis in range(3)
    ]
    upper = [
        percentile([point[axis] for point in points], 100.0 - amount)
        for axis in range(3)
    ]
    return [
        point
        for point in points
        if all(lower[axis] <= point[axis] <= upper[axis] for axis in range(3))
    ]


def load_cameras(path: Path) -> tuple[list[tuple[str, Matrix]], list[list[float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = [
        (name, Matrix(matrix))
        for name, matrix in sorted(data["extrinsics"].items())
    ]
    return cameras, data["intrinsics"]


def make_material(
    name: str,
    color: tuple[float, float, float, float],
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    if emission_strength > 0.0:
        material.use_nodes = True
        principled = material.node_tree.nodes.get("Principled BSDF")
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Emission Color"].default_value = color
        principled.inputs["Emission Strength"].default_value = emission_strength
    return material


def add_point_markers(
    points: list[tuple[float, float, float]],
    size: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    offsets = (
        (size, size, size),
        (-size, -size, size),
        (-size, size, -size),
        (size, -size, -size),
    )
    local_faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    vertices = []
    faces = []
    for point in points:
        base = len(vertices)
        vertices.extend(
            (
                point[0] + offset[0],
                point[1] + offset[1],
                point[2] + offset[2],
            )
            for offset in offsets
        )
        faces.extend(tuple(base + index for index in face) for face in local_faces)

    mesh = bpy.data.meshes.new("SparsePoints")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    point_object = bpy.data.objects.new("Sparse Points", mesh)
    bpy.context.collection.objects.link(point_object)
    point_object.data.materials.append(material)
    return point_object


def add_camera_trajectory(
    centers: list[Vector],
    radius: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    curve = bpy.data.curves.new("CameraTrajectory", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(centers) - 1)
    for point, center in zip(spline.points, centers):
        point.co = (*center, 1.0)
    trajectory = bpy.data.objects.new("Camera Trajectory", curve)
    bpy.context.collection.objects.link(trajectory)
    trajectory.data.materials.append(material)
    return trajectory


def add_reconstructed_cameras(
    cameras: list[tuple[str, Matrix]],
    intrinsics: list[list[float]],
    display_size: float,
) -> list[Vector]:
    collection = bpy.data.collections.new("Reconstructed Cameras")
    bpy.context.scene.collection.children.link(collection)
    cv_to_blender = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    sensor_width = 36.0
    image_width = max(2.0 * float(intrinsics[0][2]), 1.0)
    lens = float(intrinsics[0][0]) * sensor_width / image_width
    centers = []

    for name, world_to_camera in cameras:
        data = bpy.data.cameras.new(f"Camera {name}")
        data.display_size = display_size
        data.lens = lens
        data.sensor_width = sensor_width
        camera = bpy.data.objects.new(f"Camera {name}", data)
        collection.objects.link(camera)
        camera.matrix_world = world_to_camera.inverted() @ cv_to_blender
        camera.color = (1.0, 0.16, 0.03, 1.0)
        camera.show_name = False
        centers.append(camera.matrix_world.translation.copy())
    return centers


def scene_bounds(
    points: list[tuple[float, float, float]],
    centers: list[Vector],
) -> tuple[Vector, float]:
    values = points + [tuple(center) for center in centers]
    lower = Vector(tuple(min(value[axis] for value in values) for axis in range(3)))
    upper = Vector(tuple(max(value[axis] for value in values) for axis in range(3)))
    center = (lower + upper) * 0.5
    scale = max((upper - lower).length, 1e-3)
    return center, scale


def add_overview_camera(center: Vector, scale: float) -> None:
    data = bpy.data.cameras.new("Overview")
    data.lens = 52.0
    camera = bpy.data.objects.new("Overview", data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((0.9, -1.25, 0.75)) * scale
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def configure_scene(center: Vector, scale: float) -> None:
    scene = bpy.context.scene
    scene.world.color = (0.015, 0.015, 0.02)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene["sfm_center"] = list(center)
    scene["sfm_scale"] = scale


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output = args.output.resolve()
    points = load_ascii_ply(result_dir / "estimated_points.ply")
    points = trim_outliers(points, args.trim_percentile)
    cameras, intrinsics = load_cameras(
        result_dir / "estimated_camera_parameters.json"
    )
    provisional_centers = [
        world_to_camera.inverted().translation for _, world_to_camera in cameras
    ]
    center, scale = scene_bounds(points, provisional_centers)
    point_size = args.point_size or 0.0015 * scale

    reset_scene()
    point_material = make_material(
        "Reconstruction Points", (0.04, 0.45, 1.0, 1.0), emission_strength=0.2
    )
    trajectory_material = make_material(
        "Camera Trajectory", (1.0, 0.035, 0.02, 1.0), emission_strength=0.5
    )
    add_point_markers(points, point_size, point_material)
    centers = add_reconstructed_cameras(
        cameras, intrinsics, display_size=0.025 * scale
    )
    add_camera_trajectory(centers, 0.0007 * scale, trajectory_material)
    add_overview_camera(center, scale)
    configure_scene(center, scale)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(
        f"Saved {output} with {len(points)} visible points "
        f"and {len(cameras)} reconstructed cameras"
    )


if __name__ == "__main__":
    main()
