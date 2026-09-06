from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Stage1Dataset
from .evaluation import evaluate_poses
from .geometry import reprojection_errors, triangulation_angles
from .reconstruction import ReconstructionResult


def write_camera_parameters(
    path: str | Path,
    dataset: Stage1Dataset,
    result: ReconstructionResult,
) -> None:
    extrinsics = {
        dataset.image_names[image_id]: result.poses[image_id].matrix().tolist()
        for image_id in sorted(result.poses)
    }
    payload = {
        "intrinsics": dataset.intrinsics.tolist(),
        "extrinsics": extrinsics,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_point_cloud(path: str | Path, result: ReconstructionResult) -> None:
    points = np.asarray(
        [result.points[track_id] for track_id in sorted(result.points)],
        dtype=np.float64,
    ).reshape(-1, 3)
    with Path(path).open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            file.write(f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g}\n")


def _load_rgb_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] < 3:
        return None
    return image[:, :, :3][:, :, ::-1]


def _sample_track_color(
    dataset: Stage1Dataset,
    result: ReconstructionResult,
    track_id: int,
    image_cache: dict[int, np.ndarray | None],
) -> tuple[int, int, int]:
    samples = []
    for image_id, observed in result.tracks[track_id].observations.items():
        if image_id not in dataset.image_paths:
            continue
        if image_id not in image_cache:
            image_cache[image_id] = _load_rgb_image(dataset.image_paths[image_id])
        image = image_cache[image_id]
        if image is None:
            continue
        x, y = np.rint(observed).astype(int)
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            samples.append(image[y, x, :3].astype(np.float64))
    if not samples:
        return (255, 255, 255)
    color = np.rint(np.mean(samples, axis=0)).clip(0, 255).astype(int)
    return (int(color[0]), int(color[1]), int(color[2]))


def _track_quality_attributes(
    dataset: Stage1Dataset,
    result: ReconstructionResult,
    track_id: int,
) -> tuple[int, int, float, float, float]:
    track = result.tracks[track_id]
    registered_ids = sorted(set(track.observations) & set(result.poses))
    errors = [
        reprojection_errors(
            result.points[track_id].reshape(1, 3),
            track.observations[image_id].reshape(1, 2),
            result.poses[image_id],
            dataset.intrinsics,
        )[0]
        for image_id in registered_ids
    ]
    maximum_angle = max(
        (
            triangulation_angles(
                result.points[track_id].reshape(1, 3),
                result.poses[first_id],
                result.poses[second_id],
            )[0]
            for index, first_id in enumerate(registered_ids)
            for second_id in registered_ids[index + 1 :]
        ),
        default=0.0,
    )
    if errors:
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
    else:
        mean_error = -1.0
        max_error = -1.0
    return (
        len(track.observations),
        len(registered_ids),
        mean_error,
        max_error,
        float(maximum_angle),
    )


def write_rich_point_cloud(
    path: str | Path,
    dataset: Stage1Dataset,
    result: ReconstructionResult,
    track_id_map: dict[int, int] | None = None,
) -> None:
    track_ids = sorted(result.points)
    image_cache: dict[int, np.ndarray | None] = {}
    with Path(path).open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(track_ids)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property int track_id\n")
        file.write("property int observations\n")
        file.write("property int registered_observations\n")
        file.write("property float mean_reprojection_error\n")
        file.write("property float max_reprojection_error\n")
        file.write("property float max_triangulation_angle\n")
        file.write("end_header\n")
        for track_id in track_ids:
            point = result.points[track_id]
            red, green, blue = _sample_track_color(
                dataset, result, track_id, image_cache
            )
            (
                observations,
                registered_observations,
                mean_error,
                max_error,
                max_angle,
            ) = _track_quality_attributes(dataset, result, track_id)
            file.write(
                f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} "
                f"{red:d} {green:d} {blue:d} "
                f"{(track_id_map or {}).get(track_id, track_id):d} "
                f"{observations:d} {registered_observations:d} "
                f"{mean_error:.10g} {max_error:.10g} {max_angle:.10g}\n"
            )


def write_summary(
    path: str | Path,
    dataset: Stage1Dataset,
    result: ReconstructionResult,
) -> None:
    pose_metrics = evaluate_poses(result.poses, dataset.ground_truth_extrinsics)
    payload = {
        "dataset": str(dataset.root),
        "images": len(dataset.image_paths),
        "registered_cameras": len(result.poses),
        "points": len(result.points),
        "tracks": len(result.tracks),
        "initial_pair": list(result.initial_pair),
        "skipped_track_conflicts": result.skipped_track_conflicts,
        "unregistered_images": sorted(set(dataset.image_ids) - set(result.poses)),
        "pose_evaluation": (
            pose_metrics.to_dict() if pose_metrics is not None else None
        ),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
