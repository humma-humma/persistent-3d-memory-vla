from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .geometry import triangulation_angles
from .models import Pose
from .open3d_viewer import choose_point_cloud_path, load_ascii_ply_table


@dataclass(frozen=True)
class DensePairCandidate:
    first_name: str
    second_name: str
    step: int
    sparse_support: int
    baseline: float
    median_triangulation_angle: float
    score: float


@dataclass(frozen=True)
class DensePairResult:
    first_name: str
    second_name: str
    points: np.ndarray
    colors: np.ndarray
    disparities: np.ndarray
    step: int = 1
    sparse_support: int = 0
    baseline: float = 0.0
    median_triangulation_angle: float = 0.0
    score: float = 0.0
    status: str = "used"


@dataclass(frozen=True)
class DenseFusionConfig:
    image_scale: float = 0.25
    pair_step: int = 1
    max_pair_step: int | None = None
    max_pairs: int | None = None
    min_pair_sparse_support: int = 20
    min_pair_triangulation_angle: float = 0.25
    min_dense_points_per_pair: int = 25
    sample_stride: int = 3
    block_size: int = 5
    max_num_disparities: int = 256
    bounds_percentile: float = 1.0
    bounds_margin_factor: float = 0.15
    max_sparse_distance: float | None = None
    max_output_points: int | None = 500_000
    spatial_bins: int = 4


def load_camera_parameters(path: str | Path) -> tuple[np.ndarray, dict[str, Pose]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    poses = {}
    for name, matrix in data["extrinsics"].items():
        transform = np.asarray(matrix, dtype=np.float64)
        poses[name] = Pose(transform[:3, :3], transform[:3, 3])
    return intrinsics, poses


def select_neighbor_pairs(
    names: list[str],
    pair_step: int = 1,
    max_pairs: int | None = None,
    max_pair_step: int | None = None,
) -> list[tuple[str, str]]:
    if pair_step < 1:
        raise ValueError("pair_step must be at least 1")
    if max_pair_step is None:
        max_pair_step = pair_step
    if max_pair_step < pair_step:
        raise ValueError("max_pair_step must be at least pair_step")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    pairs = []
    for step in range(pair_step, max_pair_step + 1):
        pairs.extend(
            (names[index], names[index + step])
            for index in range(0, len(names) - step)
        )
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
    return [pairs[index] for index in indices]


def _project_visibility(
    points: np.ndarray,
    pose: Pose,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    camera_points = (pose.rotation @ points.T + pose.translation).T
    visible = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-8)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if visible.any():
        projected = (intrinsics @ camera_points[visible].T).T
        pixels[visible] = projected[:, :2] / projected[:, 2:3]
    return (
        visible
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )


def score_neighbor_pairs(
    names: list[str],
    pairs: list[tuple[str, str]],
    poses: dict[str, Pose],
    intrinsics: np.ndarray,
    sparse_points: np.ndarray,
    image_size: tuple[int, int],
) -> list[DensePairCandidate]:
    name_index = {name: index for index, name in enumerate(names)}
    visibility = {
        name: _project_visibility(sparse_points, poses[name], intrinsics, image_size)
        for name in names
    }
    candidates = []
    for first_name, second_name in pairs:
        common = visibility[first_name] & visibility[second_name]
        support = int(common.sum())
        baseline = float(
            np.linalg.norm(
                poses[first_name].camera_center - poses[second_name].camera_center
            )
        )
        if support >= 2 and baseline > 1e-8:
            angles = triangulation_angles(
                sparse_points[common],
                poses[first_name],
                poses[second_name],
            )
            finite_angles = angles[np.isfinite(angles)]
            median_angle = (
                float(np.median(finite_angles)) if len(finite_angles) else 0.0
            )
        else:
            median_angle = 0.0
        step = abs(name_index[second_name] - name_index[first_name])
        score = float(support * min(median_angle, 30.0))
        candidates.append(
            DensePairCandidate(
                first_name=first_name,
                second_name=second_name,
                step=step,
                sparse_support=support,
                baseline=baseline,
                median_triangulation_angle=median_angle,
                score=score,
            )
        )
    return candidates


def select_useful_pair_candidates(
    candidates: list[DensePairCandidate],
    min_sparse_support: int,
    min_triangulation_angle: float,
    max_pairs: int | None = None,
) -> list[DensePairCandidate]:
    if min_sparse_support < 0:
        raise ValueError("min_pair_sparse_support must be non-negative")
    if min_triangulation_angle < 0.0:
        raise ValueError("min_pair_triangulation_angle must be non-negative")
    useful = [
        candidate
        for candidate in candidates
        if candidate.sparse_support >= min_sparse_support
        and candidate.median_triangulation_angle >= min_triangulation_angle
    ]
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        useful = sorted(useful, key=lambda item: item.score, reverse=True)[:max_pairs]
    return sorted(useful, key=lambda item: (item.step, item.first_name, item.second_name))


def camera_to_camera_transform(first: Pose, second: Pose) -> tuple[np.ndarray, np.ndarray]:
    rotation = second.rotation @ first.rotation.T
    translation = second.translation - rotation @ first.translation
    return rotation, translation.reshape(3, 1)


def scale_intrinsics(intrinsics: np.ndarray, image_scale: float) -> np.ndarray:
    if image_scale <= 0.0 or image_scale > 1.0:
        raise ValueError("image_scale must be in the interval (0, 1]")
    scaled = np.asarray(intrinsics, dtype=np.float64).copy()
    scaled[0, 0] *= image_scale
    scaled[1, 1] *= image_scale
    scaled[0, 2] *= image_scale
    scaled[1, 2] *= image_scale
    return scaled


def sparse_bounds(
    sparse_points: np.ndarray,
    percentile: float,
    margin_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    if percentile < 0.0 or percentile >= 50.0:
        raise ValueError("bounds_percentile must be in [0, 50)")
    if margin_factor < 0.0:
        raise ValueError("bounds_margin_factor must be non-negative")
    points = np.asarray(sparse_points, dtype=np.float64).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("sparse cloud has no finite points")
    lower, upper = np.percentile(finite, [percentile, 100.0 - percentile], axis=0)
    extent = np.maximum(upper - lower, 1e-6)
    margin = margin_factor * extent
    return lower - margin, upper + margin


def _project_rectified(
    points: np.ndarray,
    pose: Pose,
    rectification: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    camera_points = (pose.rotation @ points.T + pose.translation).T
    rectified = (rectification @ camera_points.T).T
    homogeneous = np.column_stack((rectified, np.ones(len(rectified)))) @ projection.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def estimate_disparity_range(
    sparse_points: np.ndarray,
    first_pose: Pose,
    second_pose: Pose,
    rectification_first: np.ndarray,
    rectification_second: np.ndarray,
    projection_first: np.ndarray,
    projection_second: np.ndarray,
    image_size: tuple[int, int],
    max_num_disparities: int,
    margin: float = 16.0,
) -> tuple[int, int]:
    if max_num_disparities < 16:
        raise ValueError("max_num_disparities must be at least 16")
    points = np.asarray(sparse_points, dtype=np.float64).reshape(-1, 3)
    first_pixels = _project_rectified(
        points, first_pose, rectification_first, projection_first
    )
    second_pixels = _project_rectified(
        points, second_pose, rectification_second, projection_second
    )
    width, height = image_size
    in_view = (
        np.isfinite(first_pixels).all(axis=1)
        & np.isfinite(second_pixels).all(axis=1)
        & (first_pixels[:, 0] >= 0.0)
        & (first_pixels[:, 0] < width)
        & (first_pixels[:, 1] >= 0.0)
        & (first_pixels[:, 1] < height)
        & (second_pixels[:, 0] >= 0.0)
        & (second_pixels[:, 0] < width)
        & (second_pixels[:, 1] >= 0.0)
        & (second_pixels[:, 1] < height)
    )
    disparities = first_pixels[in_view, 0] - second_pixels[in_view, 0]
    disparities = disparities[np.isfinite(disparities)]
    if len(disparities) < 10:
        return _clamp_disparity_range(-64, min(128, max_num_disparities), width)

    lower, upper = np.percentile(disparities, [1, 99])
    minimum = int(np.floor((lower - margin) / 16.0) * 16)
    maximum = int(np.ceil((upper + margin) / 16.0) * 16)
    if maximum <= minimum:
        maximum = minimum + 16
    num_disparities = maximum - minimum
    if num_disparities > max_num_disparities:
        center = 0.5 * (lower + upper)
        minimum = int(np.floor((center - max_num_disparities / 2) / 16.0) * 16)
        num_disparities = int(np.ceil(max_num_disparities / 16.0) * 16)
    return _clamp_disparity_range(minimum, max(16, num_disparities), width)


def _clamp_disparity_range(
    min_disparity: int,
    num_disparities: int,
    image_width: int,
) -> tuple[int, int]:
    num_disparities = int(np.ceil(num_disparities / 16.0) * 16)
    if min_disparity + num_disparities < image_width:
        return min_disparity, max(16, num_disparities)

    available = image_width - min_disparity - 1
    num_disparities = int(np.floor(available / 16.0) * 16)
    if num_disparities >= 16:
        return min_disparity, num_disparities

    min_disparity = int(np.floor((image_width - 17) / 16.0) * 16)
    return min_disparity, 16


def _load_scaled_image(path: Path, image_scale: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image_scale != 1.0:
        width = max(1, int(round(image.shape[1] * image_scale)))
        height = max(1, int(round(image.shape[0] * image_scale)))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _load_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _create_matcher(
    min_disparity: int,
    num_disparities: int,
    block_size: int,
) -> cv2.StereoSGBM:
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    num_disparities = int(np.ceil(num_disparities / 16.0) * 16)
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity),
        numDisparities=max(16, num_disparities),
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _pair_result(
    candidate: DensePairCandidate,
    points: np.ndarray,
    colors: np.ndarray,
    disparities: np.ndarray,
    status: str = "used",
) -> DensePairResult:
    return DensePairResult(
        first_name=candidate.first_name,
        second_name=candidate.second_name,
        points=points,
        colors=colors,
        disparities=disparities,
        step=candidate.step,
        sparse_support=candidate.sparse_support,
        baseline=candidate.baseline,
        median_triangulation_angle=candidate.median_triangulation_angle,
        score=candidate.score,
        status=status,
    )


def fuse_pair(
    candidate: DensePairCandidate,
    first_path: Path,
    second_path: Path,
    intrinsics: np.ndarray,
    poses: dict[str, Pose],
    sparse_points: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    sparse_tree: cKDTree | None,
    config: DenseFusionConfig,
) -> DensePairResult:
    first_name = candidate.first_name
    second_name = candidate.second_name
    first_image = _load_scaled_image(first_path, config.image_scale)
    second_image = _load_scaled_image(second_path, config.image_scale)
    height, width = first_image.shape[:2]
    scaled_intrinsics = scale_intrinsics(intrinsics, config.image_scale)
    rotation, translation = camera_to_camera_transform(
        poses[first_name], poses[second_name]
    )
    rect_first, rect_second, proj_first, proj_second, q_matrix, _, _ = cv2.stereoRectify(
        scaled_intrinsics,
        None,
        scaled_intrinsics,
        None,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    min_disparity, num_disparities = estimate_disparity_range(
        sparse_points,
        poses[first_name],
        poses[second_name],
        rect_first,
        rect_second,
        proj_first,
        proj_second,
        (width, height),
        config.max_num_disparities,
    )
    map_first_x, map_first_y = cv2.initUndistortRectifyMap(
        scaled_intrinsics,
        None,
        rect_first,
        proj_first,
        (width, height),
        cv2.CV_32FC1,
    )
    map_second_x, map_second_y = cv2.initUndistortRectifyMap(
        scaled_intrinsics,
        None,
        rect_second,
        proj_second,
        (width, height),
        cv2.CV_32FC1,
    )
    rectified_first = cv2.remap(first_image, map_first_x, map_first_y, cv2.INTER_LINEAR)
    rectified_second = cv2.remap(second_image, map_second_x, map_second_y, cv2.INTER_LINEAR)
    gray_first = cv2.cvtColor(rectified_first, cv2.COLOR_BGR2GRAY)
    gray_second = cv2.cvtColor(rectified_second, cv2.COLOR_BGR2GRAY)
    matcher = _create_matcher(
        min_disparity,
        num_disparities,
        config.block_size,
    )
    disparity = matcher.compute(gray_first, gray_second).astype(np.float32) / 16.0
    rows, cols = np.mgrid[0:height:config.sample_stride, 0:width:config.sample_stride]
    sampled_disparity = disparity[rows, cols]
    valid = np.isfinite(sampled_disparity) & (sampled_disparity > min_disparity)
    if not valid.any():
        return _pair_result(
            candidate,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float64),
        )

    rectified_points = cv2.reprojectImageTo3D(disparity, q_matrix)
    sampled_rectified = rectified_points[rows, cols][valid]
    sampled_disparity = sampled_disparity[valid]
    finite = np.isfinite(sampled_rectified).all(axis=1) & (sampled_rectified[:, 2] > 0.0)
    sampled_rectified = sampled_rectified[finite]
    sampled_disparity = sampled_disparity[finite]
    if len(sampled_rectified) == 0:
        return _pair_result(
            candidate,
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.float64),
        )

    camera_points = (rect_first.T @ sampled_rectified.T).T
    pose = poses[first_name]
    world_points = (pose.rotation.T @ (camera_points.T - pose.translation)).T
    lower, upper = bounds
    in_bounds = np.all((world_points >= lower) & (world_points <= upper), axis=1)
    world_points = world_points[in_bounds]
    sampled_disparity = sampled_disparity[in_bounds]
    colors = rectified_first[rows, cols][valid][finite][in_bounds][:, ::-1]
    if sparse_tree is not None and config.max_sparse_distance is not None:
        distances, _ = sparse_tree.query(world_points, k=1)
        near_sparse = distances <= config.max_sparse_distance
        world_points = world_points[near_sparse]
        sampled_disparity = sampled_disparity[near_sparse]
        colors = colors[near_sparse]
    return _pair_result(
        candidate,
        world_points.astype(np.float64, copy=False),
        colors.astype(np.uint8, copy=False),
        sampled_disparity.astype(np.float64, copy=False),
    )


def write_dense_point_cloud(
    path: str | Path,
    pair_results: list[DensePairResult],
    max_output_points: int | None = None,
) -> int:
    points = []
    colors = []
    pair_indices = []
    image_ids = []
    disparities = []
    for pair_index, result in enumerate(pair_results):
        if len(result.points) == 0:
            continue
        first_id = int(Path(result.first_name).stem)
        second_id = int(Path(result.second_name).stem)
        points.append(result.points)
        colors.append(result.colors)
        pair_indices.append(np.full(len(result.points), pair_index, dtype=np.int32))
        image_ids.append(
            np.column_stack(
                (
                    np.full(len(result.points), first_id, dtype=np.int32),
                    np.full(len(result.points), second_id, dtype=np.int32),
                )
            )
        )
        disparities.append(result.disparities)

    if points:
        all_points = np.vstack(points)
        all_colors = np.vstack(colors)
        all_pair_indices = np.concatenate(pair_indices)
        all_image_ids = np.vstack(image_ids)
        all_disparities = np.concatenate(disparities)
    else:
        all_points = np.empty((0, 3), dtype=np.float64)
        all_colors = np.empty((0, 3), dtype=np.uint8)
        all_pair_indices = np.empty((0,), dtype=np.int32)
        all_image_ids = np.empty((0, 2), dtype=np.int32)
        all_disparities = np.empty((0,), dtype=np.float64)

    if max_output_points is not None and len(all_points) > max_output_points:
        indices = np.linspace(0, len(all_points) - 1, max_output_points, dtype=int)
        all_points = all_points[indices]
        all_colors = all_colors[indices]
        all_pair_indices = all_pair_indices[indices]
        all_image_ids = all_image_ids[indices]
        all_disparities = all_disparities[indices]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(all_points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property int pair_index\n")
        file.write("property int source_image_id\n")
        file.write("property int target_image_id\n")
        file.write("property float disparity\n")
        file.write("end_header\n")
        for point, color, pair_index, image_id, disparity in zip(
            all_points,
            all_colors,
            all_pair_indices,
            all_image_ids,
            all_disparities,
        ):
            file.write(
                f"{point[0]:.10g} {point[1]:.10g} {point[2]:.10g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(pair_index)} {int(image_id[0])} {int(image_id[1])} "
                f"{float(disparity):.10g}\n"
            )
    return len(all_points)


def _candidate_status(
    candidate: DensePairCandidate,
    active_keys: set[tuple[str, str]],
    result: DensePairResult | None,
    config: DenseFusionConfig,
) -> str:
    if (candidate.first_name, candidate.second_name) not in active_keys:
        if candidate.sparse_support < config.min_pair_sparse_support:
            return "low_sparse_support"
        return "low_triangulation_angle"
    if result is None:
        return "not_fused"
    if len(result.points) < config.min_dense_points_per_pair:
        return "too_few_dense_points"
    return "used"


def _pair_records(
    candidates: list[DensePairCandidate],
    active_candidates: list[DensePairCandidate],
    pair_results: list[DensePairResult],
    config: DenseFusionConfig,
) -> list[dict[str, object]]:
    active_keys = {
        (candidate.first_name, candidate.second_name)
        for candidate in active_candidates
    }
    result_by_pair = {
        (result.first_name, result.second_name): result
        for result in pair_results
    }
    records = []
    for candidate in candidates:
        result = result_by_pair.get((candidate.first_name, candidate.second_name))
        status = _candidate_status(candidate, active_keys, result, config)
        records.append(
            {
                "first": candidate.first_name,
                "second": candidate.second_name,
                "step": candidate.step,
                "sparse_support": candidate.sparse_support,
                "baseline": candidate.baseline,
                "median_triangulation_angle": candidate.median_triangulation_angle,
                "score": candidate.score,
                "dense_points": 0 if result is None else len(result.points),
                "status": status,
            }
        )
    return records


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_pair_heatmap(
    path: Path,
    names: list[str],
    pair_results: list[DensePairResult],
) -> None:
    index_by_name = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for result in pair_results:
        first = index_by_name[result.first_name]
        second = index_by_name[result.second_name]
        matrix[first, second] += len(result.points)
        matrix[second, first] += len(result.points)
    with path.open("w", encoding="ascii", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["camera", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *row.tolist()])


def _camera_contribution_rows(
    names: list[str],
    pair_results: list[DensePairResult],
) -> list[dict[str, object]]:
    source_counts = {name: 0 for name in names}
    target_counts = {name: 0 for name in names}
    pair_counts = {name: 0 for name in names}
    for result in pair_results:
        points = len(result.points)
        source_counts[result.first_name] += points
        target_counts[result.second_name] += points
        pair_counts[result.first_name] += 1
        pair_counts[result.second_name] += 1
    return [
        {
            "camera": name,
            "source_points": source_counts[name],
            "target_points": target_counts[name],
            "total_points": source_counts[name] + target_counts[name],
            "used_pairs": pair_counts[name],
        }
        for name in names
    ]


def _spatial_density_rows(
    pair_results: list[DensePairResult],
    bounds: tuple[np.ndarray, np.ndarray],
    bins: int,
) -> list[dict[str, object]]:
    if bins < 1:
        raise ValueError("spatial_bins must be positive")
    point_sets = [result.points for result in pair_results if len(result.points)]
    if not point_sets:
        return []
    points = np.vstack(point_sets)
    lower, upper = bounds
    extent = np.maximum(upper - lower, 1e-9)
    indices = np.floor((points - lower) / extent * bins).astype(int)
    indices = np.clip(indices, 0, bins - 1)
    unique, counts = np.unique(indices, axis=0, return_counts=True)
    rows = []
    bin_size = extent / bins
    for bin_index, count in zip(unique, counts):
        minimum = lower + bin_index * bin_size
        maximum = minimum + bin_size
        rows.append(
            {
                "x_bin": int(bin_index[0]),
                "y_bin": int(bin_index[1]),
                "z_bin": int(bin_index[2]),
                "count": int(count),
                "min_x": float(minimum[0]),
                "max_x": float(maximum[0]),
                "min_y": float(minimum[1]),
                "max_y": float(maximum[1]),
                "min_z": float(minimum[2]),
                "max_z": float(maximum[2]),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["count"]), row["x_bin"], row["y_bin"], row["z_bin"]))


def write_dense_diagnostics(
    output: Path,
    names: list[str],
    candidates: list[DensePairCandidate],
    active_candidates: list[DensePairCandidate],
    pair_results: list[DensePairResult],
    used_pair_results: list[DensePairResult],
    bounds: tuple[np.ndarray, np.ndarray],
    config: DenseFusionConfig,
) -> dict[str, str]:
    pair_path = output.with_suffix(".pairs.csv")
    heatmap_path = output.with_suffix(".pair_heatmap.csv")
    camera_path = output.with_suffix(".cameras.csv")
    spatial_path = output.with_suffix(".spatial_density.csv")

    pair_rows = _pair_records(candidates, active_candidates, pair_results, config)
    _write_csv(
        pair_path,
        pair_rows,
        [
            "first",
            "second",
            "step",
            "sparse_support",
            "baseline",
            "median_triangulation_angle",
            "score",
            "dense_points",
            "status",
        ],
    )
    _write_pair_heatmap(heatmap_path, names, used_pair_results)
    _write_csv(
        camera_path,
        _camera_contribution_rows(names, used_pair_results),
        ["camera", "source_points", "target_points", "total_points", "used_pairs"],
    )
    _write_csv(
        spatial_path,
        _spatial_density_rows(used_pair_results, bounds, config.spatial_bins),
        [
            "x_bin",
            "y_bin",
            "z_bin",
            "count",
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "min_z",
            "max_z",
        ],
    )
    return {
        "pairs": str(pair_path),
        "pair_heatmap": str(heatmap_path),
        "cameras": str(camera_path),
        "spatial_density": str(spatial_path),
    }


def run_dense_fusion(
    dataset_dir: Path,
    result_dir: Path,
    output: Path,
    config: DenseFusionConfig,
) -> dict[str, object]:
    intrinsics, poses = load_camera_parameters(
        result_dir / "estimated_camera_parameters.json"
    )
    names = sorted(poses)
    image_paths = {name: dataset_dir / "images" / name for name in names}
    missing = [str(path) for path in image_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images for dense fusion: {missing[:3]}")

    sparse_table = load_ascii_ply_table(choose_point_cloud_path(result_dir, None))
    sparse_points = sparse_table.points
    bounds = sparse_bounds(
        sparse_points,
        config.bounds_percentile,
        config.bounds_margin_factor,
    )
    if config.max_sparse_distance is not None and config.max_sparse_distance <= 0.0:
        raise ValueError("max_sparse_distance must be positive")
    if config.min_dense_points_per_pair < 0:
        raise ValueError("min_dense_points_per_pair must be non-negative")
    sparse_tree = (
        cKDTree(sparse_points)
        if config.max_sparse_distance is not None
        else None
    )
    max_pair_step = config.max_pair_step or config.pair_step
    pairs = select_neighbor_pairs(
        names,
        pair_step=config.pair_step,
        max_pair_step=max_pair_step,
    )
    image_size = _load_image_size(image_paths[names[0]])
    candidates = score_neighbor_pairs(
        names,
        pairs,
        poses,
        intrinsics,
        sparse_points,
        image_size,
    )
    active_candidates = select_useful_pair_candidates(
        candidates,
        config.min_pair_sparse_support,
        config.min_pair_triangulation_angle,
        config.max_pairs,
    )
    print(
        f"Selected {len(active_candidates)}/{len(candidates)} dense candidate "
        "pairs after sparse-support scoring."
    )
    pair_results = []
    for index, candidate in enumerate(active_candidates, start=1):
        result = fuse_pair(
            candidate,
            image_paths[candidate.first_name],
            image_paths[candidate.second_name],
            intrinsics,
            poses,
            sparse_points,
            bounds,
            sparse_tree,
            config,
        )
        pair_results.append(result)
        print(
            f"Dense pair {index}/{len(active_candidates)} "
            f"{candidate.first_name}->{candidate.second_name}: "
            f"{len(result.points)} points"
        )

    used_pair_results = [
        result
        for result in pair_results
        if len(result.points) >= config.min_dense_points_per_pair
    ]
    written = write_dense_point_cloud(
        output,
        used_pair_results,
        max_output_points=config.max_output_points,
    )
    diagnostics = write_dense_diagnostics(
        output,
        names,
        candidates,
        active_candidates,
        pair_results,
        used_pair_results,
        bounds,
        config,
    )
    pair_records = _pair_records(
        candidates,
        active_candidates,
        pair_results,
        config,
    )
    summary = {
        "dataset": str(dataset_dir),
        "result_dir": str(result_dir),
        "output": str(output),
        "pairs": pair_records,
        "candidate_pairs": len(candidates),
        "fused_pairs": len(pair_results),
        "used_pairs": len(used_pair_results),
        "skipped_near_zero_pairs": int(
            sum(
                1
                for result in pair_results
                if len(result.points) < config.min_dense_points_per_pair
            )
        ),
        "points_before_output_limit": int(
            sum(len(result.points) for result in used_pair_results)
        ),
        "points_written": int(written),
        "diagnostics": diagnostics,
        "config": {
            "image_scale": config.image_scale,
            "pair_step": config.pair_step,
            "max_pair_step": config.max_pair_step,
            "max_pairs": config.max_pairs,
            "min_pair_sparse_support": config.min_pair_sparse_support,
            "min_pair_triangulation_angle": config.min_pair_triangulation_angle,
            "min_dense_points_per_pair": config.min_dense_points_per_pair,
            "sample_stride": config.sample_stride,
            "block_size": config.block_size,
            "max_num_disparities": config.max_num_disparities,
            "bounds_percentile": config.bounds_percentile,
            "bounds_margin_factor": config.bounds_margin_factor,
            "max_sparse_distance": config.max_sparse_distance,
            "max_output_points": config.max_output_points,
            "spatial_bins": config.spatial_bins,
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse dense stereo points from SfM camera poses."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-scale", type=float, default=0.25)
    parser.add_argument("--pair-step", type=int, default=1)
    parser.add_argument("--max-pair-step", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--min-pair-sparse-support", type=int, default=20)
    parser.add_argument("--min-pair-triangulation-angle", type=float, default=0.25)
    parser.add_argument("--min-dense-points-per-pair", type=int, default=25)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--max-num-disparities", type=int, default=256)
    parser.add_argument("--bounds-percentile", type=float, default=1.0)
    parser.add_argument("--bounds-margin-factor", type=float, default=0.15)
    parser.add_argument("--max-sparse-distance", type=float)
    parser.add_argument("--max-output-points", type=int, default=500_000)
    parser.add_argument("--spatial-bins", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result_dir = args.result_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else result_dir / "dense_points.ply"
    )
    config = DenseFusionConfig(
        image_scale=args.image_scale,
        pair_step=args.pair_step,
        max_pair_step=args.max_pair_step,
        max_pairs=args.max_pairs,
        min_pair_sparse_support=args.min_pair_sparse_support,
        min_pair_triangulation_angle=args.min_pair_triangulation_angle,
        min_dense_points_per_pair=args.min_dense_points_per_pair,
        sample_stride=args.sample_stride,
        block_size=args.block_size,
        max_num_disparities=args.max_num_disparities,
        bounds_percentile=args.bounds_percentile,
        bounds_margin_factor=args.bounds_margin_factor,
        max_sparse_distance=args.max_sparse_distance,
        max_output_points=args.max_output_points,
        spatial_bins=args.spatial_bins,
    )
    summary = run_dense_fusion(args.dataset.resolve(), result_dir, output, config)
    print(f"Wrote {summary['points_written']} dense points: {output}")


if __name__ == "__main__":
    main()
