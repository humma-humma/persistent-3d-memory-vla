from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Stage1Dataset
from .geometry import estimate_relative_pose
from .learned_matching import create_superpoint_lightglue_matcher
from .place_recognition import create_dinov2_retriever, select_global_loop_candidates


@dataclass(frozen=True)
class MatchingConfig:
    max_features: int = 1500
    feature_mode: str = "sift"
    sift_contrast_threshold: float = 0.03
    sift_edge_threshold: float = 10.0
    pair_window: int = 3
    pair_source: str = "circular"
    ratio_threshold: float = 0.75
    essential_threshold: float = 1.0
    min_inliers: int = 15
    mask_apriltags: bool = False
    apriltag_padding: float = 0.12
    learned_device: str = "auto"
    learned_filter_threshold: float = 0.2
    learned_cycle_filter: bool = False
    learned_min_cycle_matches: int = 15
    learned_augment_supplied: bool = False
    wide_baseline: bool = False
    wide_pose_only: bool = False
    wide_retrieval_max_pairs: int = 100
    wide_min_frame_gap: int = 4
    wide_min_similarity: float = 0.7
    wide_min_inlier_ratio: float = 0.25
    wide_min_spatial_coverage: float = 0.25
    wide_min_cycle_matches: int = 15
    wide_max_pairs_per_image: int = 1
    wide_retrieval_device: str = "auto"
    wide_retrieval_batch_size: int = 16


@dataclass(frozen=True)
class FeatureSet:
    points: np.ndarray
    descriptors: np.ndarray
    descriptor_type: str
    source: str


@dataclass(frozen=True)
class MatchingSummary:
    candidate_pairs: int
    accepted_pairs: int
    correspondences: int


@dataclass
class _LearnedPairResult:
    pair: tuple[int, int]
    pair_type: str
    temporal_gap: int
    retrieval_similarity: float | None
    proposal_rank: int | None
    raw_matches: int
    essential_inliers: int
    inlier_ratio: float
    first_coverage: float
    second_coverage: float
    indices: np.ndarray
    correspondences: np.ndarray
    cycle_matches: int = 0
    kept_matches: int = 0
    accepted: bool = False
    reject_reason: str = ""


def circular_image_pairs(
    image_ids: list[int], pair_window: int
) -> list[tuple[int, int]]:
    if pair_window < 1:
        raise ValueError("pair_window must be positive")
    if len(image_ids) < 2:
        return []
    pair_window = min(pair_window, len(image_ids) - 1)
    pairs: set[tuple[int, int]] = set()
    for index, first_id in enumerate(image_ids):
        for offset in range(1, pair_window + 1):
            second_id = image_ids[(index + offset) % len(image_ids)]
            pairs.add(tuple(sorted((first_id, second_id))))
    return sorted(pairs)


def matching_image_pairs(
    dataset: Stage1Dataset, config: MatchingConfig
) -> list[tuple[int, int]]:
    if config.pair_source == "circular":
        return circular_image_pairs(dataset.image_ids, config.pair_window)
    if config.pair_source == "supplied":
        pairs = sorted(dataset.correspondence_paths)
        if not pairs:
            raise ValueError("pair_source='supplied' requires supplied correspondences")
        return pairs
    raise ValueError("pair_source must be 'circular' or 'supplied'")


def root_sift(descriptors: np.ndarray) -> np.ndarray:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    sums = np.sum(descriptors, axis=1, keepdims=True)
    normalized = descriptors / np.maximum(sums, 1e-12)
    return np.sqrt(normalized)


def match_descriptors(
    first: np.ndarray,
    second: np.ndarray,
    ratio_threshold: float,
    descriptor_type: str = "float",
) -> np.ndarray:
    """Custom mutual nearest-neighbor matching with Lowe's ratio test."""
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1")
    if descriptor_type == "binary":
        return _match_binary_descriptors(first, second, ratio_threshold)
    if descriptor_type != "float":
        raise ValueError(f"Unsupported descriptor type: {descriptor_type}")

    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if len(first) == 0 or len(second) < 2:
        return np.empty((0, 2), dtype=np.int32)

    first_norm = np.sum(first * first, axis=1, keepdims=True)
    second_norm = np.sum(second * second, axis=1)
    distances = first_norm + second_norm - 2.0 * (first @ second.T)
    np.maximum(distances, 0.0, out=distances)

    nearest_two = np.argpartition(distances, kth=1, axis=1)[:, :2]
    nearest_distances = np.take_along_axis(distances, nearest_two, axis=1)
    order = np.argsort(nearest_distances, axis=1)
    nearest_two = np.take_along_axis(nearest_two, order, axis=1)
    nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)
    reverse_best = np.argmin(distances, axis=0)

    accepted = (
        nearest_distances[:, 0]
        < (ratio_threshold * ratio_threshold) * nearest_distances[:, 1]
    )
    first_indices = np.flatnonzero(accepted)
    second_indices = nearest_two[accepted, 0]
    mutual = reverse_best[second_indices] == first_indices
    return np.column_stack(
        (first_indices[mutual], second_indices[mutual])
    ).astype(np.int32)


def _match_binary_descriptors(
    first: np.ndarray,
    second: np.ndarray,
    ratio_threshold: float,
) -> np.ndarray:
    first = np.asarray(first, dtype=np.uint8)
    second = np.asarray(second, dtype=np.uint8)
    if len(first) == 0 or len(second) < 2:
        return np.empty((0, 2), dtype=np.int32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    forward = matcher.knnMatch(first, second, k=2)
    reverse = matcher.knnMatch(second, first, k=1)
    reverse_best = {
        matches[0].queryIdx: matches[0].trainIdx
        for matches in reverse
        if matches
    }
    accepted = []
    for matches in forward:
        if len(matches) < 2:
            continue
        nearest, second_nearest = matches
        if nearest.distance >= ratio_threshold * second_nearest.distance:
            continue
        if reverse_best.get(nearest.trainIdx) == nearest.queryIdx:
            accepted.append((nearest.queryIdx, nearest.trainIdx))
    if not accepted:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(accepted, dtype=np.int32)


def _feature_cache_path(cache_dir: Path, image_id: int, method: str) -> Path:
    return cache_dir / f"{image_id}_{method}.npz"


def _feature_methods(feature_mode: str) -> tuple[str, ...]:
    if feature_mode == "sift":
        return ("sift",)
    if feature_mode == "akaze":
        return ("akaze",)
    if feature_mode in {"sift+akaze", "akaze+sift"}:
        return ("sift", "akaze")
    if feature_mode == "superpoint-lightglue":
        return ("superpoint-lightglue",)
    raise ValueError(
        "feature_mode must be one of: sift, akaze, sift+akaze, "
        "superpoint-lightglue"
    )


def _feature_mask(
    image: np.ndarray,
    mask_apriltags: bool,
    apriltag_padding: float,
) -> np.ndarray | None:
    if not mask_apriltags:
        return None
    if apriltag_padding < 0.0:
        raise ValueError("apriltag_padding must be non-negative")

    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    detector = cv2.aruco.ArucoDetector(
        dictionary, cv2.aruco.DetectorParameters()
    )
    corners, _, _ = detector.detectMarkers(image)
    for corner in corners:
        polygon = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        center = polygon.mean(axis=0, keepdims=True)
        polygon = center + (polygon - center) * (1.0 + apriltag_padding)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 0)
    return mask


def _load_cached_feature(
    cache_path: Path,
    method: str,
    config: MatchingConfig,
) -> FeatureSet | None:
    if not cache_path.is_file():
        return None
    cached = np.load(cache_path)
    cached_mask = bool(cached["mask_apriltags"]) if (
        "mask_apriltags" in cached
    ) else False
    cached_padding = float(cached["apriltag_padding"]) if (
        "apriltag_padding" in cached
    ) else 0.0
    cached_contrast = float(cached["sift_contrast_threshold"]) if (
        "sift_contrast_threshold" in cached
    ) else 0.03
    cached_edge = float(cached["sift_edge_threshold"]) if (
        "sift_edge_threshold" in cached
    ) else 10.0
    cached_method = str(cached["method"]) if "method" in cached else method
    if (
        cached_method == method
        and int(cached["max_features"]) == config.max_features
        and cached_mask == config.mask_apriltags
        and np.isclose(cached_padding, config.apriltag_padding)
        and np.isclose(cached_contrast, config.sift_contrast_threshold)
        and np.isclose(cached_edge, config.sift_edge_threshold)
    ):
        descriptor_type = str(cached["descriptor_type"]) if (
            "descriptor_type" in cached
        ) else "float"
        return FeatureSet(
            cached["points"],
            cached["descriptors"],
            descriptor_type,
            method,
        )
    return None


def _detect_feature_set(
    gray: np.ndarray,
    mask: np.ndarray | None,
    method: str,
    config: MatchingConfig,
) -> FeatureSet:
    if method == "sift":
        detector = cv2.SIFT_create(
            nfeatures=config.max_features,
            nOctaveLayers=4,
            contrastThreshold=config.sift_contrast_threshold,
            edgeThreshold=config.sift_edge_threshold,
            sigma=1.6,
        )
        keypoints, descriptors = detector.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) == 0:
            raise RuntimeError("No SIFT features detected")
        return FeatureSet(
            points=np.asarray(
                [keypoint.pt for keypoint in keypoints], dtype=np.float64
            ),
            descriptors=root_sift(descriptors),
            descriptor_type="float",
            source=method,
        )
    if method == "akaze":
        detector = cv2.AKAZE_create()
        keypoints, descriptors = detector.detectAndCompute(gray, mask)
        if descriptors is None or len(keypoints) == 0:
            raise RuntimeError("No AKAZE features detected")
        if config.max_features > 0 and len(keypoints) > config.max_features:
            responses = np.asarray(
                [keypoint.response for keypoint in keypoints], dtype=np.float64
            )
            selected = np.argsort(responses)[-config.max_features :]
            keypoints = [keypoints[index] for index in selected]
            descriptors = descriptors[selected]
        return FeatureSet(
            points=np.asarray(
                [keypoint.pt for keypoint in keypoints], dtype=np.float64
            ),
            descriptors=np.asarray(descriptors, dtype=np.uint8),
            descriptor_type="binary",
            source=method,
        )
    raise ValueError(f"Unsupported feature method: {method}")


def _extract_features(
    image_path: Path,
    image_id: int,
    cache_dir: Path,
    config: MatchingConfig,
    overwrite: bool,
) -> tuple[FeatureSet, ...]:
    cached_features = []
    methods = _feature_methods(config.feature_mode)
    for method in methods:
        cache_path = _feature_cache_path(cache_dir, image_id, method)
        if not overwrite:
            cached = _load_cached_feature(cache_path, method, config)
            if cached is not None:
                cached_features.append(cached)
                continue
        break
    else:
        return tuple(cached_features)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _feature_mask(image, config.mask_apriltags, config.apriltag_padding)
    features = []
    for method in methods:
        cache_path = _feature_cache_path(cache_dir, image_id, method)
        if not overwrite:
            cached = _load_cached_feature(cache_path, method, config)
            if cached is not None:
                features.append(cached)
                continue
        feature = _detect_feature_set(gray, mask, method, config)
        np.savez_compressed(
            cache_path,
            method=np.asarray(method),
            points=feature.points,
            descriptors=feature.descriptors,
            descriptor_type=np.asarray(feature.descriptor_type),
            max_features=np.asarray(config.max_features),
            mask_apriltags=np.asarray(config.mask_apriltags),
            apriltag_padding=np.asarray(config.apriltag_padding),
            sift_contrast_threshold=np.asarray(config.sift_contrast_threshold),
            sift_edge_threshold=np.asarray(config.sift_edge_threshold),
        )
        features.append(feature)
    return tuple(features)


def _match_feature_sets(
    first_features: tuple[FeatureSet, ...],
    second_features: tuple[FeatureSet, ...],
    ratio_threshold: float,
) -> np.ndarray:
    correspondences = []
    for first, second in zip(first_features, second_features):
        if first.source != second.source:
            raise ValueError("Feature sources must match")
        descriptor_matches = match_descriptors(
            first.descriptors,
            second.descriptors,
            ratio_threshold,
            first.descriptor_type,
        )
        if len(descriptor_matches) == 0:
            continue
        correspondences.append(
            np.hstack(
                (
                    first.points[descriptor_matches[:, 0]],
                    second.points[descriptor_matches[:, 1]],
                )
            )
        )
    if not correspondences:
        return np.empty((0, 4), dtype=np.float64)
    return np.vstack(correspondences)


def _spatial_coverage(
    points: np.ndarray, image_shape: tuple[int, int], grid_size: int = 4
) -> float:
    if len(points) == 0:
        return 0.0
    height, width = image_shape
    points = np.asarray(points, dtype=np.float64)
    columns = np.clip((points[:, 0] * grid_size / width).astype(int), 0, grid_size - 1)
    rows = np.clip((points[:, 1] * grid_size / height).astype(int), 0, grid_size - 1)
    return len(set(zip(rows.tolist(), columns.tolist()))) / float(grid_size**2)


def _directed_match_map(
    result: _LearnedPairResult, source_id: int, target_id: int
) -> dict[int, int]:
    first_id, second_id = result.pair
    if (source_id, target_id) == (first_id, second_id):
        return dict(zip(result.indices[:, 0], result.indices[:, 1]))
    if (source_id, target_id) == (second_id, first_id):
        return dict(zip(result.indices[:, 1], result.indices[:, 0]))
    raise ValueError("pair does not contain requested direction")


def _cycle_supported_mask(
    result: _LearnedPairResult,
    results: dict[tuple[int, int], _LearnedPairResult],
) -> np.ndarray:
    first_id, second_id = result.pair
    intermediates = _cycle_intermediates(result, results)
    supported = np.zeros(len(result.indices), dtype=bool)
    for intermediate_id in intermediates:
        first_pair = tuple(sorted((first_id, intermediate_id)))
        second_pair = tuple(sorted((intermediate_id, second_id)))
        first_map = _directed_match_map(
            results[first_pair], first_id, intermediate_id
        )
        second_map = _directed_match_map(
            results[second_pair], intermediate_id, second_id
        )
        for match_index, (first_feature, second_feature) in enumerate(result.indices):
            intermediate_feature = first_map.get(int(first_feature))
            if (
                intermediate_feature is not None
                and second_map.get(intermediate_feature) == int(second_feature)
            ):
                supported[match_index] = True
    return supported


def _cycle_intermediates(
    result: _LearnedPairResult,
    results: dict[tuple[int, int], _LearnedPairResult],
) -> set[int]:
    first_id, second_id = result.pair
    neighbors: dict[int, set[int]] = {}
    for pair, other in results.items():
        if other.reject_reason or len(other.indices) == 0:
            continue
        neighbors.setdefault(pair[0], set()).add(pair[1])
        neighbors.setdefault(pair[1], set()).add(pair[0])
    return neighbors.get(first_id, set()) & neighbors.get(second_id, set())


def _write_pair_diagnostics(
    path: Path,
    results: list[_LearnedPairResult],
    conflicts_by_pair: dict[tuple[int, int], int],
) -> None:
    fields = [
        "first_id", "second_id", "pair_type", "temporal_gap",
        "retrieval_similarity", "proposal_rank", "raw_matches",
        "essential_inliers", "inlier_ratio", "first_coverage",
        "second_coverage", "cycle_matches", "kept_matches",
        "track_conflicts", "accepted", "reject_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "first_id": result.pair[0],
                    "second_id": result.pair[1],
                    "pair_type": result.pair_type,
                    "temporal_gap": result.temporal_gap,
                    "retrieval_similarity": (
                        "" if result.retrieval_similarity is None
                        else _format_matching_float(result.retrieval_similarity)
                    ),
                    "proposal_rank": result.proposal_rank or "",
                    "raw_matches": result.raw_matches,
                    "essential_inliers": result.essential_inliers,
                    "inlier_ratio": _format_matching_float(result.inlier_ratio),
                    "first_coverage": _format_matching_float(result.first_coverage),
                    "second_coverage": _format_matching_float(result.second_coverage),
                    "cycle_matches": result.cycle_matches,
                    "kept_matches": result.kept_matches,
                    "track_conflicts": conflicts_by_pair.get(result.pair, 0),
                    "accepted": int(result.accepted),
                    "reject_reason": result.reject_reason,
                }
            )


def _format_matching_float(value: float) -> str:
    return f"{float(value):.9g}"


def _generate_wide_learned_correspondences(
    dataset: Stage1Dataset,
    cache_root: Path,
    correspondence_dir: Path,
    summary_path: Path,
    config: MatchingConfig,
    learned_matcher,
    features: dict[int, dict],
    image_shapes: dict[int, tuple[int, int]],
) -> tuple[Path, MatchingSummary]:
    wide_correspondence_dir = cache_root / "wide_correspondences"
    if config.wide_pose_only:
        wide_correspondence_dir.mkdir(parents=True, exist_ok=True)
        for path in wide_correspondence_dir.glob("*.txt"):
            path.unlink()
    image_ids = dataset.image_ids
    local_pairs = set(matching_image_pairs(dataset, config))
    retriever = create_dinov2_retriever(
        device=config.wide_retrieval_device,
        batch_size=config.wide_retrieval_batch_size,
    )
    descriptors = retriever.describe([dataset.image_paths[image_id] for image_id in image_ids])
    proposals = select_global_loop_candidates(
        descriptors,
        list(range(len(image_ids))),
        min_separation=config.wide_min_frame_gap,
        max_candidates=config.wide_retrieval_max_pairs,
        endpoint_spacing=0,
        min_similarity=config.wide_min_similarity,
    )
    retrieved = {
        tuple(sorted((image_ids[item.source_index], image_ids[item.target_index]))): item
        for item in proposals
    }
    pairs = sorted(local_pairs | set(retrieved))
    positions = {image_id: index for index, image_id in enumerate(image_ids)}
    results: dict[tuple[int, int], _LearnedPairResult] = {}

    for pair_index, pair in enumerate(pairs, start=1):
        first_id, second_id = pair
        proposal = retrieved.get(pair)
        pair_type = "local+retrieved" if pair in local_pairs and proposal else (
            "local" if pair in local_pairs else "retrieved"
        )
        match_indices, first_points, second_points = learned_matcher.match_with_indices(
            features[first_id], features[second_id]
        )
        raw_matches = len(match_indices)
        inlier_mask = np.zeros(raw_matches, dtype=bool)
        if raw_matches >= max(5, config.min_inliers):
            try:
                relative = estimate_relative_pose(
                    first_points,
                    second_points,
                    dataset.intrinsics,
                    config.essential_threshold,
                )
                inlier_mask = relative.inliers
            except (RuntimeError, ValueError):
                pass
        inlier_indices = match_indices[inlier_mask]
        inlier_correspondences = np.hstack(
            (first_points[inlier_mask], second_points[inlier_mask])
        )
        inlier_count = len(inlier_indices)
        inlier_ratio = inlier_count / raw_matches if raw_matches else 0.0
        first_coverage = _spatial_coverage(
            first_points[inlier_mask], image_shapes[first_id]
        )
        second_coverage = _spatial_coverage(
            second_points[inlier_mask], image_shapes[second_id]
        )
        result = _LearnedPairResult(
            pair=pair,
            pair_type=pair_type,
            temporal_gap=abs(positions[first_id] - positions[second_id]),
            retrieval_similarity=None if proposal is None else proposal.similarity,
            proposal_rank=None if proposal is None else proposal.rank,
            raw_matches=raw_matches,
            essential_inliers=inlier_count,
            inlier_ratio=inlier_ratio,
            first_coverage=first_coverage,
            second_coverage=second_coverage,
            indices=inlier_indices,
            correspondences=inlier_correspondences,
        )
        if inlier_count < config.min_inliers:
            result.reject_reason = "too few essential inliers"
        elif pair not in local_pairs and inlier_ratio < config.wide_min_inlier_ratio:
            result.reject_reason = "low essential inlier ratio"
        elif pair not in local_pairs and min(first_coverage, second_coverage) < config.wide_min_spatial_coverage:
            result.reject_reason = "low spatial coverage"
        results[pair] = result
        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(f"Geometrically checked {pair_index}/{len(pairs)} learned pairs.")

    for pair, result in results.items():
        if result.reject_reason:
            continue
        if pair not in local_pairs:
            cycle_mask = _cycle_supported_mask(result, results)
            result.cycle_matches = int(np.count_nonzero(cycle_mask))
            if result.cycle_matches < config.wide_min_cycle_matches:
                result.reject_reason = "insufficient cycle support"
                continue
            result.indices = result.indices[cycle_mask]
            result.correspondences = result.correspondences[cycle_mask]
        result.kept_matches = len(result.correspondences)
        if result.kept_matches < config.min_inliers:
            result.reject_reason = "too few matches after cycle filtering"

    if config.wide_max_pairs_per_image < 1:
        raise ValueError("wide_max_pairs_per_image must be positive")
    wide_degree: dict[int, int] = {}
    wide_results = sorted(
        (
            result
            for pair, result in results.items()
            if pair not in local_pairs and not result.reject_reason
        ),
        key=lambda result: (
            result.proposal_rank if result.proposal_rank is not None else float("inf"),
            result.pair,
        ),
    )
    for result in wide_results:
        first_id, second_id = result.pair
        if (
            wide_degree.get(first_id, 0) >= config.wide_max_pairs_per_image
            or wide_degree.get(second_id, 0) >= config.wide_max_pairs_per_image
        ):
            result.reject_reason = "wide endpoint degree limit"
            continue
        wide_degree[first_id] = wide_degree.get(first_id, 0) + 1
        wide_degree[second_id] = wide_degree.get(second_id, 0) + 1

    for result in results.values():
        if result.reject_reason:
            continue
        pair = result.pair
        output_dir = (
            wide_correspondence_dir
            if config.wide_pose_only and pair not in local_pairs
            else correspondence_dir
        )
        np.savetxt(
            output_dir / f"{pair[0]}_{pair[1]}.txt",
            result.correspondences,
            fmt="%.8f",
        )
        result.accepted = True

    accepted_results = [result for result in results.values() if result.accepted]
    correspondence_paths = {
        result.pair: correspondence_dir / f"{result.pair[0]}_{result.pair[1]}.txt"
        for result in accepted_results
        if not config.wide_pose_only or result.pair in local_pairs
    }
    from .tracks import build_tracks

    track_result = build_tracks(
        replace(dataset, correspondence_paths=correspondence_paths),
        min_observations=2,
    )
    _write_pair_diagnostics(
        cache_root / "pair_diagnostics.csv",
        list(results.values()),
        track_result.skipped_conflicts_by_pair,
    )
    summary = MatchingSummary(
        candidate_pairs=len(pairs),
        accepted_pairs=len(accepted_results),
        correspondences=sum(result.kept_matches for result in accepted_results),
    )
    summary_path.write_text(
        json.dumps({"config": asdict(config), "summary": asdict(summary)}, indent=2),
        encoding="utf-8",
    )
    return correspondence_dir, summary


def _generate_cycle_filtered_learned_correspondences(
    dataset: Stage1Dataset,
    cache_root: Path,
    correspondence_dir: Path,
    summary_path: Path,
    config: MatchingConfig,
    learned_matcher,
    features: dict[int, dict],
    image_shapes: dict[int, tuple[int, int]],
) -> tuple[Path, MatchingSummary]:
    pairs = matching_image_pairs(dataset, config)
    positions = {image_id: index for index, image_id in enumerate(dataset.image_ids)}
    results: dict[tuple[int, int], _LearnedPairResult] = {}
    for pair_index, pair in enumerate(pairs, start=1):
        first_id, second_id = pair
        match_indices, first_points, second_points = learned_matcher.match_with_indices(
            features[first_id], features[second_id]
        )
        raw_matches = len(match_indices)
        inlier_mask = np.zeros(raw_matches, dtype=bool)
        if raw_matches >= max(5, config.min_inliers):
            try:
                relative = estimate_relative_pose(
                    first_points,
                    second_points,
                    dataset.intrinsics,
                    config.essential_threshold,
                )
                inlier_mask = relative.inliers
            except (RuntimeError, ValueError):
                pass
        inlier_indices = match_indices[inlier_mask]
        correspondences = np.hstack(
            (first_points[inlier_mask], second_points[inlier_mask])
        )
        inlier_count = len(inlier_indices)
        result = _LearnedPairResult(
            pair=pair,
            pair_type="supplied" if config.pair_source == "supplied" else "local",
            temporal_gap=abs(positions[first_id] - positions[second_id]),
            retrieval_similarity=None,
            proposal_rank=None,
            raw_matches=raw_matches,
            essential_inliers=inlier_count,
            inlier_ratio=inlier_count / raw_matches if raw_matches else 0.0,
            first_coverage=_spatial_coverage(
                first_points[inlier_mask], image_shapes[first_id]
            ),
            second_coverage=_spatial_coverage(
                second_points[inlier_mask], image_shapes[second_id]
            ),
            indices=inlier_indices,
            correspondences=correspondences,
        )
        if inlier_count < config.min_inliers:
            result.reject_reason = "too few essential inliers"
        results[pair] = result
        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(f"Geometrically checked {pair_index}/{len(pairs)} learned pairs.")

    for result in results.values():
        if result.reject_reason:
            continue
        if not _cycle_intermediates(result, results):
            result.kept_matches = len(result.correspondences)
            np.savetxt(
                correspondence_dir / f"{result.pair[0]}_{result.pair[1]}.txt",
                result.correspondences,
                fmt="%.8f",
            )
            result.accepted = True
            continue
        cycle_mask = _cycle_supported_mask(result, results)
        result.cycle_matches = int(np.count_nonzero(cycle_mask))
        if result.cycle_matches < config.learned_min_cycle_matches:
            result.reject_reason = "insufficient cycle support"
            continue
        result.indices = result.indices[cycle_mask]
        result.correspondences = result.correspondences[cycle_mask]
        result.kept_matches = len(result.correspondences)
        np.savetxt(
            correspondence_dir / f"{result.pair[0]}_{result.pair[1]}.txt",
            result.correspondences,
            fmt="%.8f",
        )
        result.accepted = True

    if config.learned_augment_supplied:
        if config.pair_source != "supplied":
            raise ValueError(
                "learned_augment_supplied requires pair_source='supplied'"
            )
        anchor_id = dataset.image_ids[0]
        for result in results.values():
            supplied = dataset.load_correspondences(result.pair)
            if result.accepted and anchor_id not in result.pair:
                result.correspondences = np.vstack(
                    (supplied, result.correspondences)
                )
                result.pair_type = "supplied+learned"
            else:
                result.correspondences = supplied
                result.pair_type = "supplied-only"
            result.kept_matches = len(result.correspondences)
            result.accepted = True
            np.savetxt(
                correspondence_dir / f"{result.pair[0]}_{result.pair[1]}.txt",
                result.correspondences,
                fmt="%.8f",
            )

    accepted = [result for result in results.values() if result.accepted]
    correspondence_paths = {
        result.pair: correspondence_dir / f"{result.pair[0]}_{result.pair[1]}.txt"
        for result in accepted
    }
    from .tracks import build_tracks

    track_result = build_tracks(
        replace(dataset, correspondence_paths=correspondence_paths),
        min_observations=2,
    )
    _write_pair_diagnostics(
        cache_root / "pair_diagnostics.csv",
        list(results.values()),
        track_result.skipped_conflicts_by_pair,
    )
    summary = MatchingSummary(
        candidate_pairs=len(pairs),
        accepted_pairs=len(accepted),
        correspondences=sum(result.kept_matches for result in accepted),
    )
    summary_path.write_text(
        json.dumps({"config": asdict(config), "summary": asdict(summary)}, indent=2),
        encoding="utf-8",
    )
    return correspondence_dir, summary


def generate_correspondences(
    dataset: Stage1Dataset,
    cache_root: str | Path,
    config: MatchingConfig | None = None,
    overwrite: bool = False,
) -> tuple[Path, MatchingSummary]:
    config = config or MatchingConfig()
    cache_root = Path(cache_root)
    feature_dir = cache_root / "features"
    correspondence_dir = cache_root / "correspondences"
    feature_dir.mkdir(parents=True, exist_ok=True)
    correspondence_dir.mkdir(parents=True, exist_ok=True)

    summary_path = cache_root / "matching_summary.json"
    prior_config = None
    if summary_path.is_file():
        prior_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        prior_config = prior_payload.get("config")
        if prior_config is not None:
            try:
                prior_config = asdict(MatchingConfig(**prior_config))
            except TypeError:
                prior_config = None
        if not overwrite and prior_config == asdict(config):
            summary = MatchingSummary(**prior_payload["summary"])
            return correspondence_dir, summary
    regenerate = overwrite or prior_config != asdict(config)
    if regenerate:
        for path in correspondence_dir.glob("*.txt"):
            path.unlink()

    learned_matcher = None
    image_shapes: dict[int, tuple[int, int]] = {}
    if config.feature_mode == "superpoint-lightglue":
        learned_matcher = create_superpoint_lightglue_matcher(
            max_keypoints=config.max_features,
            device=config.learned_device,
            filter_threshold=config.learned_filter_threshold,
            image_scale=1.0,
        )
        features = {}
        for image_id in dataset.image_ids:
            image_path = dataset.image_paths[image_id]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            mask = _feature_mask(
                image, config.mask_apriltags, config.apriltag_padding
            )
            features[image_id] = learned_matcher.extract(image_path, mask=mask)
            image_shapes[image_id] = image.shape[:2]
    else:
        features = {
            image_id: _extract_features(
                dataset.image_paths[image_id],
                image_id,
                feature_dir,
                config,
                regenerate,
            )
            for image_id in dataset.image_ids
        }
    if config.wide_baseline:
        if learned_matcher is None:
            raise ValueError(
                "wide-baseline retrieval currently requires "
                "--feature-mode superpoint-lightglue"
            )
        return _generate_wide_learned_correspondences(
            dataset,
            cache_root,
            correspondence_dir,
            summary_path,
            config,
            learned_matcher,
            features,
            image_shapes,
        )
    if config.learned_cycle_filter:
        if learned_matcher is None:
            raise ValueError(
                "learned_cycle_filter requires superpoint-lightglue"
            )
        return _generate_cycle_filtered_learned_correspondences(
            dataset,
            cache_root,
            correspondence_dir,
            summary_path,
            config,
            learned_matcher,
            features,
            image_shapes,
        )
    if config.learned_augment_supplied:
        raise ValueError(
            "learned_augment_supplied requires learned_cycle_filter"
        )
    if config.wide_pose_only:
        raise ValueError("wide_pose_only requires wide_baseline")
    pairs = matching_image_pairs(dataset, config)
    accepted_pairs = 0
    correspondence_count = 0

    for pair_index, (first_id, second_id) in enumerate(pairs, start=1):
        output_path = correspondence_dir / f"{first_id}_{second_id}.txt"
        if output_path.is_file() and not regenerate:
            values = np.loadtxt(output_path, ndmin=2)
            accepted_pairs += 1
            correspondence_count += len(values)
            continue

        if learned_matcher is not None:
            first_points, second_points = learned_matcher.match(
                features[first_id], features[second_id]
            )
            raw_correspondences = np.hstack((first_points, second_points))
        else:
            raw_correspondences = _match_feature_sets(
                features[first_id],
                features[second_id],
                config.ratio_threshold,
            )
        if len(raw_correspondences) >= max(5, config.min_inliers):
            first_points = raw_correspondences[:, :2]
            second_points = raw_correspondences[:, 2:]
            try:
                relative = estimate_relative_pose(
                    first_points,
                    second_points,
                    dataset.intrinsics,
                    config.essential_threshold,
                )
                inliers = relative.inliers
            except (RuntimeError, ValueError):
                inliers = np.zeros(len(raw_correspondences), dtype=bool)
            if int(inliers.sum()) >= config.min_inliers:
                correspondences = np.hstack(
                    (first_points[inliers], second_points[inliers])
                )
                np.savetxt(output_path, correspondences, fmt="%.8f")
                accepted_pairs += 1
                correspondence_count += len(correspondences)

        if pair_index % 25 == 0 or pair_index == len(pairs):
            print(
                f"Matched {pair_index}/{len(pairs)} image pairs; "
                f"accepted {accepted_pairs}."
            )

    summary = MatchingSummary(
        candidate_pairs=len(pairs),
        accepted_pairs=accepted_pairs,
        correspondences=correspondence_count,
    )
    summary_path.write_text(
        json.dumps(
            {"config": asdict(config), "summary": asdict(summary)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return correspondence_dir, summary
