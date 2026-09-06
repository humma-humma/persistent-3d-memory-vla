"""Causal acquisition-order reconstruction, memory export, and validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from sfm_reconstruction.dataset import Stage1Dataset, load_stage1_dataset
from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.io import (
    write_camera_parameters,
    write_point_cloud,
    write_rich_point_cloud,
    write_summary,
)
from sfm_reconstruction.reconstruction import (
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct,
)
from sfm_reconstruction.tracks import ObservationKey, observation_key

from .reconstruction_bridge import process_reconstruction_sequence


@dataclass
class StableTrackRegistry:
    """Carry point identities across independently reconstructed prefixes."""

    tolerance: float = 1e-3
    observation_to_id: dict[ObservationKey, int] = field(default_factory=dict)
    next_id: int = 0
    merged_identifiers: int = 0
    split_identifiers: int = 0

    def assign(self, result: ReconstructionResult) -> dict[int, int]:
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for track_id in sorted(result.points):
            track = result.tracks[track_id]
            keys = [
                observation_key(image_id, point, self.tolerance)
                for image_id, point in track.observations.items()
            ]
            known = sorted(
                {self.observation_to_id[key] for key in keys if key in self.observation_to_id}
            )
            stable_id = known[0] if known else self.next_id
            if not known:
                self.next_id += 1
            elif len(known) > 1:
                self.merged_identifiers += len(known) - 1
            if stable_id in used:
                stable_id = self.next_id
                self.next_id += 1
                self.split_identifiers += 1
            used.add(stable_id)
            assigned[track_id] = stable_id
            for key in keys:
                self.observation_to_id[key] = stable_id
        return assigned


def _points_observed_in(
    result: ReconstructionResult,
    image_ids: tuple[int, ...],
) -> ReconstructionResult:
    visible = set(image_ids) & set(result.poses)
    points = {
        track_id: point
        for track_id, point in result.points.items()
        if visible & set(result.tracks[track_id].observations)
    }
    return ReconstructionResult(
        poses=result.poses,
        points=points,
        tracks=result.tracks,
        initial_pair=result.initial_pair,
        skipped_track_conflicts=result.skipped_track_conflicts,
    )


def _load_ground_truth_points(dataset: Stage1Dataset) -> np.ndarray | None:
    path = dataset.root / "gt_points.ply"
    if not path.is_file():
        return None
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("ground-truth point validation requires trimesh") from exc
    geometry = trimesh.load(path, process=False)
    if not hasattr(geometry, "vertices"):
        raise ValueError(f"expected point-cloud vertices in {path}")
    return np.asarray(geometry.vertices, dtype=np.float64)


def _nearest_errors(
    positions: np.ndarray,
    scale: float | None,
    ground_truth_tree: cKDTree | None,
) -> np.ndarray | None:
    if ground_truth_tree is None or scale is None or scale <= 0.0 or not len(positions):
        return None
    return ground_truth_tree.query(np.asarray(positions) / scale)[0]


def _quality_summary(
    confidence: np.ndarray,
    errors: np.ndarray,
    correctness_threshold: float,
) -> dict:
    correct = errors <= correctness_threshold
    brier = (confidence - correct.astype(np.float64)) ** 2
    bins = []
    for lower in np.linspace(0.0, 0.8, 5):
        upper = lower + 0.2
        selected = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if selected.any():
            bins.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": int(selected.sum()),
                    "mean_confidence": float(np.mean(confidence[selected])),
                    "empirical_correct": float(np.mean(correct[selected])),
                    "brier": float(np.mean(brier[selected])),
                }
            )
    return {
        "count": len(errors),
        "mean_nearest_gt_error": float(np.mean(errors)),
        "median_nearest_gt_error": float(np.median(errors)),
        "fraction_correct": float(np.mean(correct)),
        "brier": float(np.mean(brier)),
        "calibration_bins": bins,
    }


def build_validation_report(
    memory_dir: Path,
    frame_rows: list[dict],
    ground_truth_points: np.ndarray | None,
    *,
    correctness_threshold: float = 0.15,
) -> dict:
    tree = cKDTree(ground_truth_points) if ground_truth_points is not None else None
    memory_rows = []
    age_values: dict[int, dict[str, list[float]]] = {}
    all_confidence = []
    all_errors = []
    for row in frame_rows:
        frame_index = row["frame_index"]
        payload = json.loads(
            (memory_dir / f"memory_frame_{frame_index:06d}.json").read_text(
                encoding="utf-8"
            )
        )
        tokens = payload["tokens"]
        positions = np.asarray([token["position"] for token in tokens], dtype=np.float64)
        confidence = np.asarray([token["confidence"] for token in tokens], dtype=np.float64)
        ages = np.asarray(
            [frame_index - int(token["last_seen"]) for token in tokens], dtype=np.int64
        )
        errors = _nearest_errors(positions, row.get("pose_scale"), tree)
        memory_row = {
            "frame_index": frame_index,
            "tokens": len(tokens),
            "current_tokens": int(np.sum(ages == 0)),
            "persisted_tokens": int(np.sum(ages > 0)),
            "maximum_age": int(np.max(ages)) if len(ages) else 0,
            "mean_confidence": float(np.mean(confidence)) if len(confidence) else None,
        }
        if errors is not None:
            memory_row["quality"] = _quality_summary(
                confidence, errors, correctness_threshold
            )
            all_confidence.extend(confidence.tolist())
            all_errors.extend(errors.tolist())
            for age, value, error in zip(ages, confidence, errors):
                bucket = age_values.setdefault(int(age), {"confidence": [], "errors": []})
                bucket["confidence"].append(float(value))
                bucket["errors"].append(float(error))
        memory_rows.append(memory_row)

    by_age = []
    for age, values in sorted(age_values.items()):
        confidence = np.asarray(values["confidence"])
        errors = np.asarray(values["errors"])
        by_age.append(
            {
                "age": age,
                **_quality_summary(confidence, errors, correctness_threshold),
                "mean_confidence": float(np.mean(confidence)),
            }
        )
    report = {
        "causal_contract": (
            "Frame t uses only images and correspondence files whose endpoints "
            "are at or before acquisition t."
        ),
        "point_error_metric": "nearest ground-truth point after pose-derived scale",
        "correctness_threshold": correctness_threshold,
        "frames": frame_rows,
        "memory_frames": memory_rows,
        "quality_by_age": by_age,
    }
    if all_errors:
        report["aggregate_memory_quality"] = _quality_summary(
            np.asarray(all_confidence),
            np.asarray(all_errors),
            correctness_threshold,
        )
    return report


def run_causal_mapping(
    dataset: Stage1Dataset,
    output_dir: Path,
    config: ReconstructionConfig | None = None,
    *,
    max_images: int | None = None,
    min_confidence: float = 0.1,
    correctness_threshold: float = 0.15,
    confirmation_views: int = 1,
) -> dict:
    dataset = dataset.subset(max_images)
    image_ids = dataset.image_ids
    initial_pair = (image_ids[0], image_ids[1])
    config = replace(config or ReconstructionConfig(), initial_pair=initial_pair)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = StableTrackRegistry(config.track_tolerance)
    observation_paths = []
    frame_rows = []
    previous_positions: dict[int, np.ndarray] = {}

    for frame_index, prefix_size in enumerate(range(2, len(image_ids) + 1)):
        prefix = dataset.subset(prefix_size)
        result = reconstruct(prefix, config)
        stable_ids = registry.assign(result)
        observed_image_ids = initial_pair if frame_index == 0 else (image_ids[prefix_size - 1],)
        observed = _points_observed_in(result, observed_image_ids)
        frame_dir = output_dir / "causal_snapshots" / f"frame_{frame_index:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        write_camera_parameters(frame_dir / "estimated_camera_parameters.json", prefix, result)
        write_point_cloud(frame_dir / "estimated_points.ply", result)
        write_rich_point_cloud(
            frame_dir / "estimated_points_rich.ply", prefix, result, stable_ids
        )
        observed_path = frame_dir / "observed_points_rich.ply"
        write_rich_point_cloud(observed_path, prefix, observed, stable_ids)
        write_summary(frame_dir / "summary.json", prefix, result)
        observation_paths.append(observed_path)

        stable_positions = {
            stable_ids[track_id]: point for track_id, point in result.points.items()
        }
        shared = sorted(set(stable_positions) & set(previous_positions))
        drift = (
            float(
                np.mean(
                    [
                        np.linalg.norm(stable_positions[key] - previous_positions[key])
                        for key in shared
                    ]
                )
            )
            if shared
            else None
        )
        pose_metrics = evaluate_poses(result.poses, prefix.ground_truth_extrinsics)
        row = {
            "frame_index": frame_index,
            "acquisition_image_id": image_ids[prefix_size - 1],
            "available_image_ids": image_ids[:prefix_size],
            "registered_cameras": len(result.poses),
            "map_points": len(result.points),
            "observed_points": len(observed.points),
            "stable_ids": len(set(stable_ids.values())),
            "shared_ids_with_previous": len(shared),
            "mean_shared_point_drift": drift,
            "pose_metrics": pose_metrics.to_dict() if pose_metrics else None,
            "pose_scale": pose_metrics.scale if pose_metrics else None,
            "snapshot_dir": str(frame_dir.relative_to(output_dir)),
        }
        frame_rows.append(row)
        previous_positions = stable_positions

    memory_dir = output_dir / "causal_memory"
    process_reconstruction_sequence(
        observation_paths,
        memory_dir,
        min_confidence=min_confidence,
        confirmation_views=confirmation_views,
    )
    ground_truth_points = _load_ground_truth_points(dataset)
    report = build_validation_report(
        memory_dir,
        frame_rows,
        ground_truth_points,
        correctness_threshold=correctness_threshold,
    )
    report["dataset"] = str(dataset.root)
    report["initial_pair"] = list(initial_pair)
    report["stable_identity"] = {
        "identifiers": registry.next_id,
        "merged_identifiers": registry.merged_identifiers,
        "split_identifiers": registry.split_identifiers,
    }
    (output_dir / "causal_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--no-bundle-adjustment", action="store_true")
    parser.add_argument("--bundle-adjustment-max-nfev", type=int, default=20)
    parser.add_argument("--min-pnp-points", type=int, default=12)
    parser.add_argument("--min-pnp-inliers", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--correctness-threshold", type=float, default=0.15)
    parser.add_argument("--confirmation-views", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_causal_mapping(
        load_stage1_dataset(args.dataset),
        args.output_dir,
        ReconstructionConfig(
            bundle_adjustment=not args.no_bundle_adjustment,
            bundle_adjustment_max_nfev=args.bundle_adjustment_max_nfev,
            min_pnp_points=args.min_pnp_points,
            min_pnp_inliers=args.min_pnp_inliers,
        ),
        max_images=args.max_images,
        min_confidence=args.min_confidence,
        correctness_threshold=args.correctness_threshold,
        confirmation_views=args.confirmation_views,
    )
    final_frame = report["frames"][-1]
    final_memory = report["memory_frames"][-1]
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "frames": len(report["frames"]),
                "final_registered_cameras": final_frame["registered_cameras"],
                "final_map_points": final_frame["map_points"],
                "final_memory_tokens": final_memory["tokens"],
                "validation": str((args.output_dir / "causal_validation.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
