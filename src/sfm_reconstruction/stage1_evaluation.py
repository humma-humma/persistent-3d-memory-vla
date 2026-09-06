from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Stage1PointMetrics:
    ground_truth_points: int
    estimated_points: int
    mean_ground_truth_to_estimate: float
    mean_estimate_to_ground_truth: float
    chamfer_distance: float


def evaluate_stage1_points(
    ground_truth: np.ndarray, estimated: np.ndarray
) -> Stage1PointMetrics:
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 3)
    estimated = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    if len(ground_truth) == 0 or len(estimated) == 0:
        raise ValueError("point clouds must be non-empty")
    gt_to_estimate = cKDTree(estimated).query(ground_truth)[0]
    estimate_to_gt = cKDTree(ground_truth).query(estimated)[0]
    first = float(np.mean(gt_to_estimate))
    second = float(np.mean(estimate_to_gt))
    return Stage1PointMetrics(
        ground_truth_points=len(ground_truth),
        estimated_points=len(estimated),
        mean_ground_truth_to_estimate=first,
        mean_estimate_to_ground_truth=second,
        chamfer_distance=first + second,
    )


def _load_vertices(path: Path) -> np.ndarray:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "Stage 1 point evaluation requires the 'evaluation' extra"
        ) from error
    geometry = trimesh.load(path, process=False)
    if not hasattr(geometry, "vertices"):
        raise ValueError(f"Expected a single point cloud: {path}")
    return np.asarray(geometry.vertices, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 points.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = evaluate_stage1_points(
        _load_vertices(args.ground_truth),
        _load_vertices(args.result_dir / "estimated_points.ply"),
    )
    payload = {
        "ground_truth": str(args.ground_truth),
        "result_dir": str(args.result_dir),
        **asdict(metrics),
    }
    output = args.output or args.result_dir / "stage1_point_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
