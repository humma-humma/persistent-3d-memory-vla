from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


MILK_BOUNDS = np.asarray(
    [[-0.5, -0.15, 0.1], [0.5, 0.4, 1.1]], dtype=np.float64
)


@dataclass(frozen=True)
class MeshProxyMetrics:
    input_points: int
    cropped_points: int
    mean_nearest_vertex_distance: float
    median_nearest_vertex_distance: float
    p90_nearest_vertex_distance: float
    fraction_within_005: float
    cropped_mean_nearest_vertex_distance: float
    cropped_median_nearest_vertex_distance: float
    cropped_p90_nearest_vertex_distance: float
    cropped_fraction_within_005: float


def evaluate_mesh_proxy(
    mesh_vertices: np.ndarray,
    estimated_points: np.ndarray,
    scale: float,
    bounds: np.ndarray = MILK_BOUNDS,
) -> MeshProxyMetrics:
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
    estimated_points = np.asarray(estimated_points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
    if len(mesh_vertices) == 0 or len(estimated_points) == 0:
        raise ValueError("mesh and estimated point clouds must be non-empty")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")

    aligned = estimated_points / scale
    inside = np.all((aligned >= bounds[0]) & (aligned <= bounds[1]), axis=1)
    cropped = aligned[inside]
    if len(cropped) == 0:
        raise ValueError("no estimated points remain inside the evaluation bounds")
    tree = cKDTree(mesh_vertices)
    distances = tree.query(aligned)[0]
    cropped_distances = tree.query(cropped)[0]
    return MeshProxyMetrics(
        input_points=len(estimated_points),
        cropped_points=len(cropped),
        mean_nearest_vertex_distance=float(np.mean(distances)),
        median_nearest_vertex_distance=float(np.median(distances)),
        p90_nearest_vertex_distance=float(np.percentile(distances, 90)),
        fraction_within_005=float(np.mean(distances <= 0.05)),
        cropped_mean_nearest_vertex_distance=float(np.mean(cropped_distances)),
        cropped_median_nearest_vertex_distance=float(np.median(cropped_distances)),
        cropped_p90_nearest_vertex_distance=float(
            np.percentile(cropped_distances, 90)
        ),
        cropped_fraction_within_005=float(np.mean(cropped_distances <= 0.05)),
    )


def _load_vertices(path: Path) -> np.ndarray:
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError(
            "Stage 2 mesh evaluation requires the 'evaluation' extra"
        ) from error
    geometry = trimesh.load(path, process=False)
    if not hasattr(geometry, "vertices"):
        raise ValueError(f"Expected a single mesh or point cloud: {path}")
    return np.asarray(geometry.vertices, dtype=np.float64)


def evaluate_result(dataset: Path, result_dir: Path) -> dict[str, object]:
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pose_evaluation = summary.get("pose_evaluation")
    if not pose_evaluation or "scale" not in pose_evaluation:
        raise ValueError(f"No pose-evaluation scale in {summary_path}")
    metrics = evaluate_mesh_proxy(
        _load_vertices(dataset / "gt_mesh.ply"),
        _load_vertices(result_dir / "estimated_points.ply"),
        float(pose_evaluation["scale"]),
    )
    return {
        "dataset": str(dataset),
        "result_dir": str(result_dir),
        "scale": float(pose_evaluation["scale"]),
        "metric": "nearest-mesh-vertex proxy",
        **asdict(metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 2 milk result.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = evaluate_result(args.dataset, args.result_dir)
    output = args.output or args.result_dir / "stage2_mesh_proxy.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
