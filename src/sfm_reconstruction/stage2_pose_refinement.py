from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .dataset import Stage1Dataset
from .geometry import estimate_relative_pose
from .models import Pose
from .pose_graph import (
    PoseGraph,
    PoseGraphConfig,
    PoseGraphEdge,
    PoseGraphNode,
    optimize_pose_graph,
)
from .reconstruction import (
    ReconstructionConfig,
    ReconstructionResult,
    retriangulate_with_poses,
)
from .stage3 import _transform_from_trajectory_pose


@dataclass(frozen=True)
class Stage2PoseRefinementConfig:
    essential_threshold: float = 1.0
    min_inliers: int = 15
    max_rotation_disagreement_degrees: float = 1.5
    max_translation_direction_disagreement_degrees: float = 5.0
    wide_rotation_weight: float = 0.15
    wide_translation_weight: float = 0.05
    optimizer_max_nfev: int = 100


@dataclass(frozen=True)
class Stage2PoseRefinementSummary:
    candidate_constraints: int
    accepted_constraints: int
    initial_cost: float
    final_cost: float
    iterations: int
    success: bool
    points_before: int
    points_after: int


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _direction_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(3)
    second = np.asarray(second, dtype=np.float64).reshape(3)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return 180.0
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pair_from_path(path: Path) -> tuple[int, int]:
    values = path.stem.split("_")
    if len(values) != 2:
        raise ValueError(f"Invalid correspondence filename: {path.name}")
    return int(values[0]), int(values[1])


def build_stage2_pose_graph(
    poses: dict[int, Pose],
    intrinsics: np.ndarray,
    wide_paths: list[Path],
    config: Stage2PoseRefinementConfig,
) -> tuple[PoseGraph, list[dict[str, object]]]:
    image_ids = sorted(poses)
    nodes = [
        PoseGraphNode(
            index=image_id,
            timestamp=float(image_id),
            camera_to_world=np.linalg.inv(poses[image_id].matrix()),
        )
        for image_id in image_ids
    ]
    edges = []
    for first_id, second_id in zip(image_ids, image_ids[1:]):
        measured = poses[second_id].matrix() @ np.linalg.inv(
            poses[first_id].matrix()
        )
        edges.append(
            PoseGraphEdge(
                source_index=first_id,
                target_index=second_id,
                measured_target_from_source=measured,
                translation_weight=1.0,
                rotation_weight=1.0,
                inlier_count=0,
                edge_type="odometry",
            )
        )

    diagnostics: list[dict[str, object]] = []
    for path in sorted(wide_paths):
        first_id, second_id = _pair_from_path(path)
        row: dict[str, object] = {
            "first_id": first_id,
            "second_id": second_id,
            "matches": 0,
            "inliers": 0,
            "rotation_disagreement_degrees": "",
            "translation_direction_disagreement_degrees": "",
            "accepted": 0,
            "reject_reason": "",
        }
        if first_id not in poses or second_id not in poses:
            row["reject_reason"] = "unregistered endpoint"
            diagnostics.append(row)
            continue
        matches = np.loadtxt(path, ndmin=2)
        row["matches"] = len(matches)
        try:
            relative = estimate_relative_pose(
                matches[:, :2],
                matches[:, 2:],
                intrinsics,
                config.essential_threshold,
            )
        except (RuntimeError, ValueError):
            row["reject_reason"] = "relative pose estimation failed"
            diagnostics.append(row)
            continue
        inliers = int(np.count_nonzero(relative.inliers))
        row["inliers"] = inliers
        if inliers < config.min_inliers:
            row["reject_reason"] = "too few relative-pose inliers"
            diagnostics.append(row)
            continue

        initial = poses[second_id].matrix() @ np.linalg.inv(
            poses[first_id].matrix()
        )
        measured = relative.pose.matrix()
        rotation_disagreement = _rotation_angle_degrees(
            measured[:3, :3] @ initial[:3, :3].T
        )
        direction_disagreement = _direction_angle_degrees(
            measured[:3, 3], initial[:3, 3]
        )
        row["rotation_disagreement_degrees"] = rotation_disagreement
        row["translation_direction_disagreement_degrees"] = direction_disagreement
        if rotation_disagreement > config.max_rotation_disagreement_degrees:
            row["reject_reason"] = "rotation disagrees with local solution"
            diagnostics.append(row)
            continue
        if (
            direction_disagreement
            > config.max_translation_direction_disagreement_degrees
        ):
            row["reject_reason"] = "translation direction disagrees with local solution"
            diagnostics.append(row)
            continue

        initial_baseline = np.linalg.norm(initial[:3, 3])
        measured_baseline = np.linalg.norm(measured[:3, 3])
        measured[:3, 3] *= initial_baseline / measured_baseline
        edges.append(
            PoseGraphEdge(
                source_index=first_id,
                target_index=second_id,
                measured_target_from_source=measured,
                translation_weight=config.wide_translation_weight,
                rotation_weight=config.wide_rotation_weight,
                inlier_count=inliers,
                edge_type="loop",
            )
        )
        row["accepted"] = 1
        diagnostics.append(row)
    return PoseGraph(nodes=nodes, edges=edges), diagnostics


def refine_stage2_poses(
    dataset: Stage1Dataset,
    result: ReconstructionResult,
    wide_correspondence_dir: Path,
    reconstruction_config: ReconstructionConfig,
    config: Stage2PoseRefinementConfig | None = None,
) -> tuple[ReconstructionResult, Stage2PoseRefinementSummary, list[dict[str, object]]]:
    config = config or Stage2PoseRefinementConfig()
    paths = sorted(wide_correspondence_dir.glob("*.txt"))
    graph, diagnostics = build_stage2_pose_graph(
        result.poses, dataset.intrinsics, paths, config
    )
    accepted = sum(int(row["accepted"]) for row in diagnostics)
    if accepted == 0:
        raise RuntimeError("No wide pose constraints passed consistency checks")
    optimization = optimize_pose_graph(
        graph,
        PoseGraphConfig(
            optimizer_loss="soft_l1",
            optimizer_f_scale=1.0,
            optimizer_max_nfev=config.optimizer_max_nfev,
        ),
    )
    optimized_poses = {}
    for node, trajectory_pose in zip(graph.nodes, optimization.poses):
        world_to_camera = np.linalg.inv(
            _transform_from_trajectory_pose(trajectory_pose)
        )
        optimized_poses[node.index] = Pose(
            world_to_camera[:3, :3], world_to_camera[:3, 3]
        )
    refined = retriangulate_with_poses(
        dataset, result, optimized_poses, reconstruction_config
    )
    summary = Stage2PoseRefinementSummary(
        candidate_constraints=len(diagnostics),
        accepted_constraints=accepted,
        initial_cost=optimization.initial_cost,
        final_cost=optimization.final_cost,
        iterations=optimization.iterations,
        success=optimization.success,
        points_before=len(result.points),
        points_after=len(refined.points),
    )
    return refined, summary, diagnostics


def write_stage2_pose_refinement(
    output_dir: Path,
    config: Stage2PoseRefinementConfig,
    summary: Stage2PoseRefinementSummary,
    diagnostics: list[dict[str, object]],
) -> None:
    (output_dir / "stage2_pose_refinement.json").write_text(
        json.dumps(
            {"config": asdict(config), "summary": asdict(summary)}, indent=2
        ),
        encoding="utf-8",
    )
    fields = list(diagnostics[0]) if diagnostics else []
    with (output_dir / "wide_pose_constraints.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
