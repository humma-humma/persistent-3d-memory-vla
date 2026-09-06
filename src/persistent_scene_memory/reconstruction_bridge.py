"""Convert sequential rich SfM point clouds into persistent memory tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .adapter import GeometryTokenAdapter
from .consistency_memory import ConsistencyGatedMemory
from .memory import GeometryObservation, PersistentSceneMemory


REQUIRED_PROPERTIES = (
    "x",
    "y",
    "z",
    "red",
    "green",
    "blue",
    "track_id",
    "registered_observations",
    "mean_reprojection_error",
    "max_triangulation_angle",
)


def _load_ascii_vertices(path: str | Path) -> tuple[list[str], np.ndarray]:
    source = Path(path)
    with source.open("r", encoding="ascii") as file:
        if file.readline().strip() != "ply" or file.readline().strip() != "format ascii 1.0":
            raise ValueError(f"{source} must be an ASCII PLY file")
        count = None
        element = None
        properties = []
        while True:
            fields = file.readline().strip().split()
            if not fields:
                raise ValueError(f"{source} has no end_header marker")
            if fields == ["end_header"]:
                break
            if fields[0] == "element" and len(fields) == 3:
                element = fields[1]
                if element == "vertex":
                    count = int(fields[2])
            elif fields[0] == "property" and element == "vertex":
                properties.append(fields[-1])
        if count is None:
            raise ValueError(f"{source} has no vertex count")
        values = (
            np.loadtxt(file, dtype=np.float64, max_rows=count, ndmin=2)
            if count
            else np.empty((0, len(properties)), dtype=np.float64)
        )
        if values.shape != (count, len(properties)):
            raise ValueError(f"{source} vertex data does not match its header")
    return properties, values


def _property(properties: list[str], values: np.ndarray, name: str) -> np.ndarray:
    if name not in properties:
        raise ValueError(f"rich point cloud is missing vertex property {name!r}")
    return values[:, properties.index(name)]


def geometric_confidence(
    registered_observations: float,
    mean_reprojection_error: float,
    max_triangulation_angle: float,
    *,
    full_support: float = 3.0,
    reprojection_scale: float = 2.0,
    full_angle: float = 5.0,
) -> float:
    """Map SfM support, residual, and baseline diagnostics into [0, 1]."""
    if full_support <= 0.0 or reprojection_scale <= 0.0 or full_angle <= 0.0:
        raise ValueError("confidence scales must be positive")
    diagnostics = np.asarray(
        [registered_observations, mean_reprojection_error, max_triangulation_angle],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(diagnostics)):
        return 0.0
    support = np.clip(registered_observations / full_support, 0.0, 1.0)
    residual = np.exp(-max(mean_reprojection_error, 0.0) / reprojection_scale)
    angle = np.clip(max_triangulation_angle / full_angle, 0.0, 1.0)
    return float(np.clip(support * residual * angle, 0.0, 1.0))


def observations_from_rich_ply(
    path: str | Path,
    frame_index: int,
    *,
    full_support: float = 3.0,
    reprojection_scale: float = 2.0,
    full_angle: float = 5.0,
) -> list[GeometryObservation]:
    """Load world-frame points, using RGB as the portable point feature."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    properties, values = _load_ascii_vertices(path)
    for name in REQUIRED_PROPERTIES:
        _property(properties, values, name)

    track_ids = _property(properties, values, "track_id")
    integer_track_ids = track_ids.astype(np.int64)
    if not np.all(track_ids == integer_track_ids):
        raise ValueError("track_id values must be integers")
    if len(np.unique(integer_track_ids)) != len(integer_track_ids):
        raise ValueError("track_id values must be unique within a snapshot")

    colors = np.column_stack(
        [_property(properties, values, name) for name in ("red", "green", "blue")]
    )
    if np.any((colors < 0.0) | (colors > 255.0)):
        raise ValueError("RGB values must be in [0, 255]")
    registered = _property(properties, values, "registered_observations")
    errors = _property(properties, values, "mean_reprojection_error")
    angles = _property(properties, values, "max_triangulation_angle")
    points = np.column_stack(
        [_property(properties, values, name) for name in ("x", "y", "z")]
    )

    return [
        GeometryObservation(
            key=f"track:{track_id}",
            position=points[row],
            feature=colors[row] / 255.0,
            confidence=geometric_confidence(
                registered[row],
                errors[row],
                angles[row],
                full_support=full_support,
                reprojection_scale=reprojection_scale,
                full_angle=full_angle,
            ),
            frame_index=frame_index,
        )
        for row, track_id in enumerate(integer_track_ids)
    ]


def process_reconstruction_sequence(
    inputs: list[Path],
    output_dir: Path,
    *,
    confidence_decay: float = 0.02,
    change_threshold: float = 0.25,
    min_confidence: float = 0.1,
    max_tokens: int = 32,
    position_scale: float = 1.0,
    full_support: float = 3.0,
    reprojection_scale: float = 2.0,
    full_angle: float = 5.0,
    confirmation_views: int = 1,
) -> dict:
    """Update memory in input order and write policy-ready snapshots."""
    if not inputs:
        raise ValueError("at least one rich point cloud is required")
    memory = (
        ConsistencyGatedMemory(
            confidence_decay=confidence_decay,
            change_threshold=change_threshold,
            confirmation_views=confirmation_views,
        )
        if confirmation_views > 1
        else PersistentSceneMemory(
            confidence_decay=confidence_decay,
            change_threshold=change_threshold,
        )
    )
    adapter = GeometryTokenAdapter(max_tokens=max_tokens, position_scale=position_scale)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for frame_index, input_path in enumerate(inputs):
        observations = observations_from_rich_ply(
            input_path,
            frame_index,
            full_support=full_support,
            reprojection_scale=reprojection_scale,
            full_angle=full_angle,
        )
        decisions = [memory.update(observation) for observation in observations]
        tokens = memory.tokens(frame_index, min_confidence=min_confidence)
        policy_tokens = (
            memory.policy_tokens(frame_index, min_confidence=min_confidence)
            if isinstance(memory, ConsistencyGatedMemory)
            else tokens
        )
        encoded = adapter.encode(
            policy_tokens,
            frame_index,
            feature_size=4 if isinstance(memory, ConsistencyGatedMemory) else 3,
        )
        memory_path = output_dir / f"memory_frame_{frame_index:06d}.json"
        adapter_path = output_dir / f"adapter_frame_{frame_index:06d}.json"
        memory.save(memory_path, frame_index, min_confidence=min_confidence)
        adapter_path.write_text(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "values": encoded.values.tolist(),
                    "mask": encoded.mask.tolist(),
                    "keys": list(encoded.keys),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        frames.append(
            {
                "frame_index": frame_index,
                "input": str(input_path),
                "observations": len(observations),
                "memory_tokens": len(tokens),
                "adapter_tokens": int(encoded.mask.sum()),
                "update_decisions": {
                    state: sum(decision.state == state for decision in decisions)
                    for state in ("retain", "uncertain", "update")
                }
                if isinstance(memory, ConsistencyGatedMemory)
                else None,
                "memory": memory_path.name,
                "adapter": adapter_path.name,
            }
        )

    summary = {
        "coordinate_contract": "All inputs must share one world frame and persistent track IDs.",
        "feature": "RGB normalized to [0, 1]",
        "update_rule": (
            f"multi-view consistency with {confirmation_views} confirming views"
            if confirmation_views > 1
            else "single-view confidence-aware update"
        ),
        "confidence": {
            "formula": "clip(obs/full_support)*exp(-reprojection/reprojection_scale)*clip(angle/full_angle)",
            "full_support": full_support,
            "reprojection_scale": reprojection_scale,
            "full_angle": full_angle,
        },
        "frames": frames,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Sequential estimated_points_rich.ply files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-decay", type=float, default=0.02)
    parser.add_argument("--change-threshold", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--full-support", type=float, default=3.0)
    parser.add_argument("--reprojection-scale", type=float, default=2.0)
    parser.add_argument("--full-angle", type=float, default=5.0)
    parser.add_argument("--confirmation-views", type=int, default=1)
    args = parser.parse_args()
    summary = process_reconstruction_sequence(
        args.inputs,
        args.output_dir,
        confidence_decay=args.confidence_decay,
        change_threshold=args.change_threshold,
        min_confidence=args.min_confidence,
        max_tokens=args.max_tokens,
        position_scale=args.position_scale,
        full_support=args.full_support,
        reprojection_scale=args.reprojection_scale,
        full_angle=args.full_angle,
        confirmation_views=args.confirmation_views,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
