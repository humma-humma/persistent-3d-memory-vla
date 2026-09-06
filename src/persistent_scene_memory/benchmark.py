"""Deterministic controlled occlusion/change benchmark for memory baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np

from .memory import GeometryObservation, PersistentSceneMemory
from .consistency_memory import ConsistencyGatedMemory

BASELINES = (
    "current_rgb",
    "short_history",
    "persistent_no_gating",
    "confidence_aware",
)


def _observation(position: np.ndarray, confidence: float, frame: int) -> GeometryObservation:
    return GeometryObservation("target", position, np.array([1.0, 0.0]), confidence, frame)


def run_benchmark(
    seed: int = 7,
    trials: int = 40,
    *,
    include_consistency: bool = False,
    single_view_outliers: bool = False,
) -> tuple[list[dict], dict, PersistentSceneMemory]:
    """Evaluate four controlled baselines over noise and occlusion conditions."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    final_memory = PersistentSceneMemory()
    baselines = BASELINES + (("multi_view_consistency",) if include_consistency else ())
    conditions = [(duration, pose, depth) for duration in (2, 5, 8) for pose in (0.0, 0.03) for depth in (0.01, 0.05)]
    for duration, pose_noise, depth_noise in conditions:
        accum = {name: {"action": [], "position": [], "brier": [], "available": []} for name in baselines}
        for trial in range(trials):
            start = np.array([1.0, 0.2, 0.4]) + rng.normal(0, 0.05, 3)
            moved = start + np.array([0.55, -0.15, 0.0])
            hide_start, move_frame = 3, 3 + duration // 2
            reveal_frame, end_frame = 3 + duration, 6 + duration
            history: deque[tuple[int, np.ndarray, float]] = deque(maxlen=3)
            naive_position = None
            naive_confidence = 0.0
            memory = PersistentSceneMemory(confidence_decay=0.12, change_threshold=0.25)
            gated_memory = (
                ConsistencyGatedMemory(
                    confidence_decay=0.12,
                    change_threshold=0.25,
                    confirmation_views=2,
                )
                if include_consistency
                else None
            )
            for frame in range(end_frame):
                truth = moved if frame >= move_frame else start
                visible = not hide_start <= frame < reveal_frame
                observed = None
                observation_confidence = 0.0
                if visible:
                    sigma = pose_noise + depth_noise
                    observed = truth + rng.normal(0, sigma, 3)
                    if single_view_outliers and trial % 2 == 0 and frame == 2:
                        observed = observed + np.array([0.55, -0.15, 0.0])
                    observation_confidence = float(np.exp(-8.0 * sigma))
                    history.append((frame, observed, observation_confidence))
                    if naive_position is None:
                        naive_position = observed.copy()
                    else:
                        naive_position = 0.5 * (naive_position + observed)
                    naive_confidence = max(naive_confidence, observation_confidence)
                    memory.update(_observation(observed, observation_confidence, frame))
                    if gated_memory is not None:
                        gated_memory.update(
                            _observation(observed, observation_confidence, frame)
                        )

                current = (observed, observation_confidence) if observed is not None else (None, 0.0)
                recent = history[-1] if history and frame - history[-1][0] < 3 else None
                short = (recent[1], recent[2]) if recent else (None, 0.0)
                aware_tokens = memory.tokens(frame, min_confidence=0.10)
                aware = (aware_tokens[0].position, aware_tokens[0].confidence) if aware_tokens else (None, 0.0)
                predictions = {
                    "current_rgb": current,
                    "short_history": short,
                    "persistent_no_gating": (naive_position, naive_confidence),
                    "confidence_aware": aware,
                }
                if gated_memory is not None:
                    gated_tokens = gated_memory.tokens(frame, min_confidence=0.10)
                    predictions["multi_view_consistency"] = (
                        (gated_tokens[0].position, gated_tokens[0].confidence)
                        if gated_tokens
                        else (None, 0.0)
                    )
                for name, (prediction, confidence) in predictions.items():
                    available = prediction is not None
                    action = prediction if available else np.zeros(3)
                    error = float(np.linalg.norm(action - truth))
                    correct = float(available and error < 0.15)
                    accum[name]["action"].append(error)
                    accum[name]["available"].append(float(available))
                    accum[name]["brier"].append((float(confidence) - correct) ** 2)
                    if available:
                        accum[name]["position"].append(error)
            final_memory = memory
        for name in baselines:
            values = accum[name]
            rows.append(
                {
                    "baseline": name,
                    "occlusion_frames": duration,
                    "pose_noise": pose_noise,
                    "depth_noise": depth_noise,
                    "action_error": float(np.mean(values["action"])),
                    "memory_position_error": float(np.mean(values["position"])),
                    "confidence_brier": float(np.mean(values["brier"])),
                    "availability": float(np.mean(values["available"])),
                }
            )
    summary = {
        name: {
            metric: float(np.mean([row[metric] for row in rows if row["baseline"] == name]))
            for metric in ("action_error", "memory_position_error", "confidence_brier", "availability")
        }
        for name in baselines
    }
    return rows, summary, final_memory


def write_results(
    output_dir: Path,
    seed: int = 7,
    trials: int = 40,
    *,
    include_consistency: bool = False,
    single_view_outliers: bool = False,
) -> dict:
    rows, summary, memory = run_benchmark(
        seed=seed,
        trials=trials,
        include_consistency=include_consistency,
        single_view_outliers=single_view_outliers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "baseline_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    memory.save(output_dir / "tokens.json", frame_index=14, min_confidence=0.0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/controlled_demo"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--include-consistency", action="store_true")
    parser.add_argument("--single-view-outliers", action="store_true")
    args = parser.parse_args()
    summary = write_results(
        args.output_dir,
        args.seed,
        args.trials,
        include_consistency=args.include_consistency,
        single_view_outliers=args.single_view_outliers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
