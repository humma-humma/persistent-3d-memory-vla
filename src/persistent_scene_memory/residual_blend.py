"""Calibrate a scalar hold-state residual blend from saved policy rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_hold_blend(
    predictions: np.ndarray,
    holds: np.ndarray,
    targets: np.ndarray,
    per_dimension: bool = False,
) -> float | list[float]:
    """Return the least-squares coefficient for hold + alpha*(policy-hold)."""
    predictions = np.asarray(predictions, dtype=np.float64)
    holds = np.asarray(holds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != holds.shape or predictions.shape != targets.shape:
        raise ValueError("predictions, holds, and targets must have identical shapes")
    correction = predictions - holds
    axes = tuple(range(correction.ndim - 1)) if per_dimension else None
    denominator = np.sum(correction**2, axis=axes)
    numerator = np.sum(correction * (targets - holds), axis=axes)
    alpha = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )
    clipped = np.clip(alpha, 0.0, 1.0)
    return clipped.tolist() if per_dimension else float(clipped)


def calibrate(evaluation: dict, per_dimension: bool = False) -> dict:
    """Fit from an evaluation artifact that explicitly contains action arrays."""
    if any(key in evaluation.get("episodes", []) for key in (0, 11, 12, 13, 14, 15)):
        raise ValueError("calibration artifact overlaps the frozen evaluation episodes")
    results = evaluation.get("results", [])
    required = ("predicted_actions", "hold_actions", "recorded_actions")
    if not results or any(any(key not in item for key in required) for item in results):
        raise ValueError("calibration evaluation must be created with --include-actions")
    predictions = np.asarray([item["predicted_actions"] for item in results])
    holds = np.asarray([item["hold_actions"] for item in results])
    targets = np.asarray([item["recorded_actions"] for item in results])
    alpha = fit_hold_blend(predictions, holds, targets, per_dimension)
    return {
        "checkpoint": evaluation["checkpoint"],
        "calibration_episodes": evaluation["episodes"],
        "calibration_anchors": evaluation["anchors"],
        "sampled_actions": int(np.prod(predictions.shape[:-1])),
        "hold_blend_alpha": alpha,
        "per_dimension": per_dimension,
        "formula": "hold + alpha * (smolvla - hold)",
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dimension", action="store_true")
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    report = calibrate(evaluation, args.per_dimension)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
