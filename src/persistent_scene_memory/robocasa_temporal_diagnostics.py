"""Plot temporal action behavior for held-out and closed-loop RoboCasa runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .robocasa_finetune import load_episode_vectors
from .robocasa_rollout import RoboCasaEpisode


def switching_stats(actions: np.ndarray) -> dict:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) == 0:
        raise ValueError("actions must have shape [steps, 12]")
    gripper = actions[:, 6] >= 0.5
    control = actions[:, 11] >= 0.5
    return {
        "frames": len(actions),
        "gripper_close_fraction": float(gripper.mean()),
        "gripper_transitions": int(np.count_nonzero(np.diff(gripper))),
        "base_control_fraction": float(control.mean()),
        "control_transitions": int(np.count_nonzero(np.diff(control))),
        "base_active_fraction": float(
            (np.linalg.norm(actions[:, 7:11], axis=1) > 0.05).mean()
        ),
    }


def export_diagnostics(
    parent_predictions: str | Path,
    candidate_predictions: str | Path,
    closed_loop_episode: str | Path,
    output_dir: str | Path,
) -> dict:
    import matplotlib.pyplot as plt

    with np.load(parent_predictions) as archive:
        target = archive["target"].copy()
        parent = archive["prediction"].copy()
    with np.load(candidate_predictions) as archive:
        candidate_target = archive["target"].copy()
        candidate = archive["prediction"].copy()
    if not np.array_equal(target, candidate_target):
        raise ValueError("parent and candidate targets do not match")
    _, closed_loop = load_episode_vectors(RoboCasaEpisode(closed_loop_episode))
    series = {
        "heldout_target": target,
        "short10_parent": parent,
        "short10_gripper_candidate": candidate,
        "candidate_closed_loop": closed_loop,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    for name, actions in series.items():
        frame = np.arange(len(actions))
        axes[0].step(frame, actions[:, 6], where="post", label=name, alpha=0.85)
        axes[1].step(frame, actions[:, 11], where="post", label=name, alpha=0.85)
        axes[2].plot(frame, np.linalg.norm(actions[:, 7:11], axis=1), label=name, alpha=0.85)
    axes[0].set_ylabel("gripper close")
    axes[1].set_ylabel("base control")
    axes[2].set_ylabel("base magnitude")
    axes[2].set_xlabel("action frame")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle("RoboCasa temporal action diagnostics")
    figure.savefig(destination / "timeline.png", dpi=160)
    plt.close(figure)

    report = {"status": "temporal_diagnostics_complete"}
    report.update({name: switching_stats(actions) for name, actions in series.items()})
    (destination / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-predictions", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--closed-loop-episode", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = export_diagnostics(
        args.parent_predictions,
        args.candidate_predictions,
        args.closed_loop_episode,
        args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
