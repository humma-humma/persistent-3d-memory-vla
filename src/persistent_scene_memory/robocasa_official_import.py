"""Create a compact native-action training cache from an official RoboCasa dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .robocasa_finetune import episode_anchors
from .robocasa_hierarchical import SKILLS
from .robocasa_rollout import _slug
from .robocasa_smolvla_replay import ACTION_FIELDS


CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
PRIORITY_WORDS = (
    "rolling pin",
    "tongs",
    "baguette",
    "wooden spoon",
    "ladle",
    "whisk",
    "peeler",
    "carrot",
    "corn",
    "water bottle",
    "boxed drink",
    "boxed food",
    "milk",
)


def reorder_official_actions(actions: np.ndarray, modality: dict) -> np.ndarray:
    """Reorder fields and map official -1/1 binary controls to live 0/1 controls."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError("official actions must have shape [frames, 12]")
    parts = []
    for field in ACTION_FIELDS:
        name = field.removeprefix("action.")
        spec = modality["action"][name]
        if spec["original_key"] != "action":
            raise ValueError(f"unexpected original action key for {name}")
        parts.append(actions[:, spec["start"] : spec["end"]])
    reordered = np.concatenate(parts, axis=1)
    if reordered.shape[1] != 12:
        raise ValueError("modality action slices do not form a 12D action")
    reordered[:, 6] = (reordered[:, 6] + 1.0) / 2.0
    reordered[:, 11] = (reordered[:, 11] + 1.0) / 2.0
    return reordered


def select_episode_rows(rows: list[dict], count: int) -> list[dict]:
    """Prioritize relevant task diversity, then fill with additional episodes."""
    if count < 1:
        raise ValueError("count must be positive")
    unique = {}
    for row in rows:
        task = row["tasks"][0]
        unique.setdefault(task, row)
    ranked = sorted(
        unique.values(),
        key=lambda row: (
            0 if any(word in row["tasks"][0].lower() for word in PRIORITY_WORDS) else 1,
            row["episode_index"],
        ),
    )
    if count > len(rows):
        raise ValueError(f"requested {count} episodes but only {len(rows)} episodes exist")
    selected = ranked[:count]
    if len(selected) < count:
        selected_ids = {int(row["episode_index"]) for row in selected}
        remaining = sorted(
            (row for row in rows if int(row["episode_index"]) not in selected_ids),
            key=lambda row: int(row["episode_index"]),
        )
        selected.extend(remaining[: count - len(selected)])
    return selected


def infer_skill_phases(actions: np.ndarray) -> list[str]:
    """Derive event-aligned six-skill supervision from a successful trajectory.

    Official atomic demonstrations do not carry composite-task subtask labels. The
    gripper command supplies two reliable semantic landmarks (grasp and release);
    the carrying interval is split into lift, transport, and insertion portions.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) < len(SKILLS):
        raise ValueError("actions must have shape [frames, 12] with at least six frames")
    closed = actions[:, 6] >= 0.5
    closing = np.flatnonzero(closed & ~np.r_[False, closed[:-1]])
    if not len(closing):
        raise ValueError("successful trajectory has no gripper-close event")
    grasp_start = int(closing[0])
    opening = np.flatnonzero((~closed) & np.r_[False, closed[:-1]])
    opening = opening[opening > grasp_start]
    release_start = int(opening[-1]) if len(opening) else len(actions) - 1
    if release_start - grasp_start < 4:
        raise ValueError("grasp and release events are too close to segment")

    carry = release_start - grasp_start
    grasp_end = min(release_start - 3, grasp_start + max(1, round(0.08 * carry)))
    remaining = release_start - grasp_end
    lift_end = grasp_end + max(1, round(0.22 * remaining))
    approach_end = lift_end + max(1, round(0.55 * remaining))
    lift_end = min(lift_end, release_start - 2)
    approach_end = min(max(approach_end, lift_end + 1), release_start - 1)

    phases = []
    for index in range(len(actions)):
        if index < grasp_start:
            phases.append("approach_object")
        elif index < grasp_end:
            phases.append("grasp")
        elif index < lift_end:
            phases.append("lift")
        elif index < approach_end:
            phases.append("approach_cabinet")
        elif index < release_start:
            phases.append("insert")
        else:
            phases.append("release")
    return phases


def skill_anchor_indices(phases: list[str]) -> set[int]:
    """Keep the start, midpoint, and end of every represented skill segment."""
    anchors = set()
    for skill in SKILLS:
        indices = [index for index, phase in enumerate(phases) if phase == skill]
        if indices:
            anchors.update((indices[0], indices[len(indices) // 2], indices[-1]))
    return anchors


def _video_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    result = {}
    frame_index = 0
    try:
        while frame_index <= max(indices):
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in indices:
                result[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_index += 1
    finally:
        capture.release()
    missing = indices - result.keys()
    if missing:
        raise ValueError(f"video {path} is missing requested frames {sorted(missing)}")
    return result


def _split_state(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "state.base_position": state[:3],
        "state.base_rotation": state[3:7],
        "state.end_effector_position_relative": state[7:10],
        "state.end_effector_rotation_relative": state[10:14],
        "state.gripper_qpos": state[14:16],
    }


def _split_action(action: np.ndarray) -> dict[str, np.ndarray]:
    widths = (3, 3, 1, 4, 1)
    result, offset = {}, 0
    for field, width in zip(ACTION_FIELDS, widths):
        result[field] = action[offset : offset + width]
        offset += width
    return result


def export_training_cache(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    episode_count: int = 32,
    stride: int = 20,
    horizon: int = 50,
) -> dict:
    """Export selected official episodes in the existing portable training contract."""
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    modality = json.loads((dataset_root / "meta" / "modality.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    selected = select_episode_rows(rows, episode_count)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []

    for row in selected:
        episode_index = int(row["episode_index"])
        name = f"episode_{episode_index:06d}"
        parquet = dataset_root / "data" / "chunk-000" / f"{name}.parquet"
        table = pd.read_parquet(parquet)
        states = np.stack(table["observation.state"]).astype(np.float32)
        actions = reorder_official_actions(np.stack(table["action"]), modality)
        phases = infer_skill_phases(actions)
        done = np.asarray(table["next.done"], dtype=bool)
        reward = np.asarray(table["next.reward"], dtype=np.float32)
        if not done[-1] or reward[-1] <= 0:
            raise ValueError(f"official episode {episode_index} is not successful")
        anchors = set(episode_anchors(len(table), horizon=horizon, stride=stride))
        anchors.update(skill_anchor_indices(phases))
        decoded = {
            camera: _video_frames(
                dataset_root
                / "videos"
                / "chunk-000"
                / f"observation.images.{camera}"
                / f"{name}.mp4",
                anchors,
            )
            for camera in CAMERAS
        }
        destination = output_root / name
        destination.mkdir(parents=True, exist_ok=True)
        frame_rows = []
        for frame_index, (state, action) in enumerate(zip(states, actions)):
            arrays = {}
            for field, value in _split_state(state).items():
                arrays[f"state__{_slug(field)}"] = value
            for field, value in _split_action(action).items():
                arrays[f"action__{_slug(field)}"] = value
            if frame_index in anchors:
                for camera in CAMERAS:
                    arrays[f"rgb__{_slug('video.' + camera)}"] = decoded[camera][frame_index]
            filename = f"frame_{frame_index:06d}.npz"
            np.savez_compressed(destination / filename, **arrays)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "file": filename,
                    "reward": float(reward[frame_index]),
                    "terminated": bool(done[frame_index]),
                    "truncated": False,
                    "success": bool(done[frame_index] and reward[frame_index] > 0),
                    "rgb_available": frame_index in anchors,
                    "action_source": {
                        "name": "official_robocasa_human",
                        "phase": phases[frame_index],
                        "phase_source": "gripper_event_segmentation_v1",
                    },
                }
            )
        manifest = {
            "format": "robocasa-rgbd-rollout-v1",
            "demonstration_id": f"robocasa-pretrain-human:{episode_index}",
            "task": row["tasks"][0],
            "cameras": ["video." + camera for camera in CAMERAS],
            "state_keys": list(_split_state(states[0])),
            "action_keys": list(ACTION_FIELDS),
            "frames": frame_rows,
            "training_cache": {"stride": stride, "horizon": horizon},
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        records.append(
            {
                "episode_index": episode_index,
                "task": row["tasks"][0],
                "frames": len(table),
                "anchors": len(anchors),
                "output": str(destination),
            }
        )
    report = {
        "status": "official_training_cache_complete",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "episodes": records,
        "episode_count": len(records),
        "stride": stride,
        "horizon": horizon,
        "action_order": list(ACTION_FIELDS),
        "phase_contract": list(SKILLS),
        "phase_source": "gripper_event_segmentation_v1",
    }
    (output_root / "import_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episode-count", type=int, default=32)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args(argv)
    report = export_training_cache(
        args.dataset_root,
        args.output_root,
        episode_count=args.episode_count,
        stride=args.stride,
        horizon=args.horizon,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
