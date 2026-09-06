"""Replay compact RoboCasa teleoperation files into the native training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .robocasa_finetune import episode_anchors
from .robocasa_rollout import _slug
from .robocasa_smolvla_replay import ACTION_FIELDS


CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
STATE_KEYS = (
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
    "robot0_gripper_qpos",
)


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


def reorder_teleop_actions(actions: np.ndarray) -> np.ndarray:
    """Map raw RoboSuite actions to the live 12D RoboCasa policy convention."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 12:
        raise ValueError("teleoperation actions must have shape [frames, 12]")
    reordered = np.concatenate(
        (
            actions[:, 0:3],
            actions[:, 3:6],
            (actions[:, 6:7] + 1.0) / 2.0,
            actions[:, 7:11],
            (actions[:, 11:12] + 1.0) / 2.0,
        ),
        axis=1,
    )
    return reordered.astype(np.float32)


def state16_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    values = [np.asarray(observation[key], dtype=np.float32) for key in STATE_KEYS]
    state = np.concatenate(values)
    if state.shape != (16,):
        raise ValueError(f"teleoperation observation produced state shape {state.shape}")
    return state


def teleop_phase(actions: np.ndarray, frame_index: int, horizon: int) -> str:
    """Label a causal window by the behavior it contributes to training."""
    window = np.asarray(actions[frame_index : frame_index + horizon])
    if np.any(np.abs(window[:, 7:11]) > 1e-5):
        return "human_base_motion"
    arm_moves = np.any(np.abs(window[:, :6]) > 1e-5)
    gripper_changes = len(window) > 1 and np.any(np.diff(window[:, 6]) != 0)
    return "human_manipulation" if arm_moves or gripper_changes else "human_hold"


def parse_env_config(value: str | dict) -> dict:
    """Decode collector metadata from both original and Windows-fixed files."""
    while isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("teleoperation env_info must decode to an object")
    return value


def _environment(config: dict, *, cameras: bool, image_size: int):
    import robocasa  # noqa: F401
    import robosuite

    kwargs = dict(config)
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=cameras,
        use_camera_obs=cameras,
    )
    if cameras:
        kwargs.update(
            camera_names=list(CAMERAS),
            camera_heights=image_size,
            camera_widths=image_size,
        )
    return robosuite.make(**kwargs)


def _restore(env, model_xml: str, state: np.ndarray) -> dict[str, np.ndarray]:
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    return env._get_observations(force_update=True)


def export_teleop_cache(
    hdf5_paths: list[str | Path],
    output_root: str | Path,
    *,
    stride: int = 20,
    horizon: int = 50,
    image_size: int = 96,
) -> dict:
    """Export successful raw teleoperation episodes with sparse RGB anchors."""
    import h5py

    if not hdf5_paths:
        raise ValueError("at least one teleoperation HDF5 path is required")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = []

    for source_index, source in enumerate(hdf5_paths):
        source = Path(source).resolve()
        stats_path = source.parent / "ep_stats.json"
        if not stats_path.exists() or not json.loads(stats_path.read_text())["success"]:
            raise ValueError(f"teleoperation episode is not marked successful: {source.parent}")
        with h5py.File(source, "r") as dataset:
            demos = list(dataset["data"])
            if len(demos) != 1:
                raise ValueError(f"expected one demonstration in {source}, found {len(demos)}")
            demo = dataset["data"][demos[0]]
            sim_states = np.asarray(demo["states"], dtype=np.float64)
            actions = reorder_teleop_actions(demo["actions"][:])
            model_xml = str(demo.attrs["model_file"])
            ep_meta = json.loads(demo.attrs["ep_meta"])
            config = parse_env_config(dataset["data"].attrs["env_info"])

        anchors = set(episode_anchors(len(actions), horizon=horizon, stride=stride))
        state_env = _environment(config, cameras=False, image_size=image_size)
        image_env = _environment(config, cameras=True, image_size=image_size)
        try:
            state_env.reset()
            state_env.reset_from_xml_string(model_xml)
            image_env.reset()
            image_env.reset_from_xml_string(model_xml)
            states = np.stack(
                [
                    state16_from_observation(_restore(state_env, model_xml, sim_state))
                    for sim_state in sim_states
                ]
            )
            images = {}
            for frame_index in sorted(anchors):
                observation = _restore(image_env, model_xml, sim_states[frame_index])
                images[frame_index] = {
                    camera: np.asarray(observation[f"{camera}_image"], dtype=np.uint8)
                    for camera in CAMERAS
                }
        finally:
            state_env.close()
            image_env.close()

        destination = output_root / f"episode_{source_index:06d}"
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
                    arrays[f"rgb__{_slug('video.' + camera)}"] = images[frame_index][camera]
            filename = f"frame_{frame_index:06d}.npz"
            np.savez_compressed(destination / filename, **arrays)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "file": filename,
                    "reward": float(frame_index == len(actions) - 1),
                    "terminated": frame_index == len(actions) - 1,
                    "truncated": False,
                    "success": frame_index == len(actions) - 1,
                    "action_source": {
                        "name": "robocasa_keyboard_teleoperation",
                        "phase": teleop_phase(actions, frame_index, horizon),
                    },
                }
            )
        manifest = {
            "format": "robocasa-rgbd-rollout-v1",
            "demonstration_id": f"robocasa-keyboard:{source.parent.name}",
            "task": ep_meta["lang"],
            "cameras": ["video." + camera for camera in CAMERAS],
            "state_keys": list(_split_state(states[0])),
            "action_keys": list(ACTION_FIELDS),
            "frames": frame_rows,
            "training_cache": {"stride": stride, "horizon": horizon},
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        base = actions[:, 7:11]
        records.append(
            {
                "source": str(source),
                "output": str(destination),
                "frames": len(actions),
                "anchors": len(anchors),
                "nonzero_base_frames": int(np.count_nonzero(np.any(np.abs(base) > 1e-5, axis=1))),
                "task": ep_meta["lang"],
            }
        )

    report = {
        "status": "teleoperation_training_cache_complete",
        "output_root": str(output_root),
        "episodes": records,
        "episode_count": len(records),
        "stride": stride,
        "horizon": horizon,
        "image_size": image_size,
        "action_order": list(ACTION_FIELDS),
    }
    (output_root / "import_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args(argv)
    report = export_teleop_cache(
        args.hdf5,
        args.output_root,
        stride=args.stride,
        horizon=args.horizon,
        image_size=args.image_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
