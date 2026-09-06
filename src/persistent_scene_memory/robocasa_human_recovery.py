"""Collect successful keyboard corrections from SmolVLA-visited RoboCasa states."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

from .robocasa_closed_loop import RemoteSmolVLA
from .robocasa_rollout import _slug


ACTION_FIELDS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)


class DiscardAttempt(RuntimeError):
    """Raised when the operator requests a reset before native success."""


def keyboard_input_to_action(input_action: dict, active_arm: str) -> dict[str, np.ndarray]:
    """Convert RoboSuite keyboard output to the live RoboCasa action contract."""
    arm = np.asarray(input_action[f"{active_arm}_delta"], dtype=np.float32)
    gripper = np.asarray(input_action[f"{active_arm}_gripper"], dtype=np.float32)
    base = np.asarray(input_action["base"], dtype=np.float32)
    torso = np.asarray(input_action["torso"], dtype=np.float32)
    base_mode = np.asarray(input_action["base_mode"], dtype=np.float32)
    if arm.shape != (6,) or base.shape != (3,) or torso.shape != (1,):
        raise ValueError("keyboard device returned an unexpected action shape")
    return {
        "action.end_effector_position": arm[:3],
        "action.end_effector_rotation": arm[3:],
        "action.gripper_close": np.asarray([(gripper.reshape(-1)[0] + 1.0) / 2.0], dtype=np.float32),
        "action.base_motion": np.concatenate((base, torso)).astype(np.float32),
        "action.control_mode": np.asarray([(base_mode.reshape(-1)[0] + 1.0) / 2.0], dtype=np.float32),
    }


def summarize_attempt(frames: list[dict], switch_frame: int) -> dict:
    modes = [frame.get("action_source", {}).get("recovery_mode") for frame in frames]
    success = any(frame.get("success", False) for frame in frames)
    policy_frames = modes.count("policy")
    human_frames = modes.count("human")
    return {
        "success": success,
        "accepted": bool(success and policy_frames and human_frames),
        "frames": len(frames),
        "policy_frames": policy_frames,
        "human_correction_frames": human_frames,
        "switch_frame": switch_frame,
    }


def _camera_names(observation: dict) -> list[str]:
    names = sorted(
        key
        for key in observation
        if key.startswith("video.")
        and not key.endswith("_depth")
        and not key.endswith("_image")
    )
    if len(names) != 3:
        raise ValueError("recovery collector requires three RGB cameras")
    return names


def _display(observation: dict, cameras: list[str], mode: str, frame_index: int, switch_frame: int) -> int:
    panels = [np.asarray(observation[camera], dtype=np.uint8) for camera in cameras]
    canvas = np.concatenate(panels, axis=1)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    canvas = cv2.resize(canvas, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    color = (80, 220, 80) if mode == "HUMAN" else (50, 190, 255)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (20, 20, 20), -1)
    cv2.putText(canvas, f"{mode} | frame {frame_index} | takeover {switch_frame}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    prompt = "T: take over now" if mode == "POLICY" else "Arrows move | B base/arm | Space gripper | Q discard"
    cv2.putText(canvas, prompt, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1)
    cv2.imshow("RoboCasa Human Recovery", canvas)
    return cv2.waitKey(1) & 0xFF


def _wait_screen(message: str, prompt: str = "Press ENTER to continue | Q to stop") -> bool:
    canvas = np.full((300, 900, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, message, (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (90, 220, 255), 2)
    cv2.putText(canvas, prompt, (35, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
    cv2.imshow("RoboCasa Human Recovery", canvas)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (10, 13):
            return True
        if key in (ord("q"), ord("Q"), 27):
            return False


def _status_screen(message: str) -> None:
    canvas = np.full((300, 900, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, message, (35, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (90, 220, 255), 2)
    cv2.imshow("RoboCasa Human Recovery", canvas)
    cv2.waitKey(1)


def _synchronize_keyboard(device, last_action: dict) -> None:
    device.start_control()
    device.grasp_states[device.active_robot][device.active_arm_index] = bool(
        np.asarray(last_action["action.gripper_close"]).reshape(-1)[0] >= 0.5
    )
    device.base_modes[device.active_robot] = False


def _keyboard_action(device) -> dict[str, np.ndarray]:
    input_action = device.input2action(mirror_actions=True)
    if input_action is None:
        raise DiscardAttempt("operator discarded recovery attempt")
    action_dict = deepcopy(input_action)
    active_robot = device.env.robots[device.active_robot]
    active_arm = device.active_arm
    controller = active_robot.part_controllers[active_arm]
    action_dict[active_arm] = input_action[
        f"{active_arm}_{'delta' if controller.input_type == 'delta' else 'abs'}"
    ]
    return keyboard_input_to_action(action_dict, active_arm)


def _frame_arrays(observation: dict, action: dict, cameras: list[str], include_rgb: bool) -> dict:
    arrays = {}
    for key, value in observation.items():
        if key.startswith("state."):
            arrays[f"state__{_slug(key)}"] = np.asarray(value)
    for key, value in action.items():
        arrays[f"action__{_slug(key)}"] = np.asarray(value)
    if include_rgb:
        for camera in cameras:
            arrays[f"rgb__{_slug(camera)}"] = np.asarray(observation[camera], dtype=np.uint8)
    return arrays


def collect_attempt(
    env,
    remote: RemoteSmolVLA,
    device,
    destination: Path,
    *,
    seed: int,
    switch_frame: int,
    steps: int,
    stride: int,
    horizon: int,
    action_scale: float,
) -> dict:
    _status_screen(f"Loading scene seed {seed} ...")
    observation, _ = env.reset(seed=seed)
    remote.reset()
    cameras = _camera_names(observation)
    frames: list[dict] = []
    buffered_arrays: list[dict] = []
    rgb_history: dict[int, dict[str, np.ndarray]] = {}
    last_action = {field: np.zeros(env.action_space[field].shape, dtype=np.float32) for field in ACTION_FIELDS}
    human_started = False
    discarded = False

    for frame_index in range(steps):
        mode = "HUMAN" if frame_index >= switch_frame else "POLICY"
        key = _display(observation, cameras, mode, frame_index, switch_frame)
        if mode == "POLICY" and key in (ord("t"), ord("T")):
            switch_frame = frame_index
            mode = "HUMAN"
        try:
            if mode == "HUMAN":
                if not human_started:
                    _synchronize_keyboard(device, last_action)
                    human_started = True
                action = _keyboard_action(device)
                source = {"name": "robocasa_keyboard_recovery", "recovery_mode": "human", "phase": "human_correction"}
            else:
                action = remote.act(env, frame_index, observation)
                source = {**remote.last_info, "recovery_mode": "policy", "phase": "policy_prefix"}
        except DiscardAttempt:
            discarded = True
            break
        action = {
            key: np.clip(value, env.action_space[key].low, env.action_space[key].high).astype(np.float32)
            if hasattr(env.action_space[key], "low")
            else np.asarray(value, dtype=np.float32)
            for key, value in action.items()
        }
        next_observation, reward, terminated, truncated, info = env.step(action)
        include_rgb = frame_index % stride == 0
        filename = f"frame_{frame_index:06d}.npz"
        buffered_arrays.append(_frame_arrays(observation, action, cameras, include_rgb))
        images = {
            f"rgb__{_slug(camera)}": np.asarray(observation[camera], dtype=np.uint8).copy()
            for camera in cameras
        }
        rgb_history[frame_index] = images
        rgb_history.pop(frame_index - horizon - 1, None)
        frame = {
            "frame_index": frame_index,
            "file": filename,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info.get("success", False)),
            "rgb_available": include_rgb,
            "action_source": {**source, "recovery_switch_frame": switch_frame},
        }
        frames.append(frame)
        observation = next_observation
        last_action = action
        if frame["success"] or terminated or truncated:
            break

    summary = summarize_attempt(frames, switch_frame)
    summary["discarded"] = discarded
    if summary["accepted"] and len(frames) >= horizon:
        final_anchor = len(frames) - horizon
        buffered_arrays[final_anchor].update(rgb_history[final_anchor])
        frames[final_anchor]["rgb_available"] = True
    manifest = {
        "format": "robocasa-rgbd-rollout-v1",
        "demonstration_id": f"human-recovery:{int(time.time())}:{seed}",
        "seed": seed,
        "task": str(observation.get("annotation.human.task_description", "")),
        "cameras": cameras,
        "state_keys": sorted(key for key in observation if key.startswith("state.")),
        "action_keys": list(ACTION_FIELDS),
        "frames": frames,
        "training_cache": {"stride": stride, "horizon": horizon},
        "human_recovery": summary,
        "action_scale": action_scale,
    }
    if summary["accepted"]:
        destination.mkdir(parents=True, exist_ok=False)
        for frame, arrays in zip(frames, buffered_arrays, strict=True):
            np.savez_compressed(destination / frame["file"], **arrays)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-successes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--policy-frames", type=int, nargs="+", default=(3, 5, 10, 20))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--camera-size", type=int, default=160)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--object-group", default="kebab_skewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--action-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.target_successes < 1 or args.max_attempts < args.target_successes:
        parser.error("max-attempts must be at least target-successes > 0")
    if min(args.policy_frames) < 1 or args.stride < 1 or args.horizon < 1:
        parser.error("policy-frames, stride, and horizon must be positive")

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robosuite.devices import Keyboard

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir = args.output_dir / "accepted"
    accepted_dir.mkdir(exist_ok=True)
    env = gym.make(
        "robocasa/PickPlaceCounterToCabinet",
        split="pretrain",
        obj_registries=("lightwheel",),
        obj_groups=args.object_group,
        camera_depths=False,
        camera_widths=args.camera_size,
        camera_heights=args.camera_size,
    )
    remote = RemoteSmolVLA(
        "Pick the kebab skewer from the counter and place it in the cabinet.",
        host=args.host,
        port=args.port,
        scale=args.action_scale,
    )
    device = Keyboard(env=env.unwrapped.env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    records = []
    accepted = 0
    try:
        for attempt in range(args.max_attempts):
            if accepted >= args.target_successes:
                break
            if not _wait_screen(
                f"Ready for attempt {attempt + 1}/{args.max_attempts}",
                "ENTER: load scene | Q: stop",
            ):
                break
            switch_frame = args.policy_frames[attempt % len(args.policy_frames)]
            destination = accepted_dir / f"episode_{accepted + 1:03d}_seed_{args.seed + attempt}"
            manifest = collect_attempt(
                env,
                remote,
                device,
                destination,
                seed=args.seed + attempt,
                switch_frame=switch_frame,
                steps=args.steps,
                stride=args.stride,
                horizon=args.horizon,
                action_scale=args.action_scale,
            )
            summary = manifest["human_recovery"]
            if summary["accepted"]:
                accepted += 1
            records.append({
                **summary,
                "seed": args.seed + attempt,
                "episode": str(destination.resolve()) if summary["accepted"] else None,
            })
            print(json.dumps(records[-1], indent=2), flush=True)
            report = {
                "status": "human_recovery_collection_in_progress",
                "accepted_episodes": accepted,
                "attempts": len(records),
                "target_successes": args.target_successes,
                "records": records,
            }
            (args.output_dir / "collection_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            if summary["accepted"]:
                if not _wait_screen("Success saved.", "ENTER: continue | Q: stop"):
                    break
            elif not _wait_screen("Attempt ended without success. Nothing was saved.", "ENTER: retry | Q: stop"):
                break
    finally:
        remote.close()
        env.close()
        cv2.destroyAllWindows()
    report = {
        "status": "human_recovery_collection_complete",
        "accepted_episodes": accepted,
        "attempts": len(records),
        "target_successes": args.target_successes,
        "records": records,
    }
    (args.output_dir / "collection_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
