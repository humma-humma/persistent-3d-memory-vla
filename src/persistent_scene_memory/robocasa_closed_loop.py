"""Run a RoboCasa episode controlled by a loopback SmolVLA policy server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import socket
import time

import numpy as np

from .policy_rpc import receive_message, send_message
from .robocasa_finetune import load_episode_vectors
from .robocasa_rollout import RoboCasaEpisode, _camera_pairs, record_episode
from .robocasa_smolvla_replay import ACTION_FIELDS


STATE_FIELDS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)


def policy_observation_images(observation: dict) -> list[np.ndarray]:
    """Return the three policy RGB views with or without paired depth streams."""
    cameras = _camera_pairs(observation)
    if cameras:
        keys = [mapped + "_image" for mapped, _ in cameras]
    else:
        keys = sorted(
            key
            for key in observation
            if key.startswith("video.")
            and not key.endswith("_depth")
            and not key.endswith("_image")
        )
    if len(keys) != 3:
        raise ValueError("RoboCasa policy observation must contain three RGB views")
    return [np.asarray(observation[key], dtype=np.uint8) for key in keys]


def live_state16(observation: dict) -> np.ndarray:
    state = np.concatenate(
        [np.asarray(observation[field], dtype=np.float32).reshape(-1) for field in STATE_FIELDS]
    )
    if state.shape != (16,):
        raise ValueError(f"expected live 16D state, received {state.shape}")
    return state


def action12_to_dict(action: np.ndarray, action_space, *, scale: float = 1.0) -> dict:
    """Unflatten and bound a policy action using the live environment contract."""
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError("policy action must be a finite 12D vector")
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    result, offset = {}, 0
    for field in ACTION_FIELDS:
        space = action_space.spaces[field]
        width = int(np.prod(space.shape))
        part = value[offset : offset + width].reshape(space.shape).copy()
        if field in {
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.base_motion",
        }:
            part *= scale
        result[field] = np.clip(part, space.low, space.high).astype(space.dtype)
        offset += width
    if offset != 12:
        raise ValueError("live action space does not match the 12D checkpoint contract")
    return result


def rollout_diagnostics(states: np.ndarray, actions: np.ndarray) -> dict:
    """Summarize whether a closed-loop policy moved and switched control modes."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 16 or actions.shape != (len(states), 12):
        raise ValueError("expected aligned states [N, 16] and actions [N, 12]")
    return {
        "base_displacement": float(np.linalg.norm(states[-1, :3] - states[0, :3])),
        "eef_relative_displacement": float(
            np.linalg.norm(states[-1, 7:10] - states[0, 7:10])
        ),
        "gripper_close_fraction": float(np.mean(actions[:, 6] > 0.5)),
        "base_control_fraction": float(np.mean(actions[:, 11] > 0.5)),
    }


class RemoteSmolVLA:
    def __init__(self, task: str, *, host: str, port: int, scale: float):
        self.task = task
        self.scale = scale
        self.connection = socket.create_connection((host, port), timeout=120)
        self.connection.settimeout(180)
        self.reset()

    def reset(self) -> None:
        send_message(self.connection, {"command": "reset"})
        response = receive_message(self.connection)
        if not response.get("ok"):
            raise RuntimeError(f"policy reset failed: {response}")
        self.last_info = {}

    def act(self, env, frame_index: int, observation: dict) -> dict:
        request = {
            "command": "act",
            "task": self.task,
            "state": live_state16(observation),
            "images": policy_observation_images(observation),
        }
        send_message(self.connection, request)
        response = receive_message(self.connection)
        if "error" in response:
            raise RuntimeError(f"policy server failed: {response['error']}")
        raw_action = np.asarray(response["action"], dtype=np.float32)
        action = action12_to_dict(raw_action, env.action_space, scale=self.scale)
        self.last_info = {
            "name": "smolvla_remote_closed_loop",
            "policy_action12": raw_action.tolist(),
            "action_scale": self.scale,
        }
        return action

    def close(self) -> None:
        try:
            send_message(self.connection, {"command": "shutdown"})
            receive_message(self.connection)
        finally:
            self.connection.close()


class OracleRecoveryPolicy:
    """Switch from a remote policy to a privileged recovery controller."""

    def __init__(self, remote, oracle, switch_frame: int):
        if switch_frame < 1:
            raise ValueError("switch_frame must be positive")
        self.remote = remote
        self.oracle = oracle
        self.switch_frame = switch_frame
        self.last_info = {}

    def act(self, env, frame_index: int, observation: dict) -> dict:
        source = self.remote if frame_index < self.switch_frame else self.oracle
        action = source.act(env, frame_index, observation)
        self.last_info = {
            **source.last_info,
            "recovery_mode": "policy" if source is self.remote else "oracle",
            "recovery_switch_frame": self.switch_frame,
        }
        return action

    def close(self) -> None:
        self.remote.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--object-group", default="kebab_skewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--oracle-recovery-after", type=int)
    args = parser.parse_args(argv)

    import gymnasium as gym
    import robocasa  # noqa: F401

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(
        "robocasa/PickPlaceCounterToCabinet",
        split="pretrain",
        obj_registries=("lightwheel",),
        obj_groups=args.object_group,
        camera_depths=True,
        camera_widths=args.camera_size,
        camera_heights=args.camera_size,
    )
    policy = None
    try:
        task = "Pick the kebab skewer from the counter and place it in the cabinet."
        remote = RemoteSmolVLA(
            task, host=args.host, port=args.port, scale=args.action_scale
        )
        if args.oracle_recovery_after is None:
            policy = remote
        else:
            from .robocasa_actions import OraclePickPlaceCounterToCabinet

            policy = OracleRecoveryPolicy(
                remote, OraclePickPlaceCounterToCabinet(), args.oracle_recovery_after
            )
        started = time.perf_counter()
        manifest = record_episode(
            env,
            args.output_dir / "episode",
            steps=args.steps,
            seed=args.seed,
            action_fn=policy,
            stop_on_success=True,
        )
        elapsed = time.perf_counter() - started
        states, actions = load_episode_vectors(
            RoboCasaEpisode(args.output_dir / "episode")
        )
        diagnostics = rollout_diagnostics(states, actions)
    finally:
        if policy is not None:
            policy.close()
        env.close()
    report = {
        "status": "closed_loop_complete",
        "seed": args.seed,
        "frames": len(manifest["frames"]),
        "success": any(frame["success"] for frame in manifest["frames"]),
        "elapsed_seconds": elapsed,
        "rgb_views": len(manifest["cameras"]),
        "state_dimension": 16,
        "action_dimension": 12,
        "action_scale": args.action_scale,
        "oracle_recovery_after": args.oracle_recovery_after,
        "episode": str((args.output_dir / "episode").resolve()),
        "diagnostics": diagnostics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "closed_loop_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
