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


def task_progress(env) -> dict:
    """Privileged pick-place progress for diagnostics only; never sent to the policy."""
    task = env.unwrapped.env
    obj = task.objects["obj"]
    eef = np.asarray(task.sim.data.site_xpos[task.robots[0].eef_site_id["right"]])
    obj_pos = np.asarray(task.sim.data.body_xpos[task.obj_body_id[obj.name]])
    return {
        "eef_world": eef.tolist(),
        "object_world": obj_pos.tolist(),
        "grasped": bool(task._check_grasp(task.robots[0].gripper["right"], obj)),
    }


def progress_summary(frames: list[dict]) -> dict:
    """How far a failed rollout got: reach, grasp, and lift of the target object."""
    rows = [frame.get("action_source", {}) for frame in frames]
    rows = [row for row in rows if "object_world" in row]
    if not rows:
        return {}
    eef = np.asarray([row["eef_world"] for row in rows])
    obj = np.asarray([row["object_world"] for row in rows])
    return {
        "min_eef_object_distance": float(np.min(np.linalg.norm(eef - obj, axis=1))),
        "grasp_fraction": float(np.mean([row["grasped"] for row in rows])),
        "max_object_lift": float(np.max(obj[:, 2] - obj[0, 2])),
        "max_object_travel": float(np.max(np.linalg.norm(obj - obj[0], axis=1))),
    }


class RemoteSmolVLA:
    def __init__(
        self,
        task: str,
        *,
        host: str,
        port: int,
        scale: float,
        shutdown_on_close: bool = True,
        seed: int | None = None,
        geometry=None,
    ):
        self.task = task
        self.geometry = geometry
        self.scale = scale
        self.shutdown_on_close = shutdown_on_close
        self.connection = socket.create_connection((host, port), timeout=120)
        self.connection.settimeout(180)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        request = {"command": "reset"}
        if seed is not None:
            request["seed"] = int(seed)
        send_message(self.connection, request)
        response = receive_message(self.connection)
        if not response.get("ok"):
            raise RuntimeError(f"policy reset failed: {response}")
        self.last_info = {}

    def act(self, env, frame_index: int, observation: dict) -> dict:
        state = live_state16(observation)
        if self.geometry is not None:
            state = np.concatenate([state, self.geometry.vector(env.unwrapped.env, frame_index)])
        request = {
            "command": "act",
            "task": self.task,
            "state": state,
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
            **({"geometry_state": state[16:].tolist()} if self.geometry is not None else {}),
            **task_progress(env),
        }
        return action

    def close(self) -> None:
        try:
            if self.shutdown_on_close:
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


class DaggerPolicy:
    """Mix policy and oracle execution; always record the oracle's action as the label.

    Both are queried every frame so the oracle's phase machine follows the visited
    states. Who executes is redrawn every `chunk` frames with oracle probability `beta`.
    """

    def __init__(self, remote, oracle, beta: float, seed: int, chunk: int = 10):
        if not 0.0 <= beta <= 1.0 or chunk < 1:
            raise ValueError("beta must be in [0, 1] and chunk positive")
        self.remote = remote
        self.oracle = oracle
        self.beta = beta
        self.chunk = chunk
        self.rng = np.random.default_rng(seed)
        self.use_oracle = False
        self.label_action = None
        self.last_info = {}

    def act(self, env, frame_index: int, observation: dict) -> dict:
        if frame_index % self.chunk == 0:
            self.use_oracle = bool(self.rng.random() < self.beta)
        policy_action = self.remote.act(env, frame_index, observation)
        self.label_action = dict(self.oracle.act(env, frame_index, observation))
        self.last_info = {
            **self.remote.last_info,
            "dagger_executor": "oracle" if self.use_oracle else "policy",
            "dagger_beta": self.beta,
            "oracle_phase": self.oracle.last_info.get("phase"),
            "oracle_position_error": self.oracle.last_info.get("position_error"),
            "oracle_orientation_error": self.oracle.last_info.get("orientation_error"),
        }
        return self.label_action if self.use_oracle else policy_action

    def close(self) -> None:
        self.remote.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument(
        "--object-group",
        default="kebab_skewer",
        help="object group name, or category:Instance for one instance",
    )
    parser.add_argument("--layout-style", help="fixed kitchen as LAYOUT,STYLE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--oracle-recovery-after", type=int)
    parser.add_argument(
        "--task-text",
        default="Pick the kebab skewer from the counter and place it in the cabinet.",
    )
    parser.add_argument(
        "--keep-server", action="store_true", help="leave the policy server running on exit"
    )
    parser.add_argument(
        "--geometry-source",
        help="memory rule feeding the 20D geometry state (privileged, current, persistent, echo, atlas, gated)",
    )
    parser.add_argument("--condition", default="clean", help="scene condition for --geometry-source")
    parser.add_argument(
        "--dagger-beta",
        type=float,
        help="record oracle labels; execute the oracle with this probability per 10-frame chunk",
    )
    args = parser.parse_args(argv)

    import gymnasium as gym
    import robocasa  # noqa: F401

    random.seed(args.seed)
    np.random.seed(args.seed)
    from .robocasa_oracle_sweep import resolve_object

    environment_kwargs = {
        "split": "pretrain",
        "obj_registries": ("lightwheel",),
        "obj_groups": resolve_object(args.object_group),
        "camera_depths": True,
        "camera_widths": args.camera_size,
        "camera_heights": args.camera_size,
    }
    if args.layout_style:
        layout, style = (int(value) for value in args.layout_style.split(","))
        # RoboCasa's named splits override explicit scene ids, so drop the split here.
        environment_kwargs["split"] = None
        environment_kwargs["layout_and_style_ids"] = [[layout, style]]
    env = gym.make("robocasa/PickPlaceCounterToCabinet", **environment_kwargs)
    policy = None
    geometry = None
    if args.geometry_source:
        from .memory_target_eval import GeometryChannel

        geometry = GeometryChannel(args.geometry_source, args.condition, args.seed)
    try:
        remote = RemoteSmolVLA(
            args.task_text,
            host=args.host,
            port=args.port,
            scale=args.action_scale,
            shutdown_on_close=not args.keep_server,
            seed=args.seed,
            geometry=geometry,
        )
        if args.dagger_beta is not None:
            from .robocasa_actions import OraclePickPlaceCounterToCabinet

            policy = DaggerPolicy(
                remote, OraclePickPlaceCounterToCabinet(), args.dagger_beta, args.seed
            )
        elif args.oracle_recovery_after is None:
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
        "state_dimension": 16 if geometry is None else 20,
        "geometry_source": args.geometry_source,
        "condition": args.condition if geometry is not None else None,
        "mean_target_error": float(np.mean(geometry.target_errors))
        if geometry is not None and geometry.target_errors
        else None,
        "action_dimension": 12,
        "action_scale": args.action_scale,
        "oracle_recovery_after": args.oracle_recovery_after,
        "dagger_beta": args.dagger_beta,
        "episode": str((args.output_dir / "episode").resolve()),
        "diagnostics": diagnostics,
        "progress": progress_summary(manifest["frames"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "closed_loop_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
