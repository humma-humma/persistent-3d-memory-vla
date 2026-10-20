"""Track A: compare memory-update rules as the object-position source for the oracle.

The privileged pick-place oracle normally reads the object position from the
simulator. Here it reads it from a memory fed by simulated per-camera
observations, so task success measures how well each update rule tracks the
object under occlusion, scene change, spurious evidence, and pose drift.

Kept privileged, and identical across rules: object orientation for grasp
alignment, the initial object and cabinet poses used to plan placement, and
in-hand contact (gripper sensing). Only the object position is remembered.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np

from .baseline_memory import ConfidenceWeightedMemory, ThresholdEMAMemory
from .consistency_memory import ConsistencyGatedMemory
from .memory import GeometryObservation, PersistentSceneMemory
from .robocasa_actions import OraclePickPlaceCounterToCabinet, _blank_action

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
RULES = ("privileged", "current", "persistent", "echo", "atlas", "gated")
CONDITIONS = ("clean", "occluded", "moved", "spurious", "drift")


@dataclass
class Condition:
    """Observation schedule for one scene condition."""

    name: str
    occlude_from: int = 10**9
    occlude_to: int = 10**9
    glimpse_every: int = 0
    move_at: int | None = None
    move_offset: tuple[float, float] = (0.0, 0.0)
    stale_offset: tuple[float, float] | None = None
    spurious_every: int = 0
    spurious_offset: float = 0.25
    drift_per_frame: float = 0.0


def make_condition(name: str) -> Condition:
    if name == "clean":
        return Condition(name)
    if name == "occluded":
        # Seen at the start, then hidden for the whole approach and grasp.
        return Condition(name, occlude_from=5, occlude_to=10**9)
    if name == "moved":
        # Revisit: memory starts from a previous visit, when the object stood 12 cm
        # away; it is now hidden and sparse single-camera glimpses reveal where it is.
        # The scene itself is untouched, so the task is as solvable as "clean".
        return Condition(
            name, occlude_from=1, occlude_to=10**9, glimpse_every=6,
            stale_offset=(0.0, 0.12),
        )
    if name == "spurious":
        # No change, but confident single-camera outliers keep arriving.
        return Condition(name, occlude_from=5, occlude_to=10**9, glimpse_every=6, spurious_every=4)
    if name == "drift":
        return Condition(name, drift_per_frame=0.0004)
    raise ValueError(f"unknown condition {name!r}")


class ObservationModel:
    """Per-camera noisy object observations with scheduled occlusion and failures."""

    def __init__(self, condition: Condition, rng: np.random.Generator, image_size: int = 128):
        self.condition = condition
        self.rng = rng
        self.image_size = image_size
        directions = rng.normal(size=(len(CAMERAS), 3))
        directions[:, 2] = 0.0
        self.drift_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    def _visible(self, sim, camera: str, position: np.ndarray) -> tuple[bool, float]:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
        )

        world_to_camera = np.linalg.inv(get_camera_extrinsic_matrix(sim, camera))
        point = world_to_camera @ np.r_[position, 1.0]
        depth = float(point[2])
        if depth <= 0.05:
            return False, depth
        pixel = get_camera_intrinsic_matrix(sim, camera, self.image_size, self.image_size) @ point[:3]
        u, v = pixel[:2] / pixel[2]
        return bool(0 <= u < self.image_size and 0 <= v < self.image_size), depth

    def observe(self, sim, frame: int, true_position: np.ndarray) -> list[tuple[str, GeometryObservation]]:
        c = self.condition
        occluded = c.occlude_from <= frame < c.occlude_to
        glimpse_camera = None
        if occluded and c.glimpse_every and frame % c.glimpse_every == 0:
            glimpse_camera = CAMERAS[(frame // c.glimpse_every) % 2]  # alternate the two head cameras
        result = []
        for index, camera in enumerate(CAMERAS):
            if occluded and camera != glimpse_camera:
                continue
            visible, depth = self._visible(sim, camera, true_position)
            if not visible:
                continue
            sigma = 0.004 + 0.006 * depth
            seen = true_position
            if frame == 0 and c.stale_offset is not None:
                seen = true_position + np.r_[c.stale_offset, 0.0]  # the previous visit
            position = seen + self.rng.normal(scale=sigma, size=3)
            position = position + c.drift_per_frame * frame * self.drift_directions[index]
            confidence = float(np.clip(1.0 - sigma / 0.05, 0.05, 0.95))
            result.append((camera, GeometryObservation("obj", position, np.zeros(1), confidence, frame)))
        if occluded and c.spurious_every and frame % c.spurious_every == 2:
            # A confident misassociation from one camera: the object appears elsewhere.
            angle = self.rng.uniform(0, 2 * np.pi)
            position = true_position + c.spurious_offset * np.array([np.cos(angle), np.sin(angle), 0.0])
            result.append((CAMERAS[0], GeometryObservation("obj", position, np.zeros(1), 0.9, frame)))
        return result


class PositionEstimator:
    """One update rule exposed as `update(observations)` and `estimate(frame)`."""

    def __init__(self, rule: str):
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r}")
        self.rule = rule
        self.current = None
        self.memory = {
            "persistent": lambda: PersistentSceneMemory(change_threshold=0.05),
            "echo": lambda: ThresholdEMAMemory(tau=0.05, alpha=0.2),
            "atlas": lambda: ConfidenceWeightedMemory(lam=0.9),
            "gated": lambda: ConsistencyGatedMemory(change_threshold=0.05, evidence="views"),
        }.get(rule, lambda: None)()

    def update(self, observations: list[tuple[str, GeometryObservation]]) -> None:
        if self.rule == "current":
            self.current = (
                np.mean([obs.position for _, obs in observations], axis=0) if observations else None
            )
            return
        for view, observation in observations:
            if self.rule == "gated":
                self.memory.update(observation, view=view)
            elif self.memory is not None:
                self.memory.update(observation)

    def estimate(self, frame: int) -> np.ndarray | None:
        if self.rule == "current":
            return self.current
        tokens = self.memory.tokens(frame)
        return tokens[0].position.copy() if tokens else None


def apply_scene_change(task, condition: Condition, frame: int) -> None:
    """Move the object once, at the condition's scheduled frame."""
    if condition.move_at is None or frame != condition.move_at:
        return
    joint = task.objects["obj"].joints[0]
    qpos = np.array(task.sim.data.get_joint_qpos(joint))
    qpos[:2] += np.asarray(condition.move_offset)
    task.sim.data.set_joint_qpos(joint, qpos)
    task.sim.forward()


class GeometryChannel:
    """Geometry state for the VLA: remembered object position relative to the gripper.

    Returns [dx, dy, dz, valid]; zeros with valid = 0 when the rule has no estimate.
    Matches `robocasa_finetune.episode_geometry`, which supplies the same values from
    recorded positions at training time.
    """

    def __init__(self, rule: str, condition_name: str, seed: int):
        self.condition = make_condition(condition_name)
        self.estimator = PositionEstimator(rule)
        self.observer = ObservationModel(self.condition, np.random.default_rng(seed))
        self.target_errors = []

    def vector(self, task, frame: int) -> np.ndarray:
        apply_scene_change(task, self.condition, frame)
        robot, obj = task.robots[0], task.objects["obj"]
        eef = np.array(task.sim.data.site_xpos[robot.eef_site_id["right"]])
        true_position = np.array(task.sim.data.body_xpos[task.obj_body_id[obj.name]])
        if self.estimator.rule == "privileged" or task._check_grasp(robot.gripper["right"], obj):
            estimate = true_position
        else:
            self.estimator.update(self.observer.observe(task.sim, frame, true_position))
            estimate = self.estimator.estimate(frame)
        if estimate is None:
            return np.zeros(4, dtype=np.float32)
        self.target_errors.append(float(np.linalg.norm(estimate - true_position)))
        return np.r_[estimate - eef, 1.0].astype(np.float32)


class MemoryTargetOracle(OraclePickPlaceCounterToCabinet):
    """Oracle whose object position comes from an estimator instead of the simulator."""

    def __init__(self, estimator: PositionEstimator, observer: ObservationModel, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.observer = observer
        self.frame = 0
        self.target_errors = []
        self.blind_frames = 0
        self._cached_frame = None
        self._cached = None

    def _positions(self, task):
        # The parent act() asks again within the same frame; observe and update only once.
        if self._cached_frame == self.frame:
            return self._cached
        self._cached_frame = self.frame
        self._cached = self._observe_positions(task)
        return self._cached

    def _observe_positions(self, task):
        eef, true_position = super()._positions(task)
        if self.estimator.rule == "privileged":
            return eef, true_position
        grasped = bool(task._check_grasp(task.robots[0].gripper["right"], task.objects["obj"]))
        if grasped:
            # In hand, the object moves with the gripper; proprioception tracks it.
            return eef, true_position
        self.estimator.update(self.observer.observe(task.sim, self.frame, true_position))
        estimate = self.estimator.estimate(self.frame)
        if estimate is not None:
            self.target_errors.append(float(np.linalg.norm(estimate - true_position)))
        return eef, estimate

    def act(self, env, frame_index: int, observation):
        self.frame = frame_index
        task = env.unwrapped.env
        if self.initial_object_position is None:
            self._initialize(task, observation["state.base_position"])
        eef, estimate = self._positions(task)
        if estimate is None:
            # Nothing seen and nothing remembered: hold still.
            self.blind_frames += 1
            gripper = 1.0 if 2 <= self.phase_index <= 5 else 0.0
            self.last_info = {"phase": self.phase, "blind": True}
            return _blank_action(gripper)
        return super().act(env, frame_index, observation)


def run_episode(env, rule: str, condition_name: str, seed: int, steps: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    condition = make_condition(condition_name)
    observer = ObservationModel(condition, np.random.default_rng(seed))
    oracle = MemoryTargetOracle(PositionEstimator(rule), observer)
    observation, _ = env.reset(seed=seed)
    task = env.unwrapped.env
    success_frame = None
    for frame in range(steps):
        apply_scene_change(task, condition, frame)
        action = oracle.act(env, frame, observation)
        observation, _, terminated, truncated, info = env.step(action)
        if info.get("success", False):
            success_frame = frame
            break
        if terminated or truncated:
            break
    errors = oracle.target_errors
    return {
        "rule": rule,
        "condition": condition_name,
        "seed": seed,
        "success": success_frame is not None,
        "success_frame": success_frame,
        "blind_frames": oracle.blind_frames,
        "mean_target_error": float(np.mean(errors)) if errors else None,
        "max_target_error": float(np.max(errors)) if errors else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object", default="cream_cheese_stick:CreamCheeseStick001")
    parser.add_argument("--layout-style", default="1,1")
    parser.add_argument("--seeds", default="2000-2009", help="inclusive range A-B")
    parser.add_argument("--rules", default=",".join(RULES))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument(
        "--resume", action="store_true", help="keep results already in --output and skip them"
    )
    args = parser.parse_args(argv)

    import gymnasium as gym
    import robocasa  # noqa: F401

    from .robocasa_oracle_sweep import resolve_object

    layout, style = (int(value) for value in args.layout_style.split(","))
    env = gym.make(
        "robocasa/PickPlaceCounterToCabinet",
        split=None,
        obj_registries=("lightwheel",),
        obj_groups=resolve_object(args.object),
        layout_and_style_ids=[[layout, style]],
        camera_depths=False,
        camera_widths=64,
        camera_heights=64,
    )
    first, last = (int(value) for value in args.seeds.split("-"))
    results = []
    if args.resume and args.output.exists():
        results = json.loads(args.output.read_text(encoding="utf-8"))
    done = {(r["seed"], r["condition"], r["rule"]) for r in results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for seed in range(first, last + 1):
            for condition in args.conditions.split(","):
                for rule in args.rules.split(","):
                    if (seed, condition, rule) in done:
                        continue
                    result = run_episode(env, rule, condition, seed, args.steps)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
