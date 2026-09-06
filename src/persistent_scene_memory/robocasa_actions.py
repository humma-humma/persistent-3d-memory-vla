"""Action sources for locally generated RoboCasa rollouts."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = x * x + y * y + z * z + w * w
    if norm == 0:
        raise ValueError("base quaternion must be nonzero")
    scale = 2.0 / norm
    return np.array(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ]
    )


def _blank_action(gripper_close: float) -> dict[str, np.ndarray]:
    return {
        "action.gripper_close": np.array([gripper_close], dtype=np.float32),
        "action.end_effector_position": np.zeros(3, dtype=np.float32),
        "action.end_effector_rotation": np.zeros(3, dtype=np.float32),
        "action.base_motion": np.zeros(4, dtype=np.float32),
        "action.control_mode": np.zeros(1, dtype=np.float32),
    }


def _top_down_grasp_rotation(object_rotation: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Return a vertical gripper rotation that closes across an object's long axis."""
    object_rotation = np.asarray(object_rotation, dtype=np.float64)
    if object_rotation.shape != (3, 3):
        raise ValueError("object rotation must have shape (3, 3)")
    long_axis = object_rotation[:2, 1]
    norm = np.linalg.norm(long_axis)
    if norm < 1e-8:
        raise ValueError("object long axis has no horizontal projection")
    long_axis = long_axis / norm
    gripper_y = sign * np.array([-long_axis[1], long_axis[0], 0.0])
    gripper_z = np.array([0.0, 0.0, -1.0])
    gripper_x = np.cross(gripper_y, gripper_z)
    return np.column_stack((gripper_x, gripper_y, gripper_z))


def _grasp_orientation_action(
    current_relative: np.ndarray,
    base_rotation: np.ndarray,
    object_rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute the shortest base-frame OSC rotation command for a top-down grasp."""
    current = _quat_xyzw_to_matrix(current_relative)
    candidates = []
    for sign in (-1.0, 1.0):
        desired_world = _top_down_grasp_rotation(object_rotation, sign)
        desired_relative = base_rotation.T @ desired_world
        rotation_vector = Rotation.from_matrix(desired_relative @ current.T).as_rotvec()
        candidates.append(rotation_vector)
    rotation_vector = min(candidates, key=np.linalg.norm)
    return np.clip(rotation_vector / 0.5, -1.0, 1.0).astype(np.float32), float(
        np.linalg.norm(rotation_vector)
    )


def _cabinet_alignment_action(
    current_relative: np.ndarray,
    base_rotation: np.ndarray,
    cabinet_yaw: float,
    *,
    command_limit: float = 0.15,
) -> tuple[np.ndarray, float]:
    """Align a top-down gripper's x-axis with the cabinet's width."""
    if command_limit <= 0:
        raise ValueError("command_limit must be positive")
    current = _quat_xyzw_to_matrix(current_relative)
    width_axis = np.array([np.cos(cabinet_yaw), np.sin(cabinet_yaw), 0.0])
    candidates = []
    for sign in (-1.0, 1.0):
        gripper_x = sign * width_axis
        gripper_z = np.array([0.0, 0.0, -1.0])
        gripper_y = np.cross(gripper_z, gripper_x)
        desired_world = np.column_stack((gripper_x, gripper_y, gripper_z))
        desired_relative = base_rotation.T @ desired_world
        rotation_vector = Rotation.from_matrix(desired_relative @ current.T).as_rotvec()
        candidates.append(rotation_vector)
    rotation_vector = min(candidates, key=np.linalg.norm)
    return np.clip(
        rotation_vector / 0.5, -command_limit, command_limit
    ).astype(np.float32), float(np.linalg.norm(rotation_vector))


def _nearest_long_axis_grasp_point(
    object_position: np.ndarray,
    object_rotation: np.ndarray,
    end_effector_position: np.ndarray,
    max_offset: float = 0.08,
) -> np.ndarray:
    """Choose a reachable interior point along a slender object's horizontal axis."""
    if max_offset < 0:
        raise ValueError("max_offset must be non-negative")
    object_position = np.asarray(object_position, dtype=np.float64)
    end_effector_position = np.asarray(end_effector_position, dtype=np.float64)
    long_axis = np.asarray(object_rotation, dtype=np.float64)[:, 1].copy()
    long_axis[2] = 0.0
    norm = np.linalg.norm(long_axis)
    if norm < 1e-8:
        return object_position.copy()
    long_axis /= norm
    offset = np.clip(
        np.dot(end_effector_position - object_position, long_axis),
        -max_offset,
        max_offset,
    )
    return object_position + offset * long_axis


class OraclePickPlaceCounterToCabinet:
    """Privileged phase controller for generating pick-place action labels."""

    phases = (
        "approach_object",
        "descend_to_object",
        "close_gripper",
        "lift_object",
        "approach_cabinet",
        "insert_object",
        "open_gripper",
        "retreat",
        "done",
    )

    def __init__(
        self,
        *,
        position_gain: float = 1.0,
        arm_reach: float = 0.38,
        max_in_hand_rotation: float = 1.25,
        grasp_orientation_tolerance: float = 0.15,
    ):
        self.position_gain = float(position_gain)
        self.arm_reach = float(arm_reach)
        self.max_in_hand_rotation = float(max_in_hand_rotation)
        self.grasp_orientation_tolerance = float(grasp_orientation_tolerance)
        if self.max_in_hand_rotation <= 0 or self.grasp_orientation_tolerance <= 0:
            raise ValueError("rotation limits must be positive")
        self.phase_index = 0
        self.phase_steps = 0
        self.initial_object_position = None
        self.place_position = None
        self.cabinet_approach = None
        self.cabinet_base_stage = None
        self.grasp_loss_steps = 0
        self.align_for_cabinet = None
        self.last_info = {}

    @property
    def phase(self) -> str:
        return self.phases[self.phase_index]

    def _initialize(self, task, base_position: np.ndarray) -> None:
        obj = task.objects["obj"]
        self.initial_object_position = np.array(
            task.sim.data.body_xpos[task.obj_body_id[obj.name]], dtype=np.float64
        )
        regions = task.cab.get_int_sites(all_points=True, relative=False)
        if not regions:
            raise ValueError("cabinet exposes no interior placement region")
        points = min(regions.values(), key=lambda value: np.mean(np.asarray(value)[:, 2]))
        points = np.asarray(points, dtype=np.float64)
        center = points.mean(axis=0)
        nearest = points[np.argmin(np.linalg.norm(points[:, :2] - self.initial_object_position[:2], axis=1))]
        place = nearest + 0.40 * (center - nearest)
        place[2] = points[:, 2].min() + 0.10
        outward = self.initial_object_position[:2] - place[:2]
        outward /= max(np.linalg.norm(outward), 1e-8)
        self.place_position = place
        object_radius = float(getattr(obj, "horizontal_radius", 0.15))
        approach_clearance = object_radius + 0.15
        self.cabinet_approach = place + np.r_[approach_clearance * outward, 0.05]
        self.cabinet_base_stage = np.asarray(base_position, dtype=np.float64) + (
            self.cabinet_approach - self.initial_object_position
        )

    def _positions(self, task):
        robot = task.robots[0]
        obj = task.objects["obj"]
        eef = np.array(task.sim.data.site_xpos[robot.eef_site_id["right"]])
        obj_pos = np.array(task.sim.data.body_xpos[task.obj_body_id[obj.name]])
        return eef, obj_pos

    def _advance(self) -> None:
        self.phase_index = min(self.phase_index + 1, len(self.phases) - 1)
        self.phase_steps = 0

    def act(self, env, frame_index: int, observation) -> dict[str, np.ndarray]:
        task = env.unwrapped.env
        if self.initial_object_position is None:
            self._initialize(task, observation["state.base_position"])
        eef, obj_pos = self._positions(task)
        obj = task.objects["obj"]
        object_rotation = np.asarray(
            task.sim.data.body_xmat[task.obj_body_id[obj.name]], dtype=np.float64
        ).reshape(3, 3)
        base_rotation = _quat_xyzw_to_matrix(observation["state.base_rotation"])
        grasp_rotation_action, grasp_orientation_error = _grasp_orientation_action(
            observation["state.end_effector_rotation_relative"],
            base_rotation,
            object_rotation,
        )
        grasp_point = _nearest_long_axis_grasp_point(obj_pos, object_rotation, eef)
        grasped_now = bool(
            task._check_grasp(task.robots[0].gripper["right"], task.objects["obj"])
        )
        if task._check_success():
            self.phase_index = len(self.phases) - 1
            self.phase_steps = 0
        gripper_close = 1.0 if 2 <= self.phase_index <= 5 else 0.0

        while True:
            phase = self.phase
            if phase == "approach_object":
                target, tolerance = grasp_point + [0, 0, 0.14], 0.06
            elif phase == "descend_to_object":
                target, tolerance = grasp_point + [0, 0, 0.008], 0.012
            elif phase == "close_gripper":
                target, tolerance = eef, 0.0
                if self.phase_steps >= 12:
                    if grasped_now:
                        self.grasp_loss_steps = 0
                        self._advance()
                    else:
                        self.phase_index = 1
                        self.phase_steps = 0
                        gripper_close = 0.0
                    continue
            elif phase == "lift_object":
                self.grasp_loss_steps = 0 if grasped_now else self.grasp_loss_steps + 1
                if self.phase_steps > 4 and self.grasp_loss_steps >= 8:
                    self.phase_index = 1
                    self.phase_steps = 0
                    self.grasp_loss_steps = 0
                    gripper_close = 0.0
                    continue
                target = np.array(
                    [eef[0], eef[1], self.initial_object_position[2] + 0.28]
                )
                tolerance = 0.04
            elif phase == "approach_cabinet":
                self.grasp_loss_steps = 0 if grasped_now else self.grasp_loss_steps + 1
                if self.grasp_loss_steps >= 8:
                    gripper_close = 0.0
                    if obj_pos[2] > self.initial_object_position[2] + 0.05:
                        phase = "wait_for_object"
                        target, tolerance = eef, 0.0
                    else:
                        self.phase_index = 0
                        self.phase_steps = 0
                        continue
                else:
                    target, tolerance = self.cabinet_approach, 0.12
            elif phase == "insert_object":
                from robocasa.utils import object_utils as OU

                if OU.obj_inside_of(task, "obj", task.cab):
                    self._advance()
                    gripper_close = 0.0
                    continue
                self.grasp_loss_steps = 0 if grasped_now else self.grasp_loss_steps + 1
                if self.grasp_loss_steps >= 8:
                    gripper_close = 0.0
                    if obj_pos[2] > self.initial_object_position[2] + 0.05:
                        phase = "wait_for_object"
                        target, tolerance = eef, 0.0
                    else:
                        self.phase_index = 0
                        self.phase_steps = 0
                        continue
                else:
                    target, tolerance = self.place_position, 0.04
            elif phase == "open_gripper":
                target, tolerance = eef, 0.0
                if self.phase_steps >= 8:
                    self._advance()
                    continue
            elif phase == "retreat":
                target, tolerance = self.cabinet_approach + [0, 0, 0.08], 0.05
            else:
                if not task._check_success():
                    self.phase_index = 0
                    self.phase_steps = 0
                    gripper_close = 0.0
                    continue
                target, tolerance = eef, 0.0

            error_world = np.asarray(target) - eef
            distance = float(np.linalg.norm(error_world))
            if tolerance and distance <= tolerance:
                if (
                    phase == "approach_object"
                    and grasp_orientation_error > self.grasp_orientation_tolerance
                ):
                    break
                self._advance()
                gripper_close = 1.0 if 2 <= self.phase_index <= 5 else 0.0
                continue
            break

        error_base = base_rotation.T @ error_world
        action = _blank_action(gripper_close)
        carrying_phase = phase in {"lift_object", "approach_cabinet", "insert_object"}
        arm_limit = 0.2 if (grasped_now or carrying_phase) else 1.0
        action["action.end_effector_position"] = np.clip(
            self.position_gain * error_base / 0.05, -arm_limit, arm_limit
        ).astype(np.float32)
        orientation_error = 0.0
        if phase in {"approach_object", "descend_to_object", "close_gripper"}:
            orientation_error = grasp_orientation_error
            action["action.end_effector_rotation"] = grasp_rotation_action
        elif phase in {"approach_cabinet", "insert_object"}:
            rotation_action, orientation_error = _cabinet_alignment_action(
                observation["state.end_effector_rotation_relative"],
                base_rotation,
                float(task.cab.rot),
            )
            if self.align_for_cabinet is None:
                self.align_for_cabinet = orientation_error <= self.max_in_hand_rotation
            if self.align_for_cabinet:
                action["action.end_effector_rotation"] = rotation_action
        base_position = np.asarray(observation["state.base_position"], dtype=np.float64)
        base_to_target = base_rotation.T @ (np.asarray(target) - base_position)
        if phase == "approach_cabinet":
            base_to_stage = base_rotation.T @ (self.cabinet_base_stage - base_position)
            navigating = (
                self.phase_steps < 80
                and np.linalg.norm(base_to_stage[:2]) > 0.06
            )
        else:
            base_to_stage = base_to_target
            horizontal_threshold = (
                0.08
                if phase in {"approach_object", "descend_to_object", "insert_object"}
                else 0.25
            )
            minimum_base_distance = (
                0.18
                if phase in {"approach_object", "descend_to_object"}
                else self.arm_reach
            )
            navigating = (
                phase not in {"close_gripper", "open_gripper", "done"}
                and phase != "lift_object"
                and np.linalg.norm(error_world[:2]) > horizontal_threshold
                and np.linalg.norm(base_to_target[:2]) > minimum_base_distance
            )
        if navigating:
            base_limit = 0.1 if (grasped_now or carrying_phase) else 1.0
            if phase == "approach_cabinet":
                action["action.base_motion"][:2] = np.clip(
                    base_to_stage[:2] / 0.4, -base_limit, base_limit
                )
                action["action.end_effector_position"][:] = 0.0
            else:
                action["action.base_motion"][:2] = np.clip(
                    base_to_target[:2] / 0.25, -base_limit, base_limit
                )
        holding_object = bool(
            grasped_now
            or (
                phase in {"lift_object", "approach_cabinet", "insert_object"}
                and self.grasp_loss_steps < 8
            )
        )
        if not navigating and phase not in {"close_gripper", "open_gripper", "done"} and abs(error_world[2]) > 0.05:
            torso_limit = 0.02 if holding_object else 0.5
            action["action.base_motion"][3] = np.clip(
                error_world[2] / 0.3, -torso_limit, torso_limit
            )
        if np.any(action["action.base_motion"] != 0):
            action["action.control_mode"][:] = 1.0
        self.phase_steps += 1
        self.last_info = {
            "name": "oracle_pick_place_counter_to_cabinet",
            "phase": phase,
            "phase_step": self.phase_steps,
            "eef_world": eef.tolist(),
            "object_world": obj_pos.tolist(),
            "grasp_point_world": grasp_point.tolist(),
            "target_world": np.asarray(target).tolist(),
            "base_world": base_position.tolist(),
            "base_motion": action["action.base_motion"].tolist(),
            "position_error": distance,
            "orientation_error": orientation_error,
            "cabinet_alignment_enabled": self.align_for_cabinet,
            "grasped": grasped_now,
            "grasp_loss_steps": self.grasp_loss_steps,
        }
        return action
