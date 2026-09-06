import numpy as np
from scipy.spatial.transform import Rotation

from persistent_scene_memory.robocasa_actions import (
    OraclePickPlaceCounterToCabinet,
    _cabinet_alignment_action,
    _grasp_orientation_action,
    _nearest_long_axis_grasp_point,
    _quat_xyzw_to_matrix,
    _top_down_grasp_rotation,
)


def test_quaternion_rotation_maps_world_delta_to_base_frame():
    rotation = _quat_xyzw_to_matrix(np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)]))

    assert np.allclose(rotation @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rotation.T @ [0, 1, 0], [1, 0, 0])


def test_oracle_declares_complete_phase_sequence():
    oracle = OraclePickPlaceCounterToCabinet()

    assert oracle.phases == (
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


def test_oracle_rejects_nonpositive_in_hand_rotation_limit():
    with np.testing.assert_raises(ValueError):
        OraclePickPlaceCounterToCabinet(max_in_hand_rotation=0)
    with np.testing.assert_raises(ValueError):
        OraclePickPlaceCounterToCabinet(grasp_orientation_tolerance=0)


def test_top_down_grasp_closes_across_object_long_axis():
    rotation = _top_down_grasp_rotation(np.eye(3))

    assert np.allclose(rotation[:, 2], [0, 0, -1])
    assert np.isclose(np.dot(rotation[:, 1], [0, 1, 0]), 0)
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1)


def test_grasp_orientation_action_is_zero_at_desired_rotation():
    desired = _top_down_grasp_rotation(np.eye(3))
    quaternion = Rotation.from_matrix(desired).as_quat()

    action, error = _grasp_orientation_action(quaternion, np.eye(3), np.eye(3))

    assert np.allclose(action, 0)
    assert np.isclose(error, 0)


def test_cabinet_alignment_uses_width_axis_and_respects_command_limit():
    current = Rotation.from_euler("z", 0.8).as_matrix()
    current[:, 2] = [0, 0, -1]
    current[:, 1] = np.cross(current[:, 2], current[:, 0])
    quaternion = Rotation.from_matrix(current).as_quat()

    action, error = _cabinet_alignment_action(
        quaternion, np.eye(3), cabinet_yaw=0.0, command_limit=0.15
    )

    assert error > 0
    assert np.max(np.abs(action)) <= 0.15 + 1e-6


def test_cabinet_alignment_is_zero_when_already_width_aligned():
    desired = np.diag([1.0, -1.0, -1.0])
    quaternion = Rotation.from_matrix(desired).as_quat()

    action, error = _cabinet_alignment_action(
        quaternion, np.eye(3), cabinet_yaw=0.0
    )

    assert np.allclose(action, 0)
    assert np.isclose(error, 0)


def test_grasp_point_moves_only_along_object_long_axis():
    point = _nearest_long_axis_grasp_point(
        np.array([1.0, 2.0, 0.5]), np.eye(3), np.array([1.1, 2.2, 1.0])
    )

    assert np.allclose(point, [1.0, 2.08, 0.5])
