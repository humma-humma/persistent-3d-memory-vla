import numpy as np
import pytest

from persistent_scene_memory.robocasa_official_import import (
    infer_skill_phases,
    reorder_official_actions,
    select_episode_rows,
    skill_anchor_indices,
)


def test_official_actions_are_reordered_by_modality_names():
    modality = {
        "action": {
            "base_motion": {"original_key": "action", "start": 0, "end": 4},
            "control_mode": {"original_key": "action", "start": 4, "end": 5},
            "end_effector_position": {"original_key": "action", "start": 5, "end": 8},
            "end_effector_rotation": {"original_key": "action", "start": 8, "end": 11},
            "gripper_close": {"original_key": "action", "start": 11, "end": 12},
        }
    }

    official = np.array([[0, 0, 0, 0, -1, 5, 6, 7, 8, 9, 10, 1]])
    result = reorder_official_actions(official, modality)

    assert result.tolist() == [[5, 6, 7, 8, 9, 10, 1, 0, 0, 0, 0, 0]]


def test_episode_selection_prioritizes_relevant_unique_tasks():
    rows = [
        {"episode_index": 0, "tasks": ["Pick the apple from the counter."]},
        {"episode_index": 1, "tasks": ["Pick the rolling pin from the counter."]},
        {"episode_index": 2, "tasks": ["Pick the apple from the counter."]},
        {"episode_index": 3, "tasks": ["Pick the milk from the counter."]},
    ]

    selected = select_episode_rows(rows, 2)

    assert [row["episode_index"] for row in selected] == [1, 3]
    assert [row["episode_index"] for row in select_episode_rows(rows, 4)] == [1, 3, 0, 2]
    with pytest.raises(ValueError, match="only 4 episodes"):
        select_episode_rows(rows, 5)


def test_skill_phases_follow_grasp_and_release_events():
    actions = np.zeros((30, 12), dtype=np.float32)
    actions[8:25, 6] = 1.0

    phases = infer_skill_phases(actions)

    assert phases[7] == "approach_object"
    assert phases[8] == "grasp"
    assert phases[24] == "insert"
    assert phases[25] == "release"
    assert set(phases) == {
        "approach_object",
        "grasp",
        "lift",
        "approach_cabinet",
        "insert",
        "release",
    }
    anchors = skill_anchor_indices(phases)
    assert 6 <= len(anchors) <= 18
    assert {phases[index] for index in anchors} == set(phases)
