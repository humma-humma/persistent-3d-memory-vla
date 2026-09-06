import pytest

from persistent_scene_memory.robocasa_hierarchical import (
    SKILLS,
    monotonic_skill,
    phase_to_skill,
)


def test_phase_to_skill_collapses_controller_micro_phases():
    assert phase_to_skill("descend_to_object") == "approach_object"
    assert phase_to_skill("close_gripper") == "grasp"
    assert phase_to_skill("retreat") == "release"
    for skill in SKILLS:
        assert phase_to_skill(skill) == skill
    with pytest.raises(ValueError, match="unsupported"):
        phase_to_skill("human_hold")


def test_monotonic_skill_accepts_only_the_next_skill():
    assert monotonic_skill("grasp", "lift") == "lift"
    assert monotonic_skill("grasp", "grasp") == "grasp"
    assert monotonic_skill("grasp", "approach_object") == "grasp"
    assert monotonic_skill("grasp", "insert") == "grasp"
