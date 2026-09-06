"""Shared skill contract for the hierarchical RoboCasa VLA scaffold."""

from __future__ import annotations


SKILLS = (
    "approach_object",
    "grasp",
    "lift",
    "approach_cabinet",
    "insert",
    "release",
)

PHASE_TO_SKILL = {
    **{skill: skill for skill in SKILLS},
    "approach_object": "approach_object",
    "descend_to_object": "approach_object",
    "close_gripper": "grasp",
    "lift_object": "lift",
    "approach_cabinet": "approach_cabinet",
    "insert_object": "insert",
    "open_gripper": "release",
    "retreat": "release",
    "done": "release",
}


def phase_to_skill(phase: str) -> str:
    """Map a successful controller phase to the public six-skill contract."""
    try:
        return PHASE_TO_SKILL[phase]
    except KeyError as error:
        raise ValueError(f"unsupported controller phase: {phase!r}") from error


def monotonic_skill(current: str, proposed: str) -> str:
    """Reject backward or multi-stage jumps from a noisy high-level predictor."""
    if current not in SKILLS or proposed not in SKILLS:
        raise ValueError("current and proposed must use the hierarchical skill contract")
    current_index = SKILLS.index(current)
    proposed_index = SKILLS.index(proposed)
    if proposed_index == current_index + 1:
        return proposed
    return current
