import pytest

from persistent_scene_memory.fine_tune_pilot import (
    DEFAULT_TRAINING_EPISODES,
    HELD_OUT_EPISODES,
    TRAINABLE_PARTS,
    balanced_example_order,
    select_adaptation_parameters,
    select_projection_parameters,
)


class Parameter:
    def __init__(self):
        self.requires_grad = True


class Policy:
    def __init__(self):
        self.parameters = {
            "model.state_proj.weight": Parameter(),
            "model.vlm.encoder.weight": Parameter(),
        }

    def named_parameters(self):
        return self.parameters.items()


def test_projection_selection_freezes_backbone():
    policy = Policy()

    selected, names = select_projection_parameters(policy)

    assert TRAINABLE_PARTS
    assert names == ["model.state_proj.weight"]
    assert selected == [policy.parameters["model.state_proj.weight"]]
    assert not policy.parameters["model.vlm.encoder.weight"].requires_grad


def test_default_training_episodes_exclude_frozen_evaluation_set():
    assert set(DEFAULT_TRAINING_EPISODES).isdisjoint(HELD_OUT_EPISODES)
    assert set(DEFAULT_TRAINING_EPISODES) | set(HELD_OUT_EPISODES) == set(range(50))


def test_balanced_order_is_deterministic_and_covers_examples_before_repeating():
    first = balanced_example_order(5, 8, seed=3)
    second = balanced_example_order(5, 8, seed=3)

    assert first == second
    assert sorted(first[:5]) == list(range(5))
    assert len(first) == 8
    with pytest.raises(ValueError, match="positive"):
        balanced_example_order(0, 1, seed=3)


def test_adaptation_selection_can_unfreeze_only_final_expert_layer():
    policy = Policy()
    policy.parameters.update(
        {
            "model.vlm_with_expert.lm_expert.layers.14.mlp.weight": Parameter(),
            "model.vlm_with_expert.lm_expert.layers.15.mlp.weight": Parameter(),
            "model.vlm_with_expert.lm_expert.norm.weight": Parameter(),
        }
    )

    _, names = select_adaptation_parameters(policy, expert_layers=1)

    assert "model.state_proj.weight" in names
    assert "model.vlm_with_expert.lm_expert.layers.15.mlp.weight" in names
    assert "model.vlm_with_expert.lm_expert.norm.weight" in names
    assert "model.vlm_with_expert.lm_expert.layers.14.mlp.weight" not in names
    assert not policy.parameters[
        "model.vlm_with_expert.lm_expert.layers.14.mlp.weight"
    ].requires_grad
