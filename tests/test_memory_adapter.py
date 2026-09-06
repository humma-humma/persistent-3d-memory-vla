import numpy as np

from persistent_scene_memory import GeometryObservation, GeometryTokenAdapter, PersistentSceneMemory


def test_adapter_pads_tokens_and_emits_mask():
    memory = PersistentSceneMemory()
    memory.update(GeometryObservation("cup", [1, 2, 3], [0.1, 0.2], 0.8, 2))

    encoded = GeometryTokenAdapter(max_tokens=3, position_scale=2).encode(memory.tokens(4), 4)

    assert encoded.values.shape == (3, 7)
    assert encoded.mask.tolist() == [True, False, False]
    assert encoded.keys == ("cup",)
    assert np.allclose(encoded.values[0, :3], [0.5, 1.0, 1.5])


def test_prompt_adapter_exposes_world_position_and_confidence():
    memory = PersistentSceneMemory()
    memory.update(GeometryObservation("cup", [1, 2, 3], [1], 0.8, 0))

    prompt = GeometryTokenAdapter().augment_instruction("Pick it up.", memory.tokens(0))

    assert "Pick it up." in prompt
    assert "cup at (1.000, 2.000, 3.000), confidence 0.80" in prompt


def test_adapter_preserves_declared_feature_width_without_tokens():
    encoded = GeometryTokenAdapter(max_tokens=2).encode([], 0, feature_size=3)

    assert encoded.values.shape == (2, 8)
    assert encoded.mask.tolist() == [False, False]
