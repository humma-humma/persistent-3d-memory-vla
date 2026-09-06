import numpy as np
import pytest

from persistent_scene_memory.robocasa_skill_geometry import (
    RelativeWorldGeometryProvider,
    geometry_feature_vector,
)


def test_geometry_feature_vector_uses_relative_world_geometry():
    initial = {
        "eef_world": [1, 2, 3],
        "object_world": [2, 2, 2],
        "base_world": [0, 0, 0],
    }
    current = {
        "eef_world": [2, 3, 4],
        "object_world": [3, 2, 2],
        "base_world": [1, 0, 0],
        "grasped": True,
    }
    feature = geometry_feature_vector(current, initial)
    assert feature.shape == (16,)
    np.testing.assert_array_equal(feature[:3], [-1, 1, 2])
    np.testing.assert_array_equal(feature[3:6], [1, 0, 0])
    assert feature[-1] == 1


def test_geometry_feature_vector_rejects_missing_metadata():
    with pytest.raises(ValueError, match="requires"):
        geometry_feature_vector({}, {})


def test_relative_world_geometry_provider_has_source_agnostic_contract():
    provider = RelativeWorldGeometryProvider("persistent_memory")
    with pytest.raises(RuntimeError, match="reset"):
        provider.features({})
    initial = {
        "eef_world": [0, 0, 0],
        "object_world": [1, 0, 0],
        "base_world": [0, 1, 0],
    }
    provider.reset(initial)
    feature = provider.features({**initial, "grasped": False})
    assert provider.source == "persistent_memory"
    assert feature.shape == (16,)
