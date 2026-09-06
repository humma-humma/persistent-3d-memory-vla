import pytest

torch = pytest.importorskip("torch")
from torch import nn

from persistent_scene_memory.geometry_fusion import GeometryStateProjection


def test_geometry_projection_preserves_shape_and_ignores_padding():
    projection = GeometryStateProjection(nn.Linear(3, 8), 4, 8, num_heads=2)
    state = torch.ones(2, 3)
    values = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    projection.set_context(values, mask)

    result = projection(state)

    assert result.shape == (2, 8)
    assert torch.isfinite(result).all()
    assert torch.allclose(result[1], projection.base_projection(state)[1])


def test_geometry_projection_requires_matching_shapes():
    projection = GeometryStateProjection(nn.Linear(3, 8), 4, 8, num_heads=2)

    with pytest.raises(ValueError, match="geometry mask"):
        projection.set_context(torch.zeros(1, 2, 4), torch.ones(1, 3))
