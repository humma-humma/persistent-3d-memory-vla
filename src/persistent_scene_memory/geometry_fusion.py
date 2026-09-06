"""Lightweight geometry cross-attention for the SmolVLA state token."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


GEOMETRY_VALUES = "observation.geometry.tokens"
GEOMETRY_MASK = "observation.geometry.mask"


class GeometryStateProjection(nn.Module):
    """Add masked geometry cross-attention to an existing state projection."""

    def __init__(
        self,
        base_projection: nn.Module,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_heads < 1:
            raise ValueError("input_dim, hidden_dim, and num_heads must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.base_projection = base_projection
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        self._geometry_values: torch.Tensor | None = None
        self._geometry_mask: torch.Tensor | None = None

    def set_context(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        if values.ndim != 3 or values.shape[-1] != self.geometry_encoder[0].normalized_shape[0]:
            raise ValueError("geometry values must have shape [batch, tokens, input_dim]")
        if mask.shape != values.shape[:2]:
            raise ValueError("geometry mask must have shape [batch, tokens]")
        self._geometry_values = values
        self._geometry_mask = mask.bool()

    def clear_context(self) -> None:
        self._geometry_values = None
        self._geometry_mask = None

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        projected = self.base_projection(state)
        if self._geometry_values is None or self._geometry_mask is None:
            return projected
        values = self._geometry_values.to(device=projected.device, dtype=projected.dtype)
        valid = self._geometry_mask.to(device=projected.device)
        if values.shape[0] != projected.shape[0]:
            raise ValueError("geometry and state batch sizes must match")
        encoded = self.geometry_encoder(values)
        empty = ~valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[empty, 0] = True
        encoded = encoded.clone()
        encoded[empty, 0] = 0
        query = projected.unsqueeze(1)
        attended, _ = self.cross_attention(
            query, encoded, encoded, key_padding_mask=~safe_valid, need_weights=False
        )
        attended[empty] = 0
        fused = self.output_norm(attended.squeeze(1))
        return projected + torch.tanh(self.fusion_scale) * fused


class GeometryConditionedSmolVLA(nn.Module):
    """Wrap SmolVLA and feed geometry through its existing state prefix token."""

    def __init__(self, policy: nn.Module, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.policy = policy
        model = policy.model
        base_projection = model.state_proj
        hidden_dim = int(base_projection.out_features)
        self.connector = GeometryStateProjection(
            base_projection, input_dim, hidden_dim, num_heads
        )
        model.state_proj = self.connector

    def _call_with_geometry(self, method, batch: dict, *args, **kwargs):
        if GEOMETRY_VALUES not in batch or GEOMETRY_MASK not in batch:
            raise KeyError(f"batch must contain {GEOMETRY_VALUES!r} and {GEOMETRY_MASK!r}")
        self.connector.set_context(batch[GEOMETRY_VALUES], batch[GEOMETRY_MASK])
        try:
            return method(batch, *args, **kwargs)
        finally:
            self.connector.clear_context()

    def forward(self, batch: dict, *args, **kwargs):
        return self._call_with_geometry(self.policy, batch, *args, **kwargs)

    def predict_action_chunk(self, batch: dict, *args, **kwargs):
        return self._call_with_geometry(
            self.policy.predict_action_chunk, batch, *args, **kwargs
        )

    def select_action(self, batch: dict, *args, **kwargs):
        return self._call_with_geometry(self.policy.select_action, batch, *args, **kwargs)

    def save_connector(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.connector.state_dict(), destination)
