"""Small, framework-neutral adapter from geometry tokens to a VLA input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .memory import MemoryToken


@dataclass(frozen=True)
class AdaptedTokens:
    values: np.ndarray
    mask: np.ndarray
    keys: tuple[str, ...]


class GeometryTokenAdapter:
    """Pad and normalize memory tokens for a learned VLA connector."""

    def __init__(self, max_tokens: int = 32, position_scale: float = 1.0):
        if max_tokens < 1 or position_scale <= 0:
            raise ValueError("max_tokens and position_scale must be positive")
        self.max_tokens = int(max_tokens)
        self.position_scale = float(position_scale)

    def encode(
        self,
        tokens: list[MemoryToken],
        frame_index: int,
        *,
        feature_size: int | None = None,
    ) -> AdaptedTokens:
        selected = tokens[: self.max_tokens]
        inferred_size = selected[0].feature.size if selected else 0
        if feature_size is None:
            feature_size = inferred_size
        if feature_size < 0:
            raise ValueError("feature_size must be non-negative")
        if any(token.feature.size != feature_size for token in selected):
            raise ValueError("all token features must have the same dimension")
        values = np.zeros((self.max_tokens, 5 + feature_size), dtype=np.float32)
        mask = np.zeros(self.max_tokens, dtype=bool)
        for row, token in enumerate(selected):
            values[row] = np.concatenate(
                [
                    token.position / self.position_scale,
                    [token.confidence, float(frame_index - token.last_seen)],
                    token.feature,
                ]
            )
            mask[row] = True
        return AdaptedTokens(values, mask, tuple(token.key for token in selected))

    def augment_instruction(self, instruction: str, tokens: list[MemoryToken]) -> str:
        """Expose geometry to text-conditioned VLAs without modifying their weights."""
        if not tokens:
            return instruction
        facts = "; ".join(
            f"{t.key} at ({t.position[0]:.3f}, {t.position[1]:.3f}, {t.position[2]:.3f}), confidence {t.confidence:.2f}"
            for t in tokens[: self.max_tokens]
        )
        return f"{instruction}\nWorld-aligned scene memory: {facts}."
