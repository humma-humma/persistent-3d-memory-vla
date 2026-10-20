"""Published memory-update rules re-implemented on the shared token interface.

Both baselines reuse ``PersistentSceneMemory`` for storage and query-time
confidence decay, so comparisons against ``ConsistencyGatedMemory`` vary only
the update rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .memory import GeometryObservation, MemoryToken, PersistentSceneMemory


class _RuleMemory:
    def __init__(self, *, confidence_decay: float):
        self.memory = PersistentSceneMemory(confidence_decay=confidence_decay)

    def tokens(self, frame_index: int, *, min_confidence: float = 0.0) -> list[MemoryToken]:
        return self.memory.tokens(frame_index, min_confidence=min_confidence)

    def export(self, frame_index: int, *, min_confidence: float = 0.0) -> dict:
        return self.memory.export(frame_index, min_confidence=min_confidence)

    def save(self, path: str | Path, frame_index: int, *, min_confidence: float = 0.0) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.export(frame_index, min_confidence=min_confidence), indent=2) + "\n",
            encoding="utf-8",
        )

    def _commit(self, observation: GeometryObservation, position, feature, confidence) -> MemoryToken:
        return self.memory.replace(
            GeometryObservation(
                key=observation.key,
                position=position,
                feature=feature,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                frame_index=observation.frame_index,
            )
        )


class ThresholdEMAMemory(_RuleMemory):
    """EchoVLA-style update: blend in new evidence only where the residual exceeds tau.

    EchoVLA (arXiv 2511.18112) updates voxels whose reconstruction error exceeds
    ``tau = 0.5`` by an exponential moving average with ``alpha = 0.2`` and keeps
    all other voxels unchanged. Here the residual is the world-frame position
    disagreement, so ``tau`` is expressed in metres.
    """

    def __init__(self, *, tau: float = 0.05, alpha: float = 0.2, confidence_decay: float = 0.02):
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        super().__init__(confidence_decay=confidence_decay)
        self.tau = float(tau)
        self.alpha = float(alpha)

    def update(self, observation: GeometryObservation) -> MemoryToken:
        previous = self.memory.token(observation.key, observation.frame_index)
        if previous is None:
            return self.memory.replace(observation)
        residual = float(np.linalg.norm(previous.position - observation.position))
        if residual > self.tau:
            position = (1.0 - self.alpha) * previous.position + self.alpha * observation.position
            feature = (1.0 - self.alpha) * previous.feature + self.alpha * observation.feature
        else:
            position, feature = previous.position, previous.feature
        confidence = max(previous.confidence, observation.confidence)
        return self._commit(observation, position, feature, confidence)


class ConfidenceWeightedMemory(_RuleMemory):
    """AtlasVLA-style TSDF fusion: running average weighted by observation confidence.

    AtlasVLA (arXiv 2608.06729) fuses ``M = (W * M + w * m) / (W + w)`` with
    ``w`` the depth confidence and accumulates ``W = lam * W + w``. It has no
    explicit rule for changed geometry; stale state is diluted only through
    ``lam`` and new weight.
    """

    def __init__(self, *, lam: float = 0.9, confidence_decay: float = 0.02):
        if not 0.0 < lam <= 1.0:
            raise ValueError("lam must be in (0, 1]")
        super().__init__(confidence_decay=confidence_decay)
        self.lam = float(lam)
        self._weights: dict[str, float] = {}

    def update(self, observation: GeometryObservation) -> MemoryToken:
        previous = self.memory.token(observation.key, observation.frame_index)
        weight = observation.confidence
        if previous is None:
            self._weights[observation.key] = weight
            return self.memory.replace(observation)
        accumulated = self._weights[observation.key]
        total = accumulated + weight
        if total == 0.0:
            position, feature = observation.position, observation.feature
        else:
            position = (accumulated * previous.position + weight * observation.position) / total
            feature = (accumulated * previous.feature + weight * observation.feature) / total
        self._weights[observation.key] = self.lam * accumulated + weight
        confidence = max(previous.confidence, observation.confidence)
        return self._commit(observation, position, feature, confidence)
