"""Persistent memory that confirms scene changes across independent views."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .memory import GeometryObservation, MemoryToken, PersistentSceneMemory


@dataclass(frozen=True)
class UpdateDecision:
    key: str
    state: str
    evidence_views: int
    residual: float
    token: MemoryToken


@dataclass
class _PendingChange:
    position: np.ndarray
    feature: np.ndarray
    confidence: float
    frames: set[int]


class ConsistencyGatedMemory:
    """Retain on one disagreement and update after repeated view agreement."""

    def __init__(
        self,
        *,
        confidence_decay: float = 0.02,
        change_threshold: float = 0.25,
        confirmation_views: int = 2,
        candidate_threshold: float | None = None,
    ):
        if confirmation_views < 2:
            raise ValueError("confirmation_views must be at least two")
        if candidate_threshold is not None and candidate_threshold <= 0.0:
            raise ValueError("candidate_threshold must be positive")
        self.change_threshold = float(change_threshold)
        self.confirmation_views = int(confirmation_views)
        self.candidate_threshold = float(candidate_threshold or change_threshold)
        self.memory = PersistentSceneMemory(
            confidence_decay=confidence_decay,
            change_threshold=change_threshold,
        )
        self._pending: dict[str, _PendingChange] = {}

    def update(self, observation: GeometryObservation) -> UpdateDecision:
        previous = self.memory.token(observation.key, observation.frame_index)
        if previous is None:
            token = self.memory.replace(observation)
            return UpdateDecision(observation.key, "update", 1, 0.0, token)

        residual = float(np.linalg.norm(previous.position - observation.position))
        if residual <= self.change_threshold:
            self._pending.pop(observation.key, None)
            token = self.memory.update(observation)
            return UpdateDecision(observation.key, "retain", 0, residual, token)

        pending = self._pending.get(observation.key)
        if pending is None or np.linalg.norm(pending.position - observation.position) > self.candidate_threshold:
            pending = _PendingChange(
                observation.position.copy(),
                observation.feature.copy(),
                observation.confidence,
                {observation.frame_index},
            )
            self._pending[observation.key] = pending
        elif observation.frame_index not in pending.frames:
            weight = len(pending.frames)
            pending.position = (weight * pending.position + observation.position) / (weight + 1)
            pending.feature = (weight * pending.feature + observation.feature) / (weight + 1)
            pending.confidence = max(pending.confidence, observation.confidence)
            pending.frames.add(observation.frame_index)

        evidence = len(pending.frames)
        if evidence < self.confirmation_views:
            return UpdateDecision(observation.key, "uncertain", evidence, residual, previous)

        confirmed = GeometryObservation(
            key=observation.key,
            position=pending.position,
            feature=pending.feature,
            confidence=pending.confidence,
            frame_index=observation.frame_index,
        )
        token = self.memory.replace(confirmed)
        self._pending.pop(observation.key, None)
        return UpdateDecision(observation.key, "update", evidence, residual, token)

    def tokens(self, frame_index: int, *, min_confidence: float = 0.0) -> list[MemoryToken]:
        return self.memory.tokens(frame_index, min_confidence=min_confidence)

    def policy_tokens(
        self, frame_index: int, *, min_confidence: float = 0.0
    ) -> list[MemoryToken]:
        """Expose committed and uncertain candidates with an uncertainty feature."""
        committed = [
            MemoryToken(
                token.key,
                token.position,
                np.concatenate([token.feature, [0.0]]),
                token.confidence,
                token.last_seen,
            )
            for token in self.tokens(frame_index, min_confidence=min_confidence)
        ]
        candidates = [
            MemoryToken(
                f"{key}?candidate",
                pending.position.copy(),
                np.concatenate([pending.feature, [1.0]]),
                pending.confidence * len(pending.frames) / self.confirmation_views,
                max(pending.frames),
            )
            for key, pending in sorted(self._pending.items())
        ]
        return committed + candidates

    def export(self, frame_index: int, *, min_confidence: float = 0.0) -> dict:
        result = self.memory.export(frame_index, min_confidence=min_confidence)
        result["uncertain"] = [
            {
                "key": key,
                "position": pending.position.tolist(),
                "confidence": pending.confidence,
                "evidence_views": len(pending.frames),
            }
            for key, pending in sorted(self._pending.items())
        ]
        return result

    def save(
        self, path: str | Path, frame_index: int, *, min_confidence: float = 0.0
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.export(frame_index, min_confidence=min_confidence), indent=2)
            + "\n",
            encoding="utf-8",
        )
