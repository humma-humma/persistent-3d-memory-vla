"""Confidence-aware world-frame scene memory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _vector(value: np.ndarray, *, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and result.size != size:
        raise ValueError(f"expected {size} values, got {result.size}")
    if not np.all(np.isfinite(result)):
        raise ValueError("vectors must contain only finite values")
    return result


def _confidence(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class GeometryObservation:
    """One world-frame observation to insert into memory."""

    key: str
    position: np.ndarray
    feature: np.ndarray
    confidence: float
    frame_index: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        object.__setattr__(self, "position", _vector(self.position, size=3))
        object.__setattr__(self, "feature", _vector(self.feature))
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class MemoryToken:
    """A compact persistent token exposed to a downstream policy."""

    key: str
    position: np.ndarray
    feature: np.ndarray
    confidence: float
    last_seen: int


class PersistentSceneMemory:
    """Fuse consistent observations and decay unseen world-frame tokens."""

    def __init__(self, *, confidence_decay: float = 0.02, change_threshold: float = 0.25):
        if not 0.0 <= confidence_decay <= 1.0:
            raise ValueError("confidence_decay must be in [0, 1]")
        if change_threshold <= 0.0:
            raise ValueError("change_threshold must be positive")
        self.confidence_decay = float(confidence_decay)
        self.change_threshold = float(change_threshold)
        self._tokens: dict[str, MemoryToken] = {}

    def _decayed_confidence(self, token: MemoryToken, frame_index: int) -> float:
        if frame_index < token.last_seen:
            raise ValueError("frame_index cannot precede a stored observation")
        age = frame_index - token.last_seen
        return token.confidence * (1.0 - self.confidence_decay) ** age

    def token(self, key: str, frame_index: int) -> MemoryToken | None:
        """Return one token with confidence decayed to the requested frame."""
        token = self._tokens.get(key)
        if token is None:
            return None
        return MemoryToken(
            key=token.key,
            position=token.position.copy(),
            feature=token.feature.copy(),
            confidence=self._decayed_confidence(token, frame_index),
            last_seen=token.last_seen,
        )

    def replace(self, observation: GeometryObservation) -> MemoryToken:
        """Commit a previously validated scene change without further gating."""
        previous = self._tokens.get(observation.key)
        if previous is not None and previous.feature.size != observation.feature.size:
            raise ValueError("feature dimensions must remain constant for a key")
        token = MemoryToken(
            key=observation.key,
            position=observation.position.copy(),
            feature=observation.feature.copy(),
            confidence=observation.confidence,
            last_seen=observation.frame_index,
        )
        self._tokens[observation.key] = token
        return token

    def update(self, observation: GeometryObservation) -> MemoryToken:
        """Insert an observation, fusing it unless it indicates scene change."""
        previous = self._tokens.get(observation.key)
        if previous is None:
            token = MemoryToken(
                key=observation.key,
                position=observation.position.copy(),
                feature=observation.feature.copy(),
                confidence=observation.confidence,
                last_seen=observation.frame_index,
            )
        else:
            if previous.feature.size != observation.feature.size:
                raise ValueError("feature dimensions must remain constant for a key")
            old_confidence = self._decayed_confidence(previous, observation.frame_index)
            displacement = np.linalg.norm(previous.position - observation.position)
            if displacement > self.change_threshold and observation.confidence >= old_confidence:
                position = observation.position.copy()
                feature = observation.feature.copy()
                confidence = observation.confidence
            else:
                total = old_confidence + observation.confidence
                if total == 0.0:
                    position = observation.position.copy()
                    feature = observation.feature.copy()
                else:
                    position = (
                        old_confidence * previous.position
                        + observation.confidence * observation.position
                    ) / total
                    feature = (
                        old_confidence * previous.feature
                        + observation.confidence * observation.feature
                    ) / total
                confidence = 1.0 - (1.0 - old_confidence) * (1.0 - observation.confidence)
            token = MemoryToken(
                key=observation.key,
                position=position,
                feature=feature,
                confidence=confidence,
                last_seen=observation.frame_index,
            )
        self._tokens[observation.key] = token
        return token

    def tokens(self, frame_index: int, *, min_confidence: float = 0.0) -> list[MemoryToken]:
        """Return confidence-sorted tokens with decay applied at query time."""
        threshold = _confidence(min_confidence)
        result = []
        for token in self._tokens.values():
            confidence = self._decayed_confidence(token, frame_index)
            if confidence >= threshold:
                result.append(
                    MemoryToken(
                        key=token.key,
                        position=token.position.copy(),
                        feature=token.feature.copy(),
                        confidence=confidence,
                        last_seen=token.last_seen,
                    )
                )
        return sorted(result, key=lambda token: (-token.confidence, token.key))

    def token_matrix(self, frame_index: int, *, min_confidence: float = 0.0) -> np.ndarray:
        """Encode tokens as [xyz, confidence, age, feature...] rows."""
        tokens = self.tokens(frame_index, min_confidence=min_confidence)
        if not tokens:
            return np.empty((0, 0), dtype=np.float64)
        feature_size = tokens[0].feature.size
        if any(token.feature.size != feature_size for token in tokens):
            raise ValueError("all exported token features must have the same dimension")
        return np.stack(
            [
                np.concatenate(
                    [
                        token.position,
                        [token.confidence, float(frame_index - token.last_seen)],
                        token.feature,
                    ]
                )
                for token in tokens
            ]
        )

    def export(self, frame_index: int, *, min_confidence: float = 0.0) -> dict:
        """Return a JSON-serializable snapshot with stable token identifiers."""
        tokens = self.tokens(frame_index, min_confidence=min_confidence)
        return {
            "frame_index": int(frame_index),
            "feature_size": int(tokens[0].feature.size) if tokens else 0,
            "tokens": [
                {
                    "key": token.key,
                    "position": token.position.tolist(),
                    "feature": token.feature.tolist(),
                    "confidence": float(token.confidence),
                    "last_seen": int(token.last_seen),
                }
                for token in tokens
            ],
        }

    def save(self, path: str | Path, frame_index: int, *, min_confidence: float = 0.0) -> None:
        """Write a token snapshot as portable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.export(frame_index, min_confidence=min_confidence), indent=2) + "\n",
            encoding="utf-8",
        )
