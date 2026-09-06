from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """World-to-camera rigid transform."""

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64).reshape(3, 1)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.eye(3), np.zeros((3, 1)))

    @property
    def camera_center(self) -> np.ndarray:
        return (-self.rotation.T @ self.translation).ravel()

    def matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation.ravel()
        return transform


@dataclass(frozen=True)
class Track:
    observations: dict[int, np.ndarray]
