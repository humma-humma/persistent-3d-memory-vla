from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import Stage1Dataset
from .models import Track


ObservationKey = tuple[int, int, int]


def observation_key(
    image_id: int, point: np.ndarray, tolerance: float
) -> ObservationKey:
    point = np.asarray(point, dtype=np.float64).ravel()
    return (
        image_id,
        int(np.rint(point[0] / tolerance)),
        int(np.rint(point[1] / tolerance)),
    )


@dataclass(frozen=True)
class TrackBuildResult:
    tracks: list[Track]
    observation_to_track: dict[ObservationKey, int]
    skipped_conflicts: int
    skipped_conflicts_by_pair: dict[tuple[int, int], int]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []
        self.images: list[set[int]] = []

    def add(self, image_id: int) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        self.images.append({image_id})
        return index

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return True
        if self.images[first_root] & self.images[second_root]:
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        self.images[first_root].update(self.images[second_root])
        return True


def build_tracks(
    dataset: Stage1Dataset,
    tolerance: float = 1e-3,
    min_observations: int = 2,
) -> TrackBuildResult:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if min_observations < 2:
        raise ValueError("min_observations must be at least 2")

    union_find = _UnionFind()
    nodes: dict[ObservationKey, int] = {}
    coordinates: dict[ObservationKey, np.ndarray] = {}
    skipped_conflicts = 0
    skipped_conflicts_by_pair: dict[tuple[int, int], int] = {}

    def get_node(image_id: int, point: np.ndarray) -> int:
        key = observation_key(image_id, point, tolerance)
        if key not in nodes:
            nodes[key] = union_find.add(image_id)
            coordinates[key] = np.asarray(point, dtype=np.float64).copy()
        return nodes[key]

    for (first_id, second_id), matches in dataset.iter_correspondences():
        for match in matches:
            first_node = get_node(first_id, match[:2])
            second_node = get_node(second_id, match[2:])
            if not union_find.union(first_node, second_node):
                skipped_conflicts += 1
                pair = (first_id, second_id)
                skipped_conflicts_by_pair[pair] = (
                    skipped_conflicts_by_pair.get(pair, 0) + 1
                )

    members: dict[int, list[ObservationKey]] = {}
    for key, node in nodes.items():
        root = union_find.find(node)
        members.setdefault(root, []).append(key)

    tracks: list[Track] = []
    observation_to_track: dict[ObservationKey, int] = {}
    for component in members.values():
        observations = {
            key[0]: coordinates[key]
            for key in sorted(component)
        }
        if len(observations) < min_observations:
            continue
        track_id = len(tracks)
        tracks.append(Track(observations=observations))
        for key in component:
            observation_to_track[key] = track_id

    return TrackBuildResult(
        tracks=tracks,
        observation_to_track=observation_to_track,
        skipped_conflicts=skipped_conflicts,
        skipped_conflicts_by_pair=skipped_conflicts_by_pair,
    )
