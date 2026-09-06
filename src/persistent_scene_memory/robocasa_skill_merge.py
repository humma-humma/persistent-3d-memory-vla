"""Merge hierarchical feature tables with deterministic per-class source caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .robocasa_hierarchical import SKILLS


def merge_skill_features(
    primary: str | Path,
    auxiliary: str | Path,
    output: str | Path,
    *,
    auxiliary_per_skill: int,
    seed: int = 17,
) -> dict:
    if auxiliary_per_skill < 1:
        raise ValueError("auxiliary_per_skill must be positive")
    tables = []
    for path in (primary, auxiliary):
        with np.load(path) as archive:
            table = {key: archive[key].copy() for key in archive.files}
        if tuple(table["skills"].tolist()) != SKILLS:
            raise ValueError("feature table does not use the hierarchical skill contract")
        tables.append(table)
    if tables[0]["features"].shape[1] != tables[1]["features"].shape[1]:
        raise ValueError("feature dimensions do not match")

    rng = np.random.default_rng(seed)
    auxiliary_indices = []
    for label in range(len(SKILLS)):
        candidates = np.flatnonzero(tables[1]["labels"] == label)
        rng.shuffle(candidates)
        auxiliary_indices.extend(candidates[:auxiliary_per_skill])
    auxiliary_indices = np.asarray(sorted(auxiliary_indices), dtype=np.int64)
    selected = [np.arange(len(tables[0]["labels"])), auxiliary_indices]
    merged = {}
    for key in ("features", "labels", "frame_indices", "episode_roots"):
        merged[key] = np.concatenate(
            [table[key][indices] for table, indices in zip(tables, selected)]
        )
    merged["skills"] = np.asarray(SKILLS)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return {
        "status": "skill_features_merged",
        "primary_samples": int(len(selected[0])),
        "auxiliary_samples": int(len(selected[1])),
        "samples": int(len(merged["labels"])),
        "auxiliary_per_skill": auxiliary_per_skill,
        "output": str(output.resolve()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--auxiliary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--auxiliary-per-skill", required=True, type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    print(json.dumps(merge_skill_features(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
