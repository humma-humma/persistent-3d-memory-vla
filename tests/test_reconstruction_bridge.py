import json

import numpy as np
import pytest

from persistent_scene_memory.reconstruction_bridge import (
    geometric_confidence,
    observations_from_rich_ply,
    process_reconstruction_sequence,
)


def write_rich_ply(path, rows):
    properties = [
        ("float", "x"), ("float", "y"), ("float", "z"),
        ("uchar", "red"), ("uchar", "green"), ("uchar", "blue"),
        ("int", "track_id"), ("int", "observations"),
        ("int", "registered_observations"),
        ("float", "mean_reprojection_error"),
        ("float", "max_reprojection_error"),
        ("float", "max_triangulation_angle"),
    ]
    lines = ["ply", "format ascii 1.0", f"element vertex {len(rows)}"]
    lines.extend(f"property {kind} {name}" for kind, name in properties)
    lines.append("end_header")
    lines.extend(" ".join(map(str, row)) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_geometric_confidence_combines_all_diagnostics():
    confidence = geometric_confidence(3, 2, 5)

    assert confidence == pytest.approx(np.exp(-1.0))
    assert geometric_confidence(3, 0, 0) == 0.0


def test_rich_ply_becomes_stable_world_frame_observations(tmp_path):
    source = tmp_path / "rich.ply"
    write_rich_ply(source, [[1, 2, 3, 255, 128, 0, 42, 4, 3, 0, 0.5, 5]])

    observations = observations_from_rich_ply(source, 7)

    assert len(observations) == 1
    assert observations[0].key == "track:42"
    assert np.allclose(observations[0].position, [1, 2, 3])
    assert np.allclose(observations[0].feature, [1, 128 / 255, 0])
    assert observations[0].confidence == 1.0
    assert observations[0].frame_index == 7


def test_duplicate_track_ids_are_rejected(tmp_path):
    source = tmp_path / "rich.ply"
    row = [1, 2, 3, 255, 128, 0, 42, 4, 3, 0, 0.5, 5]
    write_rich_ply(source, [row, row])

    with pytest.raises(ValueError, match="unique"):
        observations_from_rich_ply(source, 0)


def test_sequence_writes_memory_and_fixed_size_adapter_snapshots(tmp_path):
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"
    write_rich_ply(first, [[0, 0, 0, 255, 0, 0, 1, 3, 3, 0, 0, 5]])
    write_rich_ply(second, [[1, 0, 0, 0, 255, 0, 2, 3, 3, 0, 0, 5]])
    output = tmp_path / "memory"

    summary = process_reconstruction_sequence(
        [first, second], output, confidence_decay=0.1, max_tokens=3
    )

    assert [frame["memory_tokens"] for frame in summary["frames"]] == [1, 2]
    memory = json.loads((output / "memory_frame_000001.json").read_text())
    assert [token["key"] for token in memory["tokens"]] == ["track:2", "track:1"]
    assert memory["tokens"][1]["confidence"] == pytest.approx(0.9)
    adapter = json.loads((output / "adapter_frame_000001.json").read_text())
    assert adapter["mask"] == [True, True, False]
    assert adapter["keys"] == ["track:2", "track:1"]
    assert len(adapter["values"]) == 3


def test_sequence_can_export_uncertain_candidates_for_policy(tmp_path):
    paths = [tmp_path / f"frame_{index}.ply" for index in range(3)]
    positions = [0.0, 1.0, 1.1]
    for path, position in zip(paths, positions):
        write_rich_ply(
            path, [[position, 0, 0, 255, 0, 0, 1, 3, 3, 0, 0, 5]]
        )
    output = tmp_path / "gated"

    summary = process_reconstruction_sequence(
        paths, output, confirmation_views=2, change_threshold=0.25, max_tokens=3
    )

    assert summary["frames"][1]["update_decisions"]["uncertain"] == 1
    assert summary["frames"][2]["update_decisions"]["update"] == 1
    uncertain = json.loads((output / "memory_frame_000001.json").read_text())
    assert uncertain["uncertain"][0]["evidence_views"] == 1
    adapter = json.loads((output / "adapter_frame_000001.json").read_text())
    assert adapter["keys"][:2] == ["track:1", "track:1?candidate"]
    assert len(adapter["values"][0]) == 9
