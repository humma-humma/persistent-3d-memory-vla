import numpy as np
import json

from persistent_scene_memory import (
    ConsistencyGatedMemory,
    GeometryObservation,
    PersistentSceneMemory,
)


def observation(key, position, confidence, frame, feature=(1.0, 0.0)):
    return GeometryObservation(
        key=key,
        position=np.asarray(position),
        feature=np.asarray(feature),
        confidence=confidence,
        frame_index=frame,
    )


def test_unseen_token_persists_with_decaying_confidence():
    memory = PersistentSceneMemory(confidence_decay=0.1)
    memory.update(observation("cup", [1, 2, 3], 0.8, 0))

    token = memory.tokens(2)[0]

    assert token.key == "cup"
    assert np.allclose(token.position, [1, 2, 3])
    assert np.isclose(token.confidence, 0.8 * 0.9**2)


def test_consistent_observations_are_confidence_weighted():
    memory = PersistentSceneMemory(confidence_decay=0.0, change_threshold=1.0)
    memory.update(observation("cup", [0, 0, 0], 0.25, 0, feature=[0, 1]))

    token = memory.update(observation("cup", [0.2, 0, 0], 0.75, 1, feature=[1, 0]))

    assert np.allclose(token.position, [0.15, 0, 0])
    assert np.allclose(token.feature, [0.75, 0.25])
    assert np.isclose(token.confidence, 0.8125)


def test_confident_inconsistent_observation_replaces_stale_geometry():
    memory = PersistentSceneMemory(confidence_decay=0.5, change_threshold=0.25)
    memory.update(observation("cup", [0, 0, 0], 0.9, 0, feature=[1, 0]))

    token = memory.update(observation("cup", [1, 0, 0], 0.8, 2, feature=[0, 1]))

    assert np.allclose(token.position, [1, 0, 0])
    assert np.allclose(token.feature, [0, 1])
    assert token.confidence == 0.8


def test_token_matrix_contains_geometry_confidence_age_and_features():
    memory = PersistentSceneMemory(confidence_decay=0.0)
    memory.update(observation("cup", [1, 2, 3], 0.8, 4, feature=[0.1, 0.2]))

    matrix = memory.token_matrix(7)

    assert matrix.shape == (1, 7)
    assert np.allclose(matrix[0], [1, 2, 3, 0.8, 3, 0.1, 0.2])


def test_json_export_preserves_stable_identifier(tmp_path):
    memory = PersistentSceneMemory(confidence_decay=0.0)
    memory.update(observation("cup", [1, 2, 3], 0.8, 4))

    destination = tmp_path / "tokens.json"
    memory.save(destination, 5)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["frame_index"] == 5
    assert payload["tokens"][0]["key"] == "cup"
    assert payload["tokens"][0]["last_seen"] == 4


def test_consistency_gate_marks_one_disagreement_uncertain_then_confirms_update():
    memory = ConsistencyGatedMemory(confidence_decay=0.0, change_threshold=0.25)
    memory.update(observation("cup", [0, 0, 0], 0.9, 0))

    first = memory.update(observation("cup", [1.0, 0, 0], 0.8, 1))
    second = memory.update(observation("cup", [1.1, 0, 0], 0.8, 2))

    assert first.state == "uncertain"
    assert np.allclose(first.token.position, [0, 0, 0])
    assert second.state == "update"
    assert second.evidence_views == 2
    assert np.allclose(second.token.position, [1.05, 0, 0])


def test_consistency_gate_rejects_single_view_outlier():
    memory = ConsistencyGatedMemory(confidence_decay=0.0, change_threshold=0.25)
    memory.update(observation("cup", [0, 0, 0], 0.9, 0))

    uncertain = memory.update(observation("cup", [1.0, 0, 0], 0.8, 1))
    retained = memory.update(observation("cup", [0.05, 0, 0], 0.8, 2))

    assert uncertain.state == "uncertain"
    assert retained.state == "retain"
    assert memory.export(2)["uncertain"] == []
    assert memory.tokens(2)[0].position[0] < 0.05


def test_same_frame_cannot_confirm_a_change_twice():
    memory = ConsistencyGatedMemory(confidence_decay=0.0, change_threshold=0.25)
    memory.update(observation("cup", [0, 0, 0], 0.9, 0))

    memory.update(observation("cup", [1.0, 0, 0], 0.8, 1))
    decision = memory.update(observation("cup", [1.1, 0, 0], 0.8, 1))

    assert decision.state == "uncertain"
    assert decision.evidence_views == 1


def test_policy_tokens_expose_uncertain_candidate_separately():
    memory = ConsistencyGatedMemory(confidence_decay=0.0, change_threshold=0.25)
    memory.update(observation("cup", [0, 0, 0], 0.9, 0, feature=[1, 0]))
    memory.update(observation("cup", [1, 0, 0], 0.8, 1, feature=[0, 1]))

    tokens = memory.policy_tokens(1)

    assert [token.key for token in tokens] == ["cup", "cup?candidate"]
    assert tokens[0].feature.tolist() == [1, 0, 0]
    assert tokens[1].feature.tolist() == [0, 1, 1]
