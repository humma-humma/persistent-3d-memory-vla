import numpy as np
import pytest

from persistent_scene_memory.robocasa_skill_head import (
    classification_metrics,
    monotonic_decode,
)


def test_monotonic_decode_holds_or_advances_one_stage():
    logits = np.full((4, 6), -5.0, dtype=np.float32)
    logits[0, 4] = 10
    logits[1, 1] = 10
    logits[2, 0] = 10
    logits[3, 2] = 10
    assert monotonic_decode(logits).tolist() == [0, 1, 1, 2]


def test_classification_metrics_reports_accuracy_and_rejects_shape():
    result = classification_metrics(np.array([0, 1, 1]), np.array([0, 1, 2]))
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="matching"):
        classification_metrics(np.array([0]), np.array([0, 1]))
