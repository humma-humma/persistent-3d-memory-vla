from persistent_scene_memory.robocasa_oracle_sweep import summarize_manifest


def _frame(index, phase, base, success=False):
    return {
        "frame_index": index,
        "success": success,
        "action_source": {"phase": phase, "base_world": [base, 0.0, 0.0]},
    }


def test_summarize_manifest_reports_success_travel_and_phase():
    manifest = {
        "task": "pick",
        "frames": [
            _frame(0, "approach_object", 1.0),
            _frame(1, "lift_object", 1.3),
            _frame(2, "descend_to_object", 1.1, success=True),
        ],
    }
    summary = summarize_manifest(manifest)
    assert summary["success"] and summary["success_frame"] == 2
    assert abs(summary["base_travel"] - 0.3) < 1e-9
    assert summary["furthest_phase"] == "lift_object"


def test_summarize_manifest_without_oracle_metadata():
    summary = summarize_manifest({"task": "pick", "frames": [{"frame_index": 0}]})
    assert not summary["success"] and summary["furthest_phase"] is None
    assert summary["base_travel"] == 0.0
