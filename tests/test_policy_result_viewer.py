from persistent_scene_memory.policy_result_viewer import build_dashboard


def test_dashboard_embeds_current_policy_series():
    metrics = {"mae": 1.0, "rmse": 2.0, "normalized_rmse": 3.0}
    base = {
        "aggregate": {"smolvla": metrics},
        "per_episode": {"0": {"smolvla": metrics}},
    }
    adapted = {
        "episodes": [0],
        "aggregate": {
            "smolvla": metrics,
            "residual_blend": metrics,
            "hold_state_baseline": metrics,
        },
        "per_episode": {
            "0": {
                "smolvla": metrics,
                "residual_blend": metrics,
                "hold_state_baseline": metrics,
            }
        },
    }

    html = build_dashboard(base, adapted, {"promotion_passed": False})

    assert "SmolVLA policy gate" in html
    assert "Residual policy" in html
    assert "Episode '+k" in html
