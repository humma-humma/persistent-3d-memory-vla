from persistent_scene_memory.promotion import compare_evaluations


def evaluation(model_rmse, model_nrmse, hold_rmse=2.0, hold_nrmse=2.0):
    model = {"mae": model_rmse, "rmse": model_rmse, "normalized_rmse": model_nrmse}
    hold = {"mae": hold_rmse, "rmse": hold_rmse, "normalized_rmse": hold_nrmse}
    return {
        "episodes": [0],
        "anchors": [0],
        "horizon": 50,
        "aggregate": {"smolvla": model, "hold_state_baseline": hold},
        "per_episode": {"0": {"smolvla": model, "hold_state_baseline": hold}},
    }


def test_promotion_requires_raw_and_normalized_wins():
    result = compare_evaluations(evaluation(3.0, 3.0), evaluation(1.0, 1.0))

    assert result["promotion_passed"]
    assert result["target_actions"] == 50
    assert result["episode_wins"]["rmse"]["adapted_better_than_hold_state"] == 1


def test_promotion_fails_when_raw_rmse_loses_to_hold_state():
    result = compare_evaluations(
        evaluation(3.0, 3.0, hold_rmse=0.5),
        evaluation(1.0, 1.0, hold_rmse=0.5),
    )

    assert not result["promotion_passed"]


def test_promotion_can_score_named_residual_policy():
    base = evaluation(3.0, 3.0)
    adapted = evaluation(3.0, 3.0)
    adapted["aggregate"]["residual_blend"] = {
        "mae": 1.0,
        "rmse": 1.0,
        "normalized_rmse": 1.0,
    }
    adapted["per_episode"]["0"]["residual_blend"] = adapted["aggregate"][
        "residual_blend"
    ]

    result = compare_evaluations(base, adapted, "residual_blend")

    assert result["promotion_passed"]
    assert result["adapted_model_key"] == "residual_blend"
