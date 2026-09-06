"""Compare matched base and adapted SmolVLA real-data evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_evaluations(
    base: dict, adapted: dict, adapted_model_key: str = "smolvla"
) -> dict:
    if base["episodes"] != adapted["episodes"] or base["anchors"] != adapted["anchors"]:
        raise ValueError("base and adapted evaluations must use identical episodes and anchors")
    metrics = ("mae", "rmse", "normalized_rmse")
    comparison = {}
    for metric in metrics:
        base_value = base["aggregate"]["smolvla"][metric]
        adapted_value = adapted["aggregate"][adapted_model_key][metric]
        hold_value = adapted["aggregate"]["hold_state_baseline"][metric]
        comparison[metric] = {
            "base": base_value,
            "adapted": adapted_value,
            "hold_state": hold_value,
            "improvement_vs_base": (base_value - adapted_value) / base_value,
            "improvement_vs_hold_state": (hold_value - adapted_value) / hold_value,
        }
    episode_wins = {}
    for metric in metrics:
        episode_wins[metric] = {
            "adapted_better_than_base": sum(
                adapted["per_episode"][key][adapted_model_key][metric]
                < base["per_episode"][key]["smolvla"][metric]
                for key in base["per_episode"]
            ),
            "adapted_better_than_hold_state": sum(
                adapted["per_episode"][key][adapted_model_key][metric]
                < adapted["per_episode"][key]["hold_state_baseline"][metric]
                for key in base["per_episode"]
            ),
            "episode_count": len(base["per_episode"]),
        }
    passed = all(
        comparison[metric]["adapted"] < comparison[metric]["base"]
        and comparison[metric]["adapted"] < comparison[metric]["hold_state"]
        for metric in ("rmse", "normalized_rmse")
    )
    return {
        "episodes": base["episodes"],
        "anchors": base["anchors"],
        "target_actions": len(base["episodes"]) * len(base["anchors"]) * base["horizon"],
        "metrics": comparison,
        "episode_wins": episode_wins,
        "promotion_passed": passed,
        "adapted_model_key": adapted_model_key,
        "promotion_rule": "Adapted aggregate RMSE and normalized RMSE must beat both base and hold-state.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smolvla_promotion_gate.json"))
    parser.add_argument("--adapted-model-key", default="smolvla")
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    adapted = json.loads(args.adapted.read_text(encoding="utf-8"))
    result = compare_evaluations(base, adapted, args.adapted_model_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
