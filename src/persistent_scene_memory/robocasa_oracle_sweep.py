"""Sweep oracle pick-place rollouts over seeds and keep only native successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np

from .robocasa_rollout import record_episode

_PHASE_ORDER = (
    "approach_object",
    "descend_to_object",
    "close_gripper",
    "lift_object",
    "approach_cabinet",
    "insert_object",
    "open_gripper",
    "retreat",
    "done",
)


def summarize_manifest(manifest: dict) -> dict:
    """Success frame, base travel, and furthest oracle phase for one rollout."""
    frames = manifest["frames"]
    success = [frame["frame_index"] for frame in frames if frame.get("success")]
    sources = [frame.get("action_source") or {} for frame in frames]
    bases = [np.asarray(s["base_world"][:2]) for s in sources if "base_world" in s]
    base_travel = max((float(np.linalg.norm(b - bases[0])) for b in bases), default=0.0)
    reached = [
        _PHASE_ORDER.index(s["phase"]) for s in sources if s.get("phase") in _PHASE_ORDER
    ]
    return {
        "frames": len(frames),
        "success": bool(success),
        "success_frame": success[0] if success else None,
        "base_travel": base_travel,
        "furthest_phase": _PHASE_ORDER[max(reached)] if reached else None,
        "task": manifest["task"],
    }


def resolve_object(spec: str) -> str:
    """Map ``category:Instance`` to RoboCasa's exact registered model.xml path."""
    if ":" not in spec:
        return spec
    from robocasa.models.objects.kitchen_object_utils import OBJ_CATEGORIES

    category, instance = spec.split(":", 1)
    paths = OBJ_CATEGORIES[category]["lightwheel"].mjcf_paths
    matches = [path for path in paths if Path(path).parent.name == instance]
    if len(matches) != 1:
        raise ValueError(f"{spec!r} matches {len(matches)} registered instances")
    return matches[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--object", required=True, help="object group name, or category:Instance for one instance"
    )
    parser.add_argument("--layout-style", help="fixed kitchen as LAYOUT,STYLE")
    parser.add_argument("--seeds", default="0-9", help="inclusive range A-B")
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--keep-failures", action="store_true")
    args = parser.parse_args(argv)

    import gymnasium as gym
    import robocasa  # noqa: F401

    from .robocasa_actions import OraclePickPlaceCounterToCabinet

    first, last = (int(value) for value in args.seeds.split("-"))
    environment_kwargs = {
        "split": "pretrain",
        "obj_registries": ("lightwheel",),
        "obj_groups": resolve_object(args.object),
        "camera_depths": True,
        "camera_widths": args.camera_size,
        "camera_heights": args.camera_size,
    }
    if args.layout_style:
        layout, style = (int(value) for value in args.layout_style.split(","))
        # RoboCasa's named splits override explicit scene ids, so drop the split here.
        environment_kwargs["split"] = None
        environment_kwargs["layout_and_style_ids"] = [[layout, style]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "sweep_summary.json"
    results = []
    env = gym.make("robocasa/PickPlaceCounterToCabinet", **environment_kwargs)
    try:
        for seed in range(first, last + 1):
            random.seed(seed)
            np.random.seed(seed)
            episode_dir = args.output_dir / f"seed{seed:03d}" / "episode"
            started = time.perf_counter()
            try:
                manifest = record_episode(
                    env,
                    episode_dir,
                    steps=args.steps,
                    seed=seed,
                    action_fn=OraclePickPlaceCounterToCabinet(),
                    stop_on_success=True,
                )
                result = {"seed": seed, **summarize_manifest(manifest)}
            except Exception as error:  # keep sweeping past single-scene failures
                result = {"seed": seed, "success": False, "error": repr(error)}
            result["elapsed_seconds"] = time.perf_counter() - started
            if not result["success"] and not args.keep_failures:
                shutil.rmtree(episode_dir.parent, ignore_errors=True)
            results.append(result)
            print(json.dumps(result), flush=True)
            summary_path.write_text(
                json.dumps({"config": vars(args) | {"output_dir": str(args.output_dir)},
                            "results": results}, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    finally:
        env.close()
    successes = sum(result["success"] for result in results)
    print(f"SUCCESS {successes}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
