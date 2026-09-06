"""Priority 5: wrapper around the assignment's official ATE evaluator.

The provided ``eval.py`` computes a per-frame scale factor and then shells out
to ``python3 eval_ate.py``.  On Windows ``python3`` frequently does not exist,
and the shipped ``eval_ate.py`` imports an ``associate`` module that is not
bundled with it.  This wrapper:

* computes the same scale factor as ``eval.py`` (mean of per-pose translation
  norm ratios),
* invokes the provided ``eval_ate.py`` with the current interpreter
  (:data:`sys.executable`) instead of ``python3``,
* injects the vendored :mod:`_tum_eval.associate` onto ``PYTHONPATH`` so the
  provided script resolves its import without editing the assignment files,
* captures the verbose output and writes ``official_eval_<label>_stdout.txt``
  and a combined ``official_eval_summary.json``.

The assignment files in ``Experiments/Stage_3_Data/stage3`` are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


_VENDOR_DIR = Path(__file__).resolve().parent / "_tum_eval"
_METRIC_PATTERN = re.compile(
    r"absolute_translational_error\.(\w+)\s+([-\d.eE+]+)\s*m"
)
_PAIRS_PATTERN = re.compile(r"compared_pose_pairs\s+(\d+)\s+pairs")


@dataclass(frozen=True)
class OfficialEvalResult:
    label: str
    estimated_path: Path
    scale: float
    returncode: int
    rmse: float | None
    metrics: dict[str, float]
    compared_pose_pairs: int | None
    stdout_path: Path


def find_evaluator_dir(start: str | Path | None = None) -> Path | None:
    """Search upward/around for a directory containing ``eval_ate.py``."""
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "Experiments" / "Stage_3_Data" / "stage3")
    for candidate in candidates:
        if (candidate / "eval_ate.py").is_file():
            return candidate.resolve()
    return None


def compute_scale(
    ground_truth_path: str | Path, estimated_path: str | Path
) -> float:
    """Replicate the scale factor from the provided ``eval.py``.

    ``eval.py`` reads both files line-by-line (skipping the first row) and
    averages ``norm(gt_translation) / norm(est_translation)``.  Rows where the
    estimated translation norm is zero are skipped to avoid division by zero
    (the first pose is at the origin); the remaining rows match ``eval.py``.
    """
    gt_lines = _data_rows(Path(ground_truth_path))
    est_lines = _data_rows(Path(estimated_path))
    count = min(len(gt_lines), len(est_lines))
    total = 0.0
    used = 0
    for index in range(1, count):
        gt = np.asarray(gt_lines[index][1:4], dtype=np.float64)
        est = np.asarray(est_lines[index][1:4], dtype=np.float64)
        est_norm = float(np.linalg.norm(est))
        if est_norm == 0.0:
            continue
        total += float(np.linalg.norm(gt)) / est_norm
        used += 1
    if used == 0:
        return 1.0
    return total / used


def run_official_eval(
    ground_truth_path: str | Path,
    estimated_path: str | Path,
    output_dir: str | Path,
    *,
    label: str,
    evaluator_dir: str | Path | None = None,
    max_difference: float = 0.02,
) -> OfficialEvalResult:
    """Run the official ATE evaluator for one estimated trajectory."""
    ground_truth_path = Path(ground_truth_path)
    estimated_path = Path(estimated_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_dir = (
        Path(evaluator_dir).resolve()
        if evaluator_dir is not None
        else find_evaluator_dir()
    )
    if resolved_dir is None or not (resolved_dir / "eval_ate.py").is_file():
        raise FileNotFoundError(
            "Could not locate eval_ate.py; pass --evaluator-dir explicitly"
        )

    scale = compute_scale(ground_truth_path, estimated_path)

    env = dict(os.environ)
    extra_paths = [str(_VENDOR_DIR), str(resolved_dir)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [path for path in extra_paths if path] + ([existing] if existing else [])
    )

    command = [
        sys.executable,
        str(resolved_dir / "eval_ate.py"),
        "--scale",
        repr(float(scale)),
        "--max_difference",
        repr(float(max_difference)),
        "--verbose",
        str(ground_truth_path.resolve()),
        str(estimated_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=str(resolved_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    stdout_path = output_dir / f"official_eval_{label}_stdout.txt"
    stdout_path.write_text(
        f"$ {' '.join(command)}\n\n"
        f"[returncode] {completed.returncode}\n\n"
        f"[stdout]\n{completed.stdout}\n"
        f"[stderr]\n{completed.stderr}\n",
        encoding="utf-8",
    )

    metrics = {
        key: float(value)
        for key, value in _METRIC_PATTERN.findall(completed.stdout)
    }
    pairs_match = _PAIRS_PATTERN.search(completed.stdout)
    compared = int(pairs_match.group(1)) if pairs_match else None
    rmse = metrics.get("rmse")

    return OfficialEvalResult(
        label=label,
        estimated_path=estimated_path,
        scale=float(scale),
        returncode=completed.returncode,
        rmse=rmse,
        metrics=metrics,
        compared_pose_pairs=compared,
        stdout_path=stdout_path,
    )


def write_official_eval_summary(
    path: str | Path, results: list[OfficialEvalResult]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluator": "TUM eval_ate.py (absolute trajectory error, Horn aligned)",
        "interpreter": sys.executable,
        "trajectories": {
            result.label: {
                "estimated_path": str(result.estimated_path),
                "scale": result.scale,
                "returncode": result.returncode,
                "rmse": result.rmse,
                "compared_pose_pairs": result.compared_pose_pairs,
                "metrics": result.metrics,
                "stdout": str(result.stdout_path),
            }
            for result in results
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _data_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(token) for token in line.replace(",", " ").split()]
        if len(values) >= 4:
            rows.append(values)
    return rows


def add_official_eval_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("official evaluator")
    group.add_argument(
        "--run-official-eval",
        action="store_true",
        help="Run the assignment's eval_ate.py on produced trajectories.",
    )
    group.add_argument(
        "--evaluator-dir",
        type=Path,
        help="Directory containing eval_ate.py (auto-detected if omitted).",
    )
    group.add_argument("--official-eval-max-difference", type=float, default=0.02)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the official Stage 3 ATE evaluator on a trajectory."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--estimated", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="estimated")
    parser.add_argument("--evaluator-dir", type=Path)
    parser.add_argument("--max-difference", type=float, default=0.02)
    args = parser.parse_args(argv)

    result = run_official_eval(
        args.ground_truth,
        args.estimated,
        args.output_dir,
        label=args.label,
        evaluator_dir=args.evaluator_dir,
        max_difference=args.max_difference,
    )
    write_official_eval_summary(
        args.output_dir / "official_eval_summary.json", [result]
    )
    print(
        f"Official ATE RMSE ({result.label}): "
        f"{result.rmse if result.rmse is not None else 'n/a'} "
        f"(scale {result.scale:.6f}, returncode {result.returncode})."
    )
    print(f"Official eval stdout: {result.stdout_path.resolve()}")


if __name__ == "__main__":
    main()
