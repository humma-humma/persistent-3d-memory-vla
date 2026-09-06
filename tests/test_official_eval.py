import json

import pytest

from sfm_reconstruction.official_eval import (
    OfficialEvalResult,
    compute_scale,
    find_evaluator_dir,
    run_official_eval,
    write_official_eval_summary,
)


def _write_trajectory(path, rows):
    path.write_text(
        "\n".join(
            " ".join(str(value) for value in row) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_compute_scale_matches_eval_py_ratio(tmp_path):
    gt = tmp_path / "gt.txt"
    est = tmp_path / "est.txt"
    # Row 0 is skipped (matches eval.py). Rows 1-2: gt translation is 2x est.
    _write_trajectory(
        gt,
        [
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.1, 2.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.2, 0.0, 4.0, 0.0, 0, 0, 0, 1],
        ],
    )
    _write_trajectory(
        est,
        [
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.1, 1.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.2, 0.0, 2.0, 0.0, 0, 0, 0, 1],
        ],
    )

    assert compute_scale(gt, est) == pytest.approx(2.0)


def test_compute_scale_skips_zero_norm_rows(tmp_path):
    gt = tmp_path / "gt.txt"
    est = tmp_path / "est.txt"
    _write_trajectory(
        gt,
        [
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.1, 0.0, 0.0, 0.0, 0, 0, 0, 1],  # est norm 0 -> skipped
            [0.2, 3.0, 0.0, 0.0, 0, 0, 0, 1],
        ],
    )
    _write_trajectory(
        est,
        [
            [0.0, 0.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.1, 0.0, 0.0, 0.0, 0, 0, 0, 1],
            [0.2, 1.0, 0.0, 0.0, 0, 0, 0, 1],
        ],
    )

    assert compute_scale(gt, est) == pytest.approx(3.0)


def test_write_official_eval_summary(tmp_path):
    result = OfficialEvalResult(
        label="pose_graph",
        estimated_path=tmp_path / "est.txt",
        scale=1.0,
        returncode=0,
        rmse=0.12,
        metrics={"rmse": 0.12, "mean": 0.1},
        compared_pose_pairs=100,
        stdout_path=tmp_path / "official_eval_pose_graph_stdout.txt",
    )
    summary_path = tmp_path / "official_eval_summary.json"

    write_official_eval_summary(summary_path, [result])

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["trajectories"]["pose_graph"]["rmse"] == 0.12
    assert data["trajectories"]["pose_graph"]["compared_pose_pairs"] == 100


def test_run_official_eval_against_provided_evaluator(tmp_path):
    evaluator_dir = find_evaluator_dir()
    if evaluator_dir is None:
        pytest.skip("provided eval_ate.py not found")

    gt = tmp_path / "gt.txt"
    est = tmp_path / "est.txt"
    rows = [
        [round(0.1 * i, 6), float(i), 0.0, 0.0, 0, 0, 0, 1] for i in range(6)
    ]
    _write_trajectory(gt, rows)
    _write_trajectory(est, rows)  # identical -> ATE RMSE ~ 0

    result = run_official_eval(
        gt, est, tmp_path / "out", label="identity", evaluator_dir=evaluator_dir
    )

    assert result.returncode == 0, result.stdout_path.read_text(encoding="utf-8")
    assert result.rmse is not None
    assert result.rmse < 1e-6
    assert result.compared_pose_pairs == 6
    assert (tmp_path / "out" / "official_eval_identity_stdout.txt").is_file()
