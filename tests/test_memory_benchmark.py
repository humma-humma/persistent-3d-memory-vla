import json

from persistent_scene_memory.benchmark import BASELINES, run_benchmark, write_results
from persistent_scene_memory.vla_smoke import run_mock


def test_benchmark_is_reproducible_and_covers_required_baselines():
    _, first, _ = run_benchmark(seed=3, trials=2)
    _, second, _ = run_benchmark(seed=3, trials=2)

    assert tuple(first) == BASELINES
    assert first == second
    assert first["confidence_aware"]["action_error"] < first["current_rgb"]["action_error"]


def test_ambiguity_variant_adds_multi_view_consistency_baseline():
    _, summary, _ = run_benchmark(
        seed=3, trials=2, include_consistency=True, single_view_outliers=True
    )

    assert tuple(summary) == BASELINES + ("multi_view_consistency",)


def test_benchmark_writes_results_and_tokens(tmp_path):
    write_results(tmp_path, seed=3, trials=2)

    assert (tmp_path / "baseline_results.csv").is_file()
    assert set(json.loads((tmp_path / "baseline_summary.json").read_text())) == set(BASELINES)
    assert json.loads((tmp_path / "tokens.json").read_text())["tokens"][0]["key"] == "target"


def test_offline_vla_smoke_contract():
    result = run_mock()

    assert result["status"] == "ok"
    assert len(result["action"]) == 7
    assert "World-aligned scene memory" in result["prompt"]
