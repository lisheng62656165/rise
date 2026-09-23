import json
import sys

import pytest
import aggregate_occubench_metrics as aggregate
import collect_results
from run_full_comparison import parse_args, validate_oagents_manifest, write_overall_results


def test_overall_report_keeps_incomplete_methods_pending(tmp_path):
    report = {
        "StateBench-EDS-ECA Original FINAL": {"passed": 2, "valid": 3, "pass_at_1": 2 / 3,
                                                  "by_category": {}},
        "OAgents Best-of-4": {"passed": 3, "valid": 3, "pass_at_1": 1.0,
                              "by_category": {}},
        "Vanilla-A": {"passed": 1, "valid": 3, "pass_at_1": 1 / 3,
                     "by_category": {}},
    }
    (tmp_path / "category_metrics.json").write_text(json.dumps(report), encoding="utf-8")

    write_overall_results(tmp_path, 3, "test-model")

    actual = json.loads((tmp_path / "overall_results.json").read_text(encoding="utf-8"))
    assert actual["status"] == "complete"
    assert actual["methods"]["OAgents Best-of-4"]["valid"] == 3
    assert "| OAgents Best-of-4 | 3/3 | 3 | 100.00% | complete |" in (
        tmp_path / "overall_results.md").read_text(encoding="utf-8")
    assert "|---|---:|---:|---:|---|" in (tmp_path / "overall_results.md").read_text(encoding="utf-8")

    report["Vanilla-A"]["valid"] = 2
    report["Vanilla-A"]["pass_at_1"] = 0.5
    (tmp_path / "category_metrics.json").write_text(json.dumps(report), encoding="utf-8")
    write_overall_results(tmp_path, 3, "test-model")
    actual = json.loads((tmp_path / "overall_results.json").read_text(encoding="utf-8"))
    assert actual["status"] == "pending_valid_labels"
    assert actual["methods"]["Vanilla"]["status"] == "pending_valid_labels"


def test_oagents_manifest_must_match_cohort_model_seed_and_selector(tmp_path):
    validate_oagents_manifest(tmp_path, [1, 2], "model", 3)
    manifest = {"task_ids": [1, 2], "model": "model", "seed": 3,
                "selector_policy": "oagents", "selector_seed": 53403}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validate_oagents_manifest(tmp_path, [1, 2], "model", 3)
    with pytest.raises(ValueError, match="selector seed"):
        validate_oagents_manifest(tmp_path, [1, 2], "model", 3, 42)

    manifest["selector_policy"] = "legacy"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        validate_oagents_manifest(tmp_path, [1, 2], "model", 3)
    except ValueError as exc:
        assert "different selector" in str(exc)
    else:
        raise AssertionError("selector mismatch was not rejected")


def test_nonempty_unmanifested_oagents_directory_requires_new_output(tmp_path):
    (tmp_path / "orphan.json").write_text("{}", encoding="utf-8")
    try:
        validate_oagents_manifest(tmp_path, [1], "model", 3)
    except ValueError as exc:
        assert "no manifest" in str(exc)
    else:
        raise AssertionError("unmanifested output was accepted")


def test_fresh_output_and_config_overrides(tmp_path):
    validate_oagents_manifest(tmp_path / "not_created", [1], "model", 3)
    args = parse_args(["--base-url", "https://example.test/v1"])
    assert (args.model, args.seed, args.selector_seed, args.oagents_selector_seed) == (
        "deepseek-v4.1-flash", 53403, 77113, 53403)
    args = parse_args(["--base-url", "https://example.test/v1", "--seed", "42",
                       "--selector-seed", "43", "--oagents-selector-seed", "44", "--model", "other"])
    assert (args.model, args.seed, args.selector_seed, args.oagents_selector_seed) == ("other", 42, 43, 44)


def test_collect_aggregate_and_overall_use_real_method_names(tmp_path, monkeypatch):
    paired, bon, scores, data = [tmp_path / name for name in ("paired", "bon", "scores", "data")]
    paired.mkdir()
    (bon / "1").mkdir(parents=True)
    data.mkdir()
    rows = [{"task_id": 1, "stage": stage, "is_correct": outcome, "verification_valid": True}
            for stage, outcome in [("A", False), ("FINAL", True)]]
    (paired / "scores.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    (bon / "1" / "result.json").write_text(json.dumps({"task_id": 1,
        "bestof4": {"is_correct": True, "verification_valid": True}}), encoding="utf-8")
    (data / "eval_benchmark_solvable.jsonl").write_text(json.dumps(
        {"task_id": 1, "task_scenario_name": "test"}), encoding="utf-8")
    (data / "task_scenario_all_pool.jsonl").write_text(json.dumps(
        {"task_scenario_name": "test", "category": "Science & Research"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["collect", "--paired-dir", str(paired), "--oagents-dir", str(bon), "--out", str(scores)])
    collect_results.main()
    monkeypatch.setattr(sys, "argv", ["aggregate", "--data-root", str(data), "--scores-dir", str(scores), "--out", str(scores / "category_metrics.json")])
    aggregate.main()
    write_overall_results(scores, 1, "test-model")
    result = json.loads((scores / "overall_results.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["methods"]["Vanilla"]["passed"] == 0
    assert result["methods"]["OAgents Best-of-4"]["by_category"]["Sci"]["passed"] == 1
    markdown = (scores / "category_metrics.md").read_text(encoding="utf-8")
    assert "| Method | Avg | Agri | Biz | Comm | Edu | Hlth | Ind | Pub | Sci | Tech | Trans |" in markdown
    assert "| Vanilla | 0/1 (0.00%)" in markdown
