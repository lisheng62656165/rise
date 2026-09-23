import json
import sys

import collect_results


def test_collect_results_excludes_invalid_oagents_verifier_rows(tmp_path, monkeypatch):
    paired = tmp_path / "paired"
    oagents = tmp_path / "oagents"
    output = tmp_path / "normalized"
    paired.mkdir()
    (oagents / "1").mkdir(parents=True)
    (oagents / "2").mkdir(parents=True)
    (paired / "scores.jsonl").write_text(
        json.dumps({"task_id": 1, "stage": "FINAL", "is_correct": True,
                    "verification_valid": True}) + "\n" +
        json.dumps({"task_id": 1, "stage": "A", "is_correct": False,
                    "verification_valid": True}) + "\n",
        encoding="utf-8",
    )
    (oagents / "1" / "result.json").write_text(json.dumps({
        "task_id": 1,
        "bestof4": {"is_correct": True, "verification_valid": True},
    }), encoding="utf-8")
    (oagents / "2" / "result.json").write_text(json.dumps({
        "task_id": 2,
        "bestof4": {"is_correct": False, "verification_valid": False},
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "collect_results.py", "--paired-dir", str(paired),
        "--oagents-dir", str(oagents), "--out", str(output),
    ])

    collect_results.main()

    rows = [json.loads(line) for line in (output / "scores.jsonl").read_text(
        encoding="utf-8").splitlines()]
    oagents_rows = [row for row in rows if row["method"] == "OAgents-Best-of-4"]
    assert [row["task_id"] for row in oagents_rows] == [1]
    assert oagents_rows[0]["is_correct"] is True
