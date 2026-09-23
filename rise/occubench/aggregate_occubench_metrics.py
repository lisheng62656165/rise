#!/usr/bin/env python3
"""Aggregate fixed OccuBench score files by the published ten categories.

Only rows with verification_valid=true and boolean is_correct are counted.
Invalid verifier/API responses remain pending and are never converted to
negative labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CATEGORY_ORDER = ["Agri", "Biz", "Comm", "Edu", "Hlth", "Ind", "Pub", "Sci", "Tech", "Trans"]
CATEGORY_MAP = {
    "Agriculture & Environment": "Agri",
    "Business & Enterprise": "Biz",
    "Commerce & Consumer": "Comm",
    "Education & Culture": "Edu",
    "Healthcare & Life Sciences": "Hlth",
    "Industrial & Engineering": "Ind",
    "Public Service & Governance": "Pub",
    "Science & Research": "Sci",
    "Technology & IT": "Tech",
    "Transportation & Logistics": "Trans",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tasks_by_category(data_root: Path) -> dict[int, str]:
    scenarios = {
        row["task_scenario_name"]: CATEGORY_MAP.get(row.get("category"), row.get("category", "Unknown"))
        for row in read_jsonl(data_root / "task_scenario_all_pool.jsonl")
    }
    tasks = {}
    for row in read_jsonl(data_root / "eval_benchmark_solvable.jsonl"):
        category = scenarios.get(row["task_scenario_name"], row.get("category", "Unknown"))
        tasks[int(row["task_id"])] = category
    return tasks


def valid_labels(path: Path, marker: str) -> dict[int, bool]:
    labels: dict[int, bool] = {}
    for row in read_jsonl(path):
        if row.get("method") != marker:
            continue
        if row.get("verification_valid") is not True or type(row.get("is_correct")) is not bool:
            continue
        labels[int(row["task_id"])] = bool(row["is_correct"])
    return labels


def summarize(labels: dict[int, bool], categories: dict[int, str]) -> dict:
    by_category = {}
    for category in CATEGORY_ORDER:
        ids = [tid for tid, name in categories.items() if name == category and tid in labels]
        passed = sum(labels[tid] for tid in ids)
        by_category[category] = {
            "passed": passed, "valid": len(ids),
            "pass_at_1": passed / len(ids) if ids else None,
        }
    return {
        "passed": sum(labels.values()),
        "valid": len(labels),
        "pass_at_1": sum(labels.values()) / len(labels) if labels else None,
        "by_category": by_category,
    }


def fmt(item: dict) -> str:
    if not item["valid"]:
        return "N/A"
    return f'{item["passed"]}/{item["valid"]} ({100 * item["pass_at_1"]:.2f}%)'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--scores-dir", type=Path, default=Path("results/occubench_382"))
    parser.add_argument("--out", type=Path, default=Path("results/occubench_382/category_metrics.json"))
    args = parser.parse_args()
    categories = tasks_by_category(args.data_root)
    score_rows = read_jsonl(args.scores_dir / "scores.jsonl")
    vanilla_rows = read_jsonl(args.scores_dir / "vanilla_scores.jsonl")
    methods = {
        "StateBench-EDS-ECA Original FINAL": {
            int(row["task_id"]): bool(row["is_correct"]) for row in score_rows
            if row.get("method") == "EDS-ECA" and row.get("verification_valid") is True
            and type(row.get("is_correct")) is bool
        },
        "OAgents Best-of-4": {
            int(row["task_id"]): bool(row["is_correct"]) for row in score_rows
            if row.get("method") in {"OAgents", "OAgents-Best-of-4"}
            and row.get("verification_valid") is True and type(row.get("is_correct")) is bool
        },
        "Vanilla-A": {
            int(row["task_id"]): bool(row["is_correct"]) for row in vanilla_rows
            if row.get("method") in {"Vanilla-A", "Vanilla"} and row.get("verification_valid") is True
            and type(row.get("is_correct")) is bool
        },
    }
    report = {name: summarize(labels, categories) for name, labels in methods.items()}
    report["metadata"] = {
        "task_count": len(categories), "category_order": CATEGORY_ORDER,
        "invalid_labels_excluded": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["| Method | Avg | " + " | ".join(CATEGORY_ORDER) + " |",
          "|---|" + "---:|" * (len(CATEGORY_ORDER) + 1)]
    for name in methods:
        item = report[name]
        cells = [f'{item["passed"]}/{item["valid"]} ({100 * item["pass_at_1"]:.2f}%)'
                 if item["valid"] else "N/A"]
        cells += [fmt(item["by_category"][category]) for category in CATEGORY_ORDER]
        md.append("| " + name + " | " + " | ".join(cells) + " |")
    md.append("")
    md.append("Invalid verifier rows are excluded; the released Vanilla-A snapshot has complete labels for all 382 tasks.")
    (args.out.with_suffix(".md")).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
