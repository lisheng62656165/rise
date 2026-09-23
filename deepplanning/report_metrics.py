"""Aggregate DeepPlanning official summaries and optionally draw the paper table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHODS = ("vanilla", "oagents_best4", "statetrace_dsr")
DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "oagents_best4": "OAgents Best-of-4",
    "statetrace_dsr": "StateTrace-DSR",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value * 100 if value <= 1 else value


def pick(d: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in d:
            return d[name]
    return None


def travel_metrics(path: Path) -> dict[str, float | None]:
    d = load(path)
    m = d.get("metrics", d)
    return {
        "cs": pct(pick(m, "commonsense_score", "cs_score")),
        "ps": pct(pick(m, "personalized_score", "ps_score")),
        "comp": pct(pick(m, "composite_score", "comp_score")),
        "case": pct(pick(m, "case_acc", "case_accuracy")),
    }


def shopping_metrics(paths: list[Path]) -> dict[str, float | None]:
    rows = [load(path) for path in paths]
    stats = [d.get("overall_statistics", d) for d in rows]
    total = sum(float(pick(s, "total_cases", "total", "num_cases") or 0) for s in stats)
    matched = sum(float(pick(s, "total_matched_products", "matched_products") or 0) for s in stats)
    expected = sum(float(pick(s, "total_expected_products", "expected_products") or 0) for s in stats)
    successful = sum(float(pick(s, "successful_cases", "success_count") or 0) for s in stats)
    # Prefer official aggregate fields when available; otherwise aggregate counts.
    match_values = [pct(pick(s, "overall_match_rate", "match_score")) for s in stats]
    case_values = [pct(pick(s, "case_acc", "case_accuracy", "case_accuracy_rate")) for s in stats]
    match = (matched / expected * 100) if expected else (sum(v for v in match_values if v is not None) / len(match_values) if match_values else None)
    case = (successful / total * 100) if total else (sum(v for v in case_values if v is not None) / len(case_values) if case_values else None)
    return {"match": match, "case": case}


def resolve_summary(root: Path, method: str, domain: str, language: str | None = None) -> Path:
    path = root / method / domain
    if language:
        path = path / language
    candidates = [path / "evaluation_summary.json", path / "summary_report.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No official summary under {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="results/<method>/{shopping,travel_zh,travel_en}/...")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--csv", type=Path, help="Optional paper-table CSV with the six displayed metrics.")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    table: dict[str, Any] = {}
    for method in METHODS:
        shopping_dir = args.results / method / "shopping"
        shopping_paths = sorted(shopping_dir.glob("**/summary_report.json"))
        if not shopping_paths:
            raise FileNotFoundError(f"No Shopping summaries under {shopping_dir}")
        row: dict[str, Any] = {"shopping": shopping_metrics(shopping_paths)}
        zh = travel_metrics(resolve_summary(args.results, method, "travel_zh"))
        en = travel_metrics(resolve_summary(args.results, method, "travel_en"))
        row["travel_zh"] = zh
        row["travel_en"] = en
        row["travel_avg"] = {key: round((zh[key] + en[key]) / 2, 4) if zh[key] is not None and en[key] is not None else None for key in zh}
        table[method] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for method in METHODS:
        t = table[method]["travel_avg"]
        s = table[method]["shopping"]
        rows.append({
            "Method": DISPLAY_NAMES[method],
            "Travel CS Score": t["cs"],
            "Travel PS Score": t["ps"],
            "Travel Comp Score": t["comp"],
            "Travel Case Acc.": t["case"],
            "Shopping Match Score": s["match"],
            "Shopping Case Acc.": s["case"],
        })
    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(table, ensure_ascii=False, indent=2))
    if args.plot:
        import matplotlib.pyplot as plt
        labels = ["CS Score", "PS Score", "Comp Score", "Case Acc.", "Match Score", "Case Acc."]
        values = []
        for method in METHODS:
            t = table[method]["travel_avg"]
            s = table[method]["shopping"]
            values.append([t["cs"], t["ps"], t["comp"], t["case"], s["match"], s["case"]])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        colors = {"vanilla": "#7f8c8d", "oagents_best4": "#3498db", "statetrace_dsr": "#e67e22"}
        x = range(4)
        for i, method in enumerate(METHODS):
            axes[0].bar([v + (i - 1) * 0.23 for v in x], values[i][:4], width=0.22, label=DISPLAY_NAMES[method], color=colors[method])
        axes[0].set_xticks(list(x), labels[:4]); axes[0].set_ylim(0, 100); axes[0].set_ylabel("Percent")
        x2 = range(2)
        for i, method in enumerate(METHODS):
            axes[1].bar([v + (i - 1) * 0.23 for v in x2], values[i][4:], width=0.22, label=DISPLAY_NAMES[method], color=colors[method])
        axes[1].set_xticks(list(x2), labels[4:]); axes[1].set_ylim(0, 100); axes[1].set_ylabel("Percent")
        axes[1].legend(frameon=False)
        fig.savefig(args.plot, dpi=220)


if __name__ == "__main__":
    main()
