#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def delta(method: float, vanilla: float) -> float:
    return method - vanilla


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f}pp"


def shopping_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    stats = payload["overall_statistics"]
    total = int(stats["total_cases"])
    successful = int(stats["successful_cases"])
    return {
        "total_cases": total,
        "successful_cases": successful,
        "task_success_rate": successful / total if total else 0.0,
        "overall_match_rate": float(stats["overall_match_rate"]),
        "incomplete_rate": float(stats["incomplete_rate"]),
        "valid": bool(stats["valid"]),
    }


def travel_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "total_test_samples": int(payload["total_test_samples"]),
        "plan_files_found": int(payload["plan_files_found"]),
        "evaluation_success_count": int(payload["evaluation_success_count"]),
        "evaluation_failed_count": int(payload["evaluation_failed_count"]),
        "delivery_rate": float(metrics["delivery_rate"]),
        "commonsense_score": float(metrics["commonsense_score"]),
        "personalized_score": float(metrics["personalized_score"]),
        "composite_score": float(metrics["composite_score"]),
        "case_acc": float(metrics["case_acc"]),
    }


def validate_full_test(result: dict[str, Any]) -> None:
    for variant, metrics in result["shopping"].items():
        if metrics["total_cases"] != 120 or not metrics["valid"]:
            raise RuntimeError(f"Incomplete Shopping result for {variant}: {metrics}")
    for language, variants in result["travel"].items():
        for variant, metrics in variants.items():
            coverage = (
                metrics["total_test_samples"],
                metrics["plan_files_found"],
                metrics["evaluation_success_count"],
                metrics["evaluation_failed_count"],
            )
            if coverage != (120, 120, 120, 0):
                raise RuntimeError(
                    f"Incomplete Travel result for {language}/{variant}: {coverage}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.method_output.resolve()
    model = "nemotron-3.5-lightning-30b-a3b"

    shopping_root = root / "shoppingplanning" / "result_report"
    shopping = {
        "vanilla": shopping_metrics(load(shopping_root / "nemotron_trial1_vanilla" / "summary_report.json")),
        "statetrace_dsr": shopping_metrics(load(shopping_root / "nemotron_trial1_statetrace" / "summary_report.json")),
    }

    travel: dict[str, dict[str, Any]] = {}
    for language in ("zh", "en"):
        vanilla_path = (
            root
            / "travel_runs"
            / "nemotron_vanilla_a_trial1"
            / f"{model}_{language}"
            / "evaluation"
            / "evaluation_summary.json"
        )
        method_path = out / f"travel_{language}" / "final" / "evaluation" / "evaluation_summary.json"
        travel[language] = {
            "vanilla": travel_metrics(load(vanilla_path)),
            "statetrace_dsr": travel_metrics(load(method_path)),
        }

    result = {
        "protocol": {
            "split": "full DeepPlanning cohorts used as test",
            "trial_count": 1,
            "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
            "method": "StateTrace-EDS-ECA-DeepPlanning (StateTrace-DSR adaptation)",
        },
        "shopping": shopping,
        "travel": travel,
    }
    validate_full_test(result)

    eval_dir = out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_dir / "vanilla_vs_statetrace_dsr.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DeepPlanning Full-Test Trial: Vanilla vs StateTrace-DSR",
        "",
        "One trial using `nvidia/nemotron-3.5-lightning-30b-a3b`; each complete cohort is treated as test.",
        "",
        "## Shopping Planning",
        "",
        "| Metric | Vanilla | StateTrace-DSR | Delta |",
        "|---|---:|---:|---:|",
    ]
    sv, sm = shopping["vanilla"], shopping["statetrace_dsr"]
    lines.extend(
        [
            f"| Successful cases | {sv['successful_cases']}/120 | {sm['successful_cases']}/120 | {sm['successful_cases'] - sv['successful_cases']:+d} |",
            f"| Task success rate | {pct(sv['task_success_rate'])} | {pct(sm['task_success_rate'])} | {pp(delta(sm['task_success_rate'], sv['task_success_rate']))} |",
            f"| Overall match rate | {pct(sv['overall_match_rate'])} | {pct(sm['overall_match_rate'])} | {pp(delta(sm['overall_match_rate'], sv['overall_match_rate']))} |",
            f"| Incomplete rate | {pct(sv['incomplete_rate'])} | {pct(sm['incomplete_rate'])} | {pp(delta(sm['incomplete_rate'], sv['incomplete_rate']))} |",
        ]
    )

    for language, title in (("zh", "Travel Planning ZH"), ("en", "Travel Planning EN")):
        vanilla = travel[language]["vanilla"]
        method = travel[language]["statetrace_dsr"]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Metric | Vanilla | StateTrace-DSR | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("delivery_rate", "Delivery rate"),
            ("commonsense_score", "Commonsense weighted"),
            ("personalized_score", "Personalized constraints"),
            ("composite_score", "Composite score"),
            ("case_acc", "Case accuracy"),
        ):
            lines.append(f"| {label} | {pct(vanilla[key])} | {pct(method[key])} | {pp(delta(method[key], vanilla[key]))} |")

    markdown_path = eval_dir / "VANILLA_VS_STATETRACE_DSR.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
