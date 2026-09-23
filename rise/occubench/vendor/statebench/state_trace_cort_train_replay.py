"""Train and replay the public-only CORT qualifier on existing candidates."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from state_trace_cort import (
    LinearEnergyModel,
    admissible_interventions,
    compile_effect_obligation_graph,
    fit_pairwise_energy,
    qualify_proposal,
    semantics_preserving_control,
)


IGNORED_NAMES = {"summary.json", "score_summary.json", "report.json", "report_scores.json"}


def read_rows(directory: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in IGNORED_NAMES or path.name.endswith((".error.json", "summary.json", "report.json", "report_scores.json")):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        key = str(row.get("task_key") or "")
        if not key or key in rows:
            raise ValueError(f"invalid or duplicate task_key in {path}")
        rows[key] = row
    return rows


def source_of(row: Mapping[str, Any]) -> str:
    for field in ("state_trace_re_selected_source", "trajectory_listwise_selected_source"):
        source = row.get(field)
        if source in {"vanilla", "best_of_k2"}:
            return str(source)
    raise ValueError(f"missing source for {row.get('task_key')}")


def calibration_keys(keys: list[str], rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_domain[str(rows[key].get("domain") or "unknown")].append(key)
    return {key for domain_keys in by_domain.values() for index, key in enumerate(sorted(domain_keys)) if index % 5 == 0}


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def train_and_replay(
    vanilla: Mapping[str, Mapping[str, Any]],
    best: Mapping[str, Mapping[str, Any]],
    base: Mapping[str, Mapping[str, Any]],
    proposals: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Any],
    *, use_transport: bool = True,
) -> tuple[LinearEnergyModel, dict[str, dict[str, Any]], dict[str, Any]]:
    key_sets = [set(rows) for rows in (vanilla, best, base, proposals)]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError({"counts": [len(keys) for keys in key_sets], "intersection": len(set.intersection(*key_sets))})
    keys = sorted(key_sets[0])
    graphs = {
        key: [
            compile_effect_obligation_graph(vanilla[key], schemas.get(str(vanilla[key].get("domain")))),
            compile_effect_obligation_graph(best[key], schemas.get(str(best[key].get("domain")))),
        ]
        for key in keys
    }
    calibration = calibration_keys(keys, vanilla)
    train_pairs, calibration_pairs = [], []
    intervention_counts = Counter()
    for key in keys:
        target = calibration_pairs if key in calibration else train_pairs
        for graph in graphs[key]:
            for intervention in admissible_interventions(graph):
                target.append((graph, intervention["graph"]))
                intervention_counts[intervention["type"]] += 1
    model = fit_pairwise_energy(train_pairs, use_transport=use_transport)
    calibration_margins = [model.comparison_improvement(harmful, clean) for clean, harmful in calibration_pairs]
    positive_margins = [value for value in calibration_margins if value > 0]
    invariance_errors = [
        abs(model.comparison_improvement(graph, semantics_preserving_control(graph)))
        for key in calibration for graph in graphs[key]
    ]
    model.threshold = max(invariance_errors, default=0.0)

    outputs: dict[str, dict[str, Any]] = {}
    proposal_disagreements = selected_changes = 0
    selected_sources = Counter()
    for key in keys:
        base_source = source_of(base[key])
        proposal_source = proposals[key].get("state_trace_re_proposed_source")
        if proposal_source not in {"vanilla", "best_of_k2"}:
            proposal_source = base_source
        base_index = 0 if base_source == "vanilla" else 1
        proposal_index = 0 if proposal_source == "vanilla" else 1
        selected_index, evidence = qualify_proposal(graphs[key], base_index, proposal_index, model)
        selected_source = "vanilla" if selected_index == 0 else "best_of_k2"
        proposal_disagreements += proposal_source != base_source
        selected_changes += selected_source != base_source
        selected_sources[selected_source] += 1
        selected_row = best[key] if selected_source == "best_of_k2" else vanilla[key]
        output = copy.deepcopy(selected_row)
        output.update({
            "state_trace_cort_base_source": base_source,
            "state_trace_cort_proposed_source": proposal_source,
            "state_trace_cort_selected_source": selected_source,
            "state_trace_cort_qualified": bool(evidence["qualified"]),
            "state_trace_cort_energy": evidence,
            "state_trace_cort_public_only": True,
            "online_gold_or_evaluator_used": False,
            "agent_runtime_context_exposed": False,
        })
        outputs[key] = output

    report = {
        "tasks": len(keys),
        "training_task_count": len(keys) - len(calibration),
        "calibration_task_count": len(calibration),
        "train_intervention_pairs": len(train_pairs),
        "calibration_intervention_pairs": len(calibration_pairs),
        "intervention_counts": dict(intervention_counts),
        "positive_calibration_margin_rate": (
            sum(value > 0 for value in calibration_margins) / len(calibration_margins)
            if calibration_margins else 0.0
        ),
        "calibration_violation_margin_p10": quantile(positive_margins, 0.10),
        "max_invariance_error": max(invariance_errors, default=0.0),
        "model": model.as_dict(),
        "use_transport": use_transport,
        "proposal_disagreements": proposal_disagreements,
        "selected_changes": selected_changes,
        "selected_sources": dict(selected_sources),
        "online_outcome_used": False,
    }
    return model, outputs, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statebench-root", type=Path, required=True)
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--best-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--proposal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-transport", action="store_true")
    args = parser.parse_args()

    vanilla, best = read_rows(args.vanilla_dir), read_rows(args.best_dir)
    sys.path.insert(0, str(args.statebench_root))
    from state_bench.domain import get_domain_config

    schemas = {
        domain: get_domain_config(domain).tool_schemas
        for domain in sorted({str(row.get("domain")) for row in vanilla.values()})
    }
    model, outputs, report = train_and_replay(
        vanilla, best, read_rows(args.base_dir), read_rows(args.proposal_dir), schemas,
        use_transport=not args.no_transport,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, row in outputs.items():
        path = args.output_dir / f"{key.replace('::', '__')}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(model.as_dict(), indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
