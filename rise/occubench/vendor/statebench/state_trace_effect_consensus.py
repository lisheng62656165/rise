"""Outcome-blind consensus selection in public state-effect space.

The selector consumes several independently completed trajectories and an
existing strong fallback choice.  It projects only public tool interactions,
finds effect slots supported by at least two trajectories, and changes the
fallback only when one candidate is the unique minimum-risk effect medoid.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from state_trace_cort_train_replay import read_rows
from state_trace_dor import find_defects, public_events
from trace_ir import public_trajectory


FORBIDDEN_FIELDS = {
    "task_completion_pass", "state_requirements_met", "task_requirements_met",
    "ux_score", "state_diff", "gold_action", "gold_state", "expected_state",
    "evaluator_label", "reward", "milestones", "target_state",
}
IGNORED_EFFECT_FIELDS = {
    "confirm", "reason", "message", "detail", "details", "description",
    "error", "exception", "success", "ok",
}


def assert_public_payload(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden field at {path}.{key}")
            assert_public_payload(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_public_payload(child, f"{path}[{index}]")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _is_entity_field(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in (
        "id", "order", "booking", "customer", "flight", "product", "item", "rental",
    ))


def _scalar_leaves(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    leaves: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            if key.lower() in IGNORED_EFFECT_FIELDS or _is_entity_field(key):
                continue
            path = f"{prefix}.{key}" if prefix else key
            leaves.extend(_scalar_leaves(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            leaves.extend(_scalar_leaves(child, f"{prefix}[]"))
    elif value is not None and prefix:
        leaves.append((prefix, _canonical(value)))
    return leaves


def _raw_calls(trajectory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = []
    for message in trajectory.get("conversation") or ():
        if message.get("role") != "assistant":
            continue
        calls.extend(call for call in message.get("tool_calls") or () if isinstance(call, Mapping))
    return calls


def public_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        "task_key": str(row.get("task_key") or ""),
        "domain": str(row.get("domain") or ""),
        **public_trajectory(row),
    }
    assert_public_payload(projection)
    return projection


def compile_effect_ledger(row: Mapping[str, Any], schemas: Any = None) -> dict[str, Any]:
    trajectory = public_projection(row)
    events = public_events(trajectory, schemas)
    calls = _raw_calls(trajectory)
    if len(events) != len(calls):
        raise ValueError("public event/call count mismatch")

    slots: dict[str, str] = {}
    slot_event_ids: dict[str, list[str]] = defaultdict(list)
    receipts = 0
    accepted_writes = 0
    for event, call in zip(events, calls):
        if event.get("operation") != "WRITE" or event.get("semantic_result") != "ACKNOWLEDGED":
            continue
        accepted_writes += 1
        refs = tuple(sorted(map(str, event.get("entity_refs") or ())))
        target = _canonical(refs)
        result = call.get("result")
        arguments = call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        leaves = _scalar_leaves(arguments, "argument") + _scalar_leaves(result, "result")
        if event.get("readback_event_ids"):
            receipts += 1
        for field, value in leaves:
            slot = _canonical([str(event.get("tool") or ""), target, field])
            slots[slot] = value
            slot_event_ids[slot].append(str(event.get("event_id") or ""))

    defects = find_defects(events)
    ledger = {
        "task_key": trajectory["task_key"],
        "effect_slots": slots,
        "slot_event_ids": dict(slot_event_ids),
        "accepted_writes": accepted_writes,
        "readback_confirmed_writes": receipts,
        "recovery_defects": defects,
        "deterministic_rejections": sum(
            event.get("operation") == "WRITE" and event.get("semantic_result") == "REJECTED"
            for event in events
        ),
        "public_only": True,
    }
    assert_public_payload(ledger)
    return ledger


def effect_consensus(
    ledgers: Sequence[Mapping[str, Any]], fallback_index: int,
) -> tuple[int, dict[str, Any]]:
    if fallback_index not in range(len(ledgers)):
        raise ValueError("fallback index outside candidate pool")
    values_by_slot: dict[str, Counter[str]] = defaultdict(Counter)
    for ledger in ledgers:
        for slot, value in dict(ledger.get("effect_slots") or {}).items():
            values_by_slot[str(slot)][str(value)] += 1

    majority: dict[str, str] = {}
    support_threshold = len(ledgers) // 2 + 1
    for slot, counts in values_by_slot.items():
        value, count = counts.most_common(1)[0]
        if count >= support_threshold:
            majority[slot] = value

    evidence = {
        "fallback_index": fallback_index,
        "candidate_count": len(ledgers),
        "majority_effect_slots": len(majority),
        "support_threshold": support_threshold,
        "candidate_scores": [],
        "changed": False,
        "abstention_reason": None,
        "public_only": True,
    }
    if not majority:
        evidence["abstention_reason"] = "no_majority_effect_slots"
        return fallback_index, evidence

    ranking = []
    for index, ledger in enumerate(ledgers):
        slots = dict(ledger.get("effect_slots") or {})
        matched = [slot for slot, value in majority.items() if slots.get(slot) == value]
        contradicted = [slot for slot, value in majority.items() if slot in slots and slots[slot] != value]
        missing = [slot for slot in majority if slot not in slots]
        defects = len(ledger.get("recovery_defects") or ())
        rejections = int(ledger.get("deterministic_rejections") or 0)
        confirmations = int(ledger.get("readback_confirmed_writes") or 0)
        # Lexicographic utility avoids outcome-tuned scalar weights.
        utility = (len(matched), -len(contradicted), -len(missing), -defects, -rejections, confirmations)
        row = {
            "candidate_index": index,
            "matched_majority_slots": len(matched),
            "contradicted_majority_slots": len(contradicted),
            "missing_majority_slots": len(missing),
            "recovery_defects": defects,
            "deterministic_rejections": rejections,
            "readback_confirmed_writes": confirmations,
            "supporting_event_ids": sorted({
                event_id for slot in matched
                for event_id in (ledger.get("slot_event_ids") or {}).get(slot, ())
            }),
            "utility": list(utility),
        }
        evidence["candidate_scores"].append(row)
        ranking.append((utility, index))

    best_utility = max(utility for utility, _ in ranking)
    winners = [index for utility, index in ranking if utility == best_utility]
    if len(winners) != 1:
        evidence["abstention_reason"] = "non_unique_effect_medoid"
        return fallback_index, evidence
    selected = winners[0]
    evidence["selected_index"] = selected
    evidence["changed"] = selected != fallback_index
    return selected, evidence


def fallback_label(row: Mapping[str, Any]) -> str:
    source = row.get("state_trace_arc_selected_source")
    if source == "anchor":
        return "anchor"
    if source == "challenger":
        return "generic"
    value = row.get("state_trace_effect_selected_source")
    if isinstance(value, str):
        return value
    raise ValueError(f"missing Generic fallback source for {row.get('task_key')}")


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=DIR")
    return label, Path(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statebench-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) < 3:
        raise ValueError("effect consensus requires at least three candidates")
    labels = [label for label, _ in args.candidate]
    if labels[:2] != ["anchor", "generic"] or len(labels) != len(set(labels)):
        raise ValueError("first candidates must be unique labels anchor,generic")

    candidates = [read_rows(path) for _, path in args.candidate]
    fallbacks = read_rows(args.fallback_dir)
    input_key_sets = [set(rows) for rows in candidates] + [set(fallbacks)]
    keys = set.intersection(*input_key_sets)
    if not keys:
        raise ValueError("candidate and fallback pools have no common tasks")
    excluded = sorted(set.union(*input_key_sets) - keys)

    sys.path.insert(0, str(args.statebench_root))
    from state_bench.domain import get_domain_config

    schemas = {
        domain: get_domain_config(domain).tool_schemas
        for domain in sorted({str(candidates[0][key].get("domain")) for key in keys})
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key in sorted(keys):
        rows = [candidate[key] for candidate in candidates]
        ledgers = [compile_effect_ledger(row, schemas.get(str(row.get("domain")))) for row in rows]
        fallback_source = fallback_label(fallbacks[key])
        fallback_index = labels.index(fallback_source)
        selected_index, evidence = effect_consensus(ledgers, fallback_index)
        output = {
            "schema_version": "state_trace_effect_consensus_choice_v1",
            "task_key": key,
            "domain": str(rows[0].get("domain") or ""),
            "candidate_labels": labels,
            "fallback_source": fallback_source,
            "selected_source": labels[selected_index],
            "changed": selected_index != fallback_index,
            "effect_consensus": evidence,
            "public_only": True,
            "online_outcome_used": False,
        }
        assert_public_payload(output)
        write_json(args.output_dir / f"{key.replace('::', '__')}.json", output)
        results.append(output)

    report = {
        "schema_version": "state_trace_effect_consensus_report_v1",
        "tasks": len(results),
        "input_task_counts": {
            **{label: len(rows) for label, rows in zip(labels, candidates)},
            "fallback": len(fallbacks),
        },
        "excluded_unpaired_tasks": excluded,
        "candidate_labels": labels,
        "selected_sources": dict(Counter(row["selected_source"] for row in results)),
        "fallback_sources": dict(Counter(row["fallback_source"] for row in results)),
        "changed": sum(bool(row["changed"]) for row in results),
        "abstentions": dict(Counter(
            row["effect_consensus"].get("abstention_reason") or "selected"
            for row in results
        )),
        "online_outcome_used": False,
    }
    assert_public_payload(report)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
