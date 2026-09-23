"""Public, deterministic StateTrace-PROT overlay.

PROT operates on already completed candidate trajectories.  It is deliberately
API-free: scores, task definitions, evaluator fields, and hidden state never
enter the graph or the decision.  The module is useful both as the runtime
selector's structured input and as an offline audit tool.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from trace_ir import extract_trace_ir, public_trajectory


RISK_TYPES = (
    "exact_failed_repeat",
    "unverified_termination",
    "missing_readback",
    "post_success_mutation",
    "target_drift",
)


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event.get("tool"),
        event.get("operation"),
        json.dumps(event.get("arguments"), sort_keys=True, default=str),
        tuple(event.get("entity_refs") or ()),
    )


def _same_entity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a = set(left.get("entity_refs") or ())
    b = set(right.get("entity_refs") or ())
    return bool(a and b and a & b)


def _graph_events(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize public events into a compact transition graph representation."""
    events = []
    for index, raw in enumerate(trace.get("events") or ()):
        event = {
            "event_id": str(raw.get("event_id") or f"call-{index:04d}"),
            "tool": str(raw.get("tool") or ""),
            "operation": str(raw.get("operation") or "UNKNOWN"),
            "entity_refs": sorted(str(x) for x in (raw.get("entity_refs") or ())),
            "arguments": copy.deepcopy(raw.get("arguments") or {}),
            "result_status": str(raw.get("result_status") or "UNKNOWN"),
            "evidence_status": str(raw.get("evidence_status") or "unknown"),
            "readback_event_ids": sorted(str(x) for x in (raw.get("readback_event_ids") or ())),
        }
        events.append(event)
    return events


def _risk_profile(events: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    risks = {name: [] for name in RISK_TYPES}
    for index, event in enumerate(events):
        if index and event.get("operation") == events[index - 1].get("operation") == "WRITE":
            previous = events[index - 1]
            if event.get("result_status") == "DETERMINISTIC_REJECTION" and _event_key(event) == _event_key(previous):
                risks["exact_failed_repeat"].append(str(event["event_id"]))
        if event.get("result_status") == "DETERMINISTIC_REJECTION":
            later = list(events[index + 1:])
            if not later or all(item.get("operation") != "READ" for item in later):
                risks["unverified_termination"].append(str(event["event_id"]))
            if any(item.get("operation") == "WRITE" and _same_entity(event, item) for item in later):
                # A subsequent write is only risky when no successful read-back
                # establishes the failed entity in between.
                between = [item for item in later if item.get("operation") == "READ"]
                if not any(_same_entity(event, item) and item.get("result_status") == "READBACK_CONFIRMED" for item in between):
                    risks["target_drift"].append(str(event["event_id"]))
        if event.get("operation") == "WRITE" and event.get("result_status") == "ACKNOWLEDGED" and not event.get("readback_event_ids"):
            risks["missing_readback"].append(str(event["event_id"]))
        if index and event.get("operation") == "WRITE" and events[index - 1].get("result_status") == "READBACK_CONFIRMED":
            risks["post_success_mutation"].append(str(event["event_id"]))
    return risks


def compile_public_graph(trajectory: Mapping[str, Any], tool_schemas: Any = None) -> dict[str, Any]:
    """Compile public TraceIR plus directed event transitions and risk evidence."""
    trace = extract_trace_ir(trajectory, tool_schemas)
    events = _graph_events(trace)
    edges = []
    for left, right in zip(events, events[1:]):
        edges.append({"from": left["event_id"], "to": right["event_id"], "relation": f"{left['operation']}->{right['operation']}"})
    return {
        "task_key": trajectory.get("task_key"),
        "events": events,
        "edges": edges,
        "risks": _risk_profile(events),
        "public_only": True,
    }


def first_recovery_divergence(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a divergence after comparable first public failure boundaries.

    Absolute event positions are not comparable: Best-of-K may take extra
    reads before reaching the same failure.  Aligning each candidate at its
    own first deterministic rejection prevents a pre-failure planning
    difference from being mislabeled as a recovery divergence.
    """
    a, b = list(left.get("events") or ()), list(right.get("events") or ())
    failure_a = next((i for i, event in enumerate(a) if event.get("result_status") == "DETERMINISTIC_REJECTION"), None)
    failure_b = next((i for i, event in enumerate(b) if event.get("result_status") == "DETERMINISTIC_REJECTION"), None)
    if failure_a is None or failure_b is None:
        return None
    # A recovery comparison is meaningful only when the public failure refers
    # to the same operation and public entity in both candidates.
    if a[failure_a].get("tool") != b[failure_b].get("tool"):
        return None
    refs_a = set(a[failure_a].get("entity_refs") or ())
    refs_b = set(b[failure_b].get("entity_refs") or ())
    if refs_a and refs_b and not refs_a & refs_b:
        return None
    suffix_a, suffix_b = a[failure_a + 1:], b[failure_b + 1:]
    limit = min(len(suffix_a), len(suffix_b))
    for index in range(limit):
        if _event_key(suffix_a[index]) == _event_key(suffix_b[index]):
            continue
        return {
            "index_after_failure": index,
            "left_failure_event_id": a[failure_a].get("event_id"),
            "right_failure_event_id": b[failure_b].get("event_id"),
            "left_event_id": suffix_a[index].get("event_id"),
            "right_event_id": suffix_b[index].get("event_id"),
            "public_failure_before_divergence": True,
        }
    if len(suffix_a) != len(suffix_b):
        return {
            "index_after_failure": limit,
            "left_failure_event_id": a[failure_a].get("event_id"),
            "right_failure_event_id": b[failure_b].get("event_id"),
            "left_event_id": suffix_a[limit].get("event_id") if len(suffix_a) > limit else None,
            "right_event_id": suffix_b[limit].get("event_id") if len(suffix_b) > limit else None,
            "public_failure_before_divergence": True,
        }
    return None


def evidence_features(graph: Mapping[str, Any], divergence: Mapping[str, Any] | None) -> dict[str, float]:
    risks = graph.get("risks") or {}
    events = list(graph.get("events") or ())
    confirmed = sum(event.get("evidence_status") == "confirmed" for event in events)
    return {
        "confirmed_effects": float(confirmed),
        "readback_count": float(sum(bool(event.get("readback_event_ids")) for event in events)),
        "risk_count": float(sum(len(risks.get(name) or ()) for name in RISK_TYPES)),
        "exact_failed_repeat": float(len(risks.get("exact_failed_repeat") or ())),
        "unverified_termination": float(len(risks.get("unverified_termination") or ())),
        "missing_readback": float(len(risks.get("missing_readback") or ())),
        "post_success_mutation": float(len(risks.get("post_success_mutation") or ())),
        "target_drift": float(len(risks.get("target_drift") or ())),
        "has_recovery_divergence": float(divergence is not None),
    }


def build_payload(request: str, candidates: Sequence[Mapping[str, Any]], graphs: Sequence[Mapping[str, Any]], divergence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Create the selector payload without carrying score/runtime metadata."""
    return {
        "visible_request": request,
        "first_public_recovery_divergence": copy.deepcopy(divergence),
        "candidates": [
            {
                "candidate_index": index,
                "trajectory": public_trajectory(candidate),
                "public_state_graph": {
                    "events": graph.get("events", []),
                    "edges": graph.get("edges", []),
                    "risks": graph.get("risks", {}),
                },
                "evidence_features": evidence_features(graph, divergence),
            }
            for index, (candidate, graph) in enumerate(zip(candidates, graphs))
        ],
    }


def choose_by_public_evidence(graphs: Sequence[Mapping[str, Any]], base_index: int = 0) -> tuple[int, dict[str, Any]]:
    """Make a conservative deterministic overlay decision.

    A change is allowed only when a recovery divergence exists and one candidate
    has strictly more confirmed/read-back evidence with no additional risks.
    Otherwise the existing Base/RE choice is preserved.
    """
    if len(graphs) != 2:
        raise ValueError("PROT requires exactly two candidates")
    divergence = first_recovery_divergence(graphs[0], graphs[1])
    if divergence is None:
        return base_index, {"used": False, "reason": "no_public_recovery_divergence", "supporting_event_ids": []}
    features = [evidence_features(graph, divergence) for graph in graphs]
    def key(item: Mapping[str, float]) -> tuple[float, float, float]:
        return (item["confirmed_effects"], item["readback_count"], -item["risk_count"])
    if key(features[0]) == key(features[1]):
        return base_index, {"used": False, "reason": "evidence_tie_preserve_base", "supporting_event_ids": []}
    winner = 0 if key(features[0]) > key(features[1]) else 1
    loser = 1 - winner
    if features[winner]["risk_count"] > features[loser]["risk_count"]:
        return base_index, {"used": False, "reason": "winner_has_additional_risk_preserve_base", "supporting_event_ids": []}
    supporting = [str(event.get("event_id")) for event in graphs[winner].get("events", []) if event.get("evidence_status") == "confirmed"]
    return winner, {"used": winner != base_index, "reason": "strict_public_evidence_advantage", "supporting_event_ids": supporting[:8]}
