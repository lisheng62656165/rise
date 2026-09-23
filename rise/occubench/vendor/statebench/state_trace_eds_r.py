"""Public Recovery-Event Graph and frontier construction for EDS-R.

The module is deliberately outcome-free.  It turns the existing public event
ledger into a small dependency graph and emits an intervention card for the
next fresh rollout.  It does not infer gold state or label strict FRD online.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from state_trace_event_scaling import (
    analyze_public_events,
    event_scaling_message,
    fuse_event_ledgers,
)


RISK_TAGS = {
    "failure_boundary",
    "repeated_failed_mutation",
    "missing_readback_candidate",
    "pending_consent_or_commit",
    "unknown_evidence",
}

DECISION_RISK_TAGS = {
    "repeated_failed_mutation": {"repeated_failed_mutation"},
    "parameter_change_candidate": {"unknown_evidence"},
    "target_drift_candidate": {"unknown_evidence"},
    "order_violation_candidate": {"pending_consent_or_commit"},
    "premature_stop_candidate": {"unknown_evidence"},
}

HR_RISK_TYPES = {
    "repeated_failed_mutation": "repeated_failed_mutation",
    "parameter_change_candidate": "parameter_change_without_regrounding",
    "target_drift_candidate": "target_drift_without_reacquisition",
    "order_violation_candidate": "order_violation",
    "premature_stop_candidate": "premature_stop",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _positions(group: Mapping[str, Any]) -> list[int]:
    return [
        int(event.get("position", -1))
        for event in group.get("events") or ()
        if isinstance(event, Mapping) and int(event.get("position", -1)) >= 0
    ]


def _refs(group: Mapping[str, Any]) -> set[str]:
    return {str(value) for value in group.get("entity_refs") or ()}


def _decision_position(decision: Mapping[str, Any], raw_events: list[Mapping[str, Any]]) -> int:
    source_ids = {str(value) for value in decision.get("source_event_ids") or ()}
    positions = [
        int(event.get("position", -1))
        for event in raw_events
        if str(event.get("event_id") or "") in source_ids
    ]
    return min((position for position in positions if position >= 0), default=10**9)


def _risk_groups(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = []
    for group in ledger.get("transaction_groups") or ():
        tags = set(group.get("risk_tags") or ())
        if tags & RISK_TAGS:
            groups.append(copy.deepcopy(dict(group)))
    raw_events = list(ledger.get("raw_events") or ())
    for decision in ledger.get("recovery_decision_events") or ():
        kind = str(decision.get("kind") or "")
        tags = set(DECISION_RISK_TAGS.get(kind) or {"unknown_evidence"})
        source_ids = {str(value) for value in decision.get("source_event_ids") or ()}
        refs = {
            str(value)
            for event in raw_events
            if str(event.get("event_id") or "") in source_ids
            for value in event.get("entity_refs") or ()
        }
        groups.append({
            "event_id": str(decision.get("event_id") or "recovery-decision"),
            "member_event_ids": list(decision.get("member_event_ids") or ()),
            "source_event_ids": list(decision.get("source_event_ids") or ()),
            "entity_refs": sorted(refs),
            "risk_tags": sorted(tags),
            "status": "UNRESOLVED_PUBLIC_HYPOTHESIS",
            "events": [],
            "position": _decision_position(decision, raw_events),
            "decision_kind": kind,
            "public_evidence": decision.get("public_evidence"),
        })
    groups.sort(key=lambda group: (
        int(group.get("position", min(_positions(group) or [10**9]))),
        0 if group.get("decision_kind") else 1,
    ))
    groups.sort(key=lambda group: int(group.get("position", min(_positions(group) or [10**9]))))
    return groups


def build_event_graph(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact graph over public transaction groups."""
    groups = [copy.deepcopy(dict(group)) for group in ledger.get("transaction_groups") or ()]
    raw_events = list(ledger.get("raw_events") or ())
    for decision in ledger.get("recovery_decision_events") or ():
        source_ids = {str(value) for value in decision.get("source_event_ids") or ()}
        refs = {
            str(value)
            for event in raw_events
            if str(event.get("event_id") or "") in source_ids
            for value in event.get("entity_refs") or ()
        }
        groups.append({
            "event_id": str(decision.get("event_id") or "recovery-decision"),
            "member_event_ids": list(decision.get("member_event_ids") or ()),
            "entity_refs": sorted(refs),
            "operations": ["RECOVERY_DECISION"],
            "event_classes": ["recovery_decision"],
            "status": "UNRESOLVED_PUBLIC_HYPOTHESIS",
            "risk_tags": sorted(DECISION_RISK_TAGS.get(str(decision.get("kind") or "")) or {"unknown_evidence"}),
            "events": [],
            "position": _decision_position(decision, raw_events),
        })
    groups.sort(key=lambda group: (
        int(group.get("position", min(_positions(group) or [10**9]))),
        0 if group.get("operations") == ["RECOVERY_DECISION"] else 1,
    ))
    nodes = []
    edges = []
    for index, group in enumerate(groups):
        node_id = str(group.get("event_id") or f"group-{index:04d}")
        nodes.append({
            "event_id": node_id,
            "position": int(group.get("position", min(_positions(group) or [index]))),
            "event_class": list(group.get("event_classes") or ()),
            "operations": list(group.get("operations") or ()),
            "entity_refs": sorted(_refs(group)),
            "status": group.get("status"),
            "risk_tags": sorted(str(tag) for tag in group.get("risk_tags") or ()),
            "member_event_ids": list(group.get("member_event_ids") or ()),
        })
        if index:
            edges.append({"source": nodes[index - 1]["event_id"], "target": node_id, "relation": "followed_by"})
        if index:
            prior = nodes[index - 1]
            if set(prior["entity_refs"]) & set(nodes[-1]["entity_refs"]):
                if "verification" in {str(x).lower() for x in nodes[-1]["event_class"]}:
                    edges.append({"source": prior["event_id"], "target": node_id, "relation": "verifies"})
        if set(nodes[-1]["risk_tags"]) & RISK_TAGS:
            for prior in reversed(nodes[:-1]):
                if set(prior["entity_refs"]) & set(nodes[-1]["entity_refs"]):
                    edges.append({"source": node_id, "target": prior["event_id"], "relation": "recovery_after"})
                    break
    return {
        "schema_version": "state_trace_recovery_event_graph_v1",
        "nodes": nodes,
        "edges": edges,
        "public_only": True,
        "outcome_used": False,
    }


def _frontier_card(ledger: Mapping[str, Any], graph: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    nodes = list(graph.get("nodes") or ())
    risky = _risk_groups(ledger)
    cards = []
    for group in risky[:limit]:
        event_id = str(group.get("event_id") or "")
        node = next((item for item in nodes if item.get("event_id") == event_id), {})
        tags = set(group.get("risk_tags") or ())
        if "repeated_failed_mutation" in tags:
            risk_type = "repeated_failed_mutation"
        elif "failure_boundary" in tags:
            risk_type = "failure_boundary"
        elif "missing_readback_candidate" in tags:
            risk_type = "missing_verification"
        elif "pending_consent_or_commit" in tags:
            risk_type = "consent_or_commit_order"
        else:
            risk_type = "unknown_public_recovery_risk"
        cards.append({
            "frontier_event_id": event_id,
            "risk_type": risk_type,
            "entity_refs": list(node.get("entity_refs") or group.get("entity_refs") or ()),
            "preserve": [
                item["event_id"] for item in nodes
                if item.get("position", 10**9) < node.get("position", -1)
                and not set(item.get("risk_tags") or ()) & RISK_TAGS
            ][-3:],
            "repair": [event_id],
            "verify": [
                item["event_id"] for item in nodes
                if item.get("position", -1) > node.get("position", 10**9)
                and "verification" in {str(x).lower() for x in item.get("event_class") or ()}
                and set(item.get("entity_refs") or ()) & set(node.get("entity_refs") or ())
            ][:2],
            "avoid": [
                "repeat the rejected mutation with the same public arguments",
                "change the target object without reacquiring its identity",
                "stop before a consequential write is publicly verified",
            ],
            "public_evidence": {
                "member_event_ids": list(group.get("member_event_ids") or ()),
                "risk_tags": sorted(str(tag) for tag in tags),
            },
        })
    return cards


def build_eds_r_frontier(
    incumbent: Mapping[str, Any], schemas: Any,
    prior_disagreement: Mapping[str, Any] | None,
    max_event_groups: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = analyze_public_events(incumbent, schemas)
    if prior_disagreement:
        ledger["cross_round_disagreement"] = copy.deepcopy(dict(prior_disagreement))
        ledger["material_disagreement"] = bool(prior_disagreement.get("material_disagreement"))
    graph = build_event_graph(ledger)
    base = fuse_event_ledgers([ledger], max_groups=max_event_groups)
    frontier = _frontier_card(ledger, graph, max(1, min(4, max_event_groups)))
    base.update({
        "schema_version": "state_trace_eds_r_frontier_v1",
        "recovery_event_graph": graph,
        "recovery_frontier": frontier,
        "event_intervention": {
            "policy": "preserve supported prefix, repair earliest public recovery frontier, verify, then stop",
            "replay_prior_calls": False,
            "reacquire_current_state": True,
        },
        "frontier_policy": "earliest unresolved public recovery-risk event with an explicit preserve/repair/verify/avoid card",
        "public_only": True,
        "outcome_used": False,
    })
    base["instruction"] = (
        "Use the recovery frontier as a local intervention hypothesis. Preserve only supported prior "
        "event structure, reacquire current objects and arguments, repair the frontier conservatively, "
        "verify every consequential write, and preserve all user-facing obligations. Never replay a prior "
        "tool call as a script and never infer hidden evaluator state."
    )
    return ledger, base


def eds_r_message(frontier: Mapping[str, Any], round_index: int) -> str:
    return event_scaling_message(frontier, round_index)


def _event_status(group: Mapping[str, Any]) -> str:
    """Assign an outcome-free public status for conditional intervention."""
    tags = set(str(tag) for tag in group.get("risk_tags") or ())
    decision_kind = str(group.get("decision_kind") or "")
    if decision_kind in HR_RISK_TYPES or tags & {
        "repeated_failed_mutation", "failure_boundary", "missing_readback_candidate",
        "pending_consent_or_commit", "unknown_evidence",
    }:
        return "UNRESOLVED"
    if str(group.get("status") or "").upper() in {"SUCCESS", "VERIFIED", "RESOLVED"}:
        return "RESOLVED"
    return "OBSERVED"


def _hierarchical_events(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Turn flat public groups into typed task/transaction/recovery/action records."""
    records = []
    for group in ledger.get("transaction_groups") or ():
        status = _event_status(group)
        tags = set(str(tag) for tag in group.get("risk_tags") or ())
        decision_kind = str(group.get("decision_kind") or "")
        is_recovery = bool(decision_kind or tags & {
            "failure_boundary", "repeated_failed_mutation", "missing_readback_candidate",
            "pending_consent_or_commit", "unknown_evidence",
        })
        records.append({
            "event_id": str(group.get("event_id") or ""),
            "level": "recovery_episode" if is_recovery else "transaction",
            "transaction_refs": sorted(str(ref) for ref in group.get("entity_refs") or ()),
            "status": status,
            "rds_type": HR_RISK_TYPES.get(decision_kind, "public_recovery_risk" if is_recovery else None),
            "member_event_ids": list(group.get("member_event_ids") or ()),
            "risk_tags": sorted(tags),
            "operations": list(group.get("operations") or ()),
        })
    for decision in ledger.get("recovery_decision_events") or ():
        source_ids = set(str(value) for value in decision.get("source_event_ids") or ())
        member_ids = set(str(value) for value in decision.get("member_event_ids") or ())
        refs = sorted({
            str(ref)
            for raw in ledger.get("raw_events") or ()
            if str(raw.get("event_id") or "") in source_ids | member_ids
            for ref in raw.get("entity_refs") or ()
        })
        records.append({
            "event_id": str(decision.get("event_id") or ""),
            "level": "recovery_decision",
            "transaction_refs": refs,
            "status": "UNRESOLVED",
            "rds_type": HR_RISK_TYPES.get(str(decision.get("kind") or ""), "public_recovery_risk"),
            "member_event_ids": list(decision.get("member_event_ids") or ()),
            "risk_tags": [str(decision.get("kind") or "unknown_evidence")],
            "operations": ["RECOVERY_DECISION"],
        })
    return records


def _event_signature(event: Mapping[str, Any]) -> str:
    return _canonical({
        "level": event.get("level"),
        "rds_type": event.get("rds_type"),
        "transaction_refs": sorted(str(ref) for ref in event.get("transaction_refs") or ()),
    })


def _track_event_states(
    events: list[dict[str, Any]], prior_states: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Track public RDS recurrence across fresh rollouts without using outcomes."""
    prior = dict(prior_states or {})
    current: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("level") != "recovery_decision":
            continue
        signature = _event_signature(event)
        previous = prior.get(signature) or {}
        item = copy.deepcopy(event)
        item["signature"] = signature
        item["status"] = "REGRESSED" if previous.get("status") == "AVOIDED" else "UNRESOLVED"
        item["observations"] = int(previous.get("observations") or 0) + 1
        current[signature] = item
    for signature, previous in prior.items():
        if signature in current:
            continue
        avoided = copy.deepcopy(dict(previous))
        avoided["status"] = "AVOIDED"
        current[signature] = avoided
    tracked = [event for event in events if event.get("level") != "recovery_decision"]
    tracked.extend(current.values())
    return tracked, current


def build_eds_hr_frontier(
    incumbent: Mapping[str, Any], schemas: Any,
    prior_disagreement: Mapping[str, Any] | None,
    max_event_groups: int,
    prior_event_states: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a hierarchical, conditional recovery contract for the next rollout."""
    ledger = analyze_public_events(incumbent, schemas)
    if prior_disagreement:
        ledger["cross_round_disagreement"] = copy.deepcopy(dict(prior_disagreement))
        ledger["material_disagreement"] = bool(prior_disagreement.get("material_disagreement"))
    graph = build_event_graph(ledger)
    events, event_states = _track_event_states(_hierarchical_events(ledger), prior_event_states)
    material = bool(ledger.get("material_disagreement"))
    candidates = [
        event for event in events
        if event.get("level") == "recovery_decision"
        and event.get("status") in {"UNRESOLVED", "REGRESSED"}
        and (
            material
            or event.get("rds_type") in {
                "repeated_failed_mutation", "target_drift_without_reacquisition", "order_violation",
            }
        )
    ]
    candidates = candidates[:max(1, min(2, max_event_groups))]
    contracts = []
    for event in candidates:
        contracts.append({
            "event_id": event["event_id"],
            "level": event["level"],
            "status": event["status"],
            "rds_type": event["rds_type"],
            "preserve": ["all already supported task and transaction obligations"],
            "change": ["re-ground the public target and avoid repeating the failed decision"],
            "verify": ["read back every consequential mutation before stopping"],
            "coverage": ["complete remaining user-facing requirements outside this event"],
            "member_event_ids": event["member_event_ids"],
        })
    fusion = fuse_event_ledgers([ledger], max_groups=max_event_groups)
    fusion.update({
        "schema_version": "state_trace_eds_hr_v1",
        "hierarchical_events": events,
        "recovery_event_states": event_states,
        "recovery_event_graph": graph,
        "recovery_contracts": contracts,
        "conditional_intervention": not not contracts,
        "public_only": True,
        "outcome_used": False,
        "instruction": (
            "Treat these public event summaries as unresolved hypotheses. Re-ground every object, "
            "argument, order, consent decision, and result in the current fresh environment. Preserve "
            "supported effects, resolve recovery risks, and do not replay a prior action merely because "
            "it appears in the summary. When a recovery contract is present, change only that explicit "
            "RDS event while preserving all other task coverage."
        ),
    })
    return ledger, fusion


def eds_hr_message(frontier: Mapping[str, Any], round_index: int) -> str:
    return event_scaling_message(frontier, round_index)
