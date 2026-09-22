"""Extracted from the experimental state_trace_event_scaling.py; algorithm bodies unchanged."""

from __future__ import annotations

import copy

import json

from collections import Counter

from typing import Any, Mapping, Sequence

from state_trace_dor import public_events

from event_shapes import argument_shape

from public_evidence import assert_public_payload, public_projection

from state_trace_sdcr import build_disagreement_packet

def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def _refs(event: Mapping[str, Any]) -> set[str]:
    return {str(value) for value in event.get("entity_refs") or ()}

def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    operation = str(event.get("operation") or "UNKNOWN")
    result = str(event.get("semantic_result") or "UNKNOWN")
    interaction = str(event.get("interaction_status") or "NONE")
    if result in {"REJECTED", "FAILED", "ERROR", "UNKNOWN"}:
        event_class = "failure"
    elif operation in {"WRITE", "CREATE", "UPDATE", "DELETE", "MUTATE"}:
        event_class = "mutation"
    elif operation in {"READ", "LIST", "SEARCH", "GET", "LOOKUP"}:
        event_class = "verification" if event.get("nearest_write_event_id") else "observation"
    elif interaction == "PREVIEW":
        event_class = "consent_boundary"
    else:
        event_class = "other"
    return {
        "event_id": str(event.get("event_id") or ""),
        "position": int(event.get("call_index", event.get("position", -1))),
        "operation": operation,
        "event_class": event_class,
        "tool": str(event.get("tool") or ""),
        "entity_refs": sorted(map(str, event.get("entity_refs") or ())),
        "arguments": copy.deepcopy(dict(event.get("arguments") or {})),
        "result": result,
        "interaction": interaction,
        "user_index": event.get("nearest_user_index"),
    }

def _status(events: Sequence[Mapping[str, Any]]) -> str:
    results = {str(event.get("semantic_result") or "UNKNOWN") for event in events}
    if results & {"REJECTED", "FAILED", "ERROR", "UNKNOWN"}:
        return "FAILURE"
    if results <= {"ACKNOWLEDGED", "SUCCESS", "PREVIEW"}:
        return "SUCCESS"
    return "MIXED"

def _risk_tags(events: Sequence[Mapping[str, Any]]) -> list[str]:
    tags: set[str] = set()
    for index, event in enumerate(events):
        result = str(event.get("semantic_result") or "UNKNOWN")
        if result in {"REJECTED", "FAILED", "ERROR"}:
            tags.add("failure_boundary")
        if result == "UNKNOWN":
            tags.add("unknown_evidence")
        if event.get("interaction_status") == "PREVIEW":
            tags.add("pending_consent_or_commit")
        if event.get("operation") == "WRITE" and result == "ACKNOWLEDGED":
            if not event.get("readback_event_ids"):
                tags.add("missing_readback_candidate")
        if index:
            previous = events[index - 1]
            same_call = (
                str(previous.get("tool")) == str(event.get("tool"))
                and _refs(previous) == _refs(event)
                and dict(previous.get("arguments") or {}) == dict(event.get("arguments") or {})
            )
            if same_call and result in {"REJECTED", "FAILED", "ERROR"}:
                tags.add("repeated_failed_mutation")
    return sorted(tags)

def _make_groups(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create contiguous transaction groups, retaining event-level provenance."""
    groups: list[list[Mapping[str, Any]]] = []
    for event in events:
        refs = _refs(event)
        if groups:
            prior = groups[-1]
            prior_refs = set().union(*(_refs(item) for item in prior)) if prior else set()
            same_user = event.get("nearest_user_index") == prior[-1].get("nearest_user_index")
            related = (
                event.get("tool") == prior[-1].get("tool")
                or event.get("operation") == prior[-1].get("operation") and bool(refs & prior_refs)
            )
            if same_user and related:
                prior.append(event)
                continue
        groups.append([event])

    output = []
    for index, group in enumerate(groups):
        payload = [_event_payload(event) for event in group]
        refs = sorted(set().union(*(_refs(event) for event in group)) if group else set())
        output.append({
            "event_id": f"group-{index:04d}",
            "level": "transaction",
            "member_event_ids": [item["event_id"] for item in payload],
            "entity_refs": refs,
            "operations": sorted({item["operation"] for item in payload}),
            "tools": sorted({item["tool"] for item in payload}),
            "status": _status(group),
            "event_classes": sorted({item["event_class"] for item in payload}),
            "risk_tags": _risk_tags(group),
            "events": payload,
        })
    return output

def _make_recovery_windows(events: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build coarse recovery windows around public failure/consent boundaries."""
    windows = []
    for group in groups:
        tags = set(group.get("risk_tags") or ())
        if not tags & {"failure_boundary", "repeated_failed_mutation", "pending_consent_or_commit", "missing_readback_candidate"}:
            continue
        member_ids = list(group.get("member_event_ids") or ())
        positions = [index for index, event in enumerate(events) if str(event.get("event_id")) in member_ids]
        if not positions:
            continue
        start, end = max(0, min(positions) - 1), min(len(events), max(positions) + 3)
        window_events = list(events[start:end])
        windows.append({
            "event_id": f"recovery-{len(windows):04d}",
            "level": "recovery_window",
            "boundary_group": group["event_id"],
            "member_event_ids": [str(event.get("event_id") or "") for event in window_events],
            "status": _status(window_events),
            "event_classes": sorted({item["event_class"] for item in (_event_payload(event) for event in window_events)}),
            "risk_tags": sorted(set().union(*(set(item.get("risk_tags") or ()) for item in groups if item["event_id"] == group["event_id"]))),
            "events": [_event_payload(event) for event in window_events],
        })
    return windows

def _same_entity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_refs = _refs(left)
    right_refs = _refs(right)
    return bool(left_refs and right_refs and left_refs & right_refs)

def _same_call(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        str(left.get("tool")) == str(right.get("tool"))
        and _refs(left) == _refs(right)
        and dict(left.get("arguments") or {}) == dict(right.get("arguments") or {})
    )

def _make_recovery_decision_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract public recovery decisions without inferring gold correctness.

    These records describe observable candidates for recovery errors.  They are
    deliberately not strict-FRD labels: the final consequence is only known to
    the offline evaluator, and the online scaler must treat every record as a
    hypothesis to re-ground.
    """
    failure_statuses = {"REJECTED", "FAILED", "ERROR"}
    output: list[dict[str, Any]] = []
    for index, failure in enumerate(events):
        if (
            failure.get("operation") != "WRITE"
            or str(failure.get("semantic_result") or "UNKNOWN") not in failure_statuses
        ):
            continue
        later = list(events[index + 1:index + 9])
        related = [event for event in later if _same_entity(failure, event)]
        recovery_entry = [
            event for event in related
            if event.get("operation") in {"READ", "WRITE", "CREATE", "UPDATE", "DELETE", "MUTATE"}
        ]
        if recovery_entry:
            output.append({
                "event_id": f"recovery-decision-{len(output):04d}",
                "level": "recovery_decision",
                "kind": "recovery_entry",
                "source_event_ids": [str(failure.get("event_id") or "")],
                "member_event_ids": [str(event.get("event_id") or "") for event in recovery_entry],
                "public_evidence": "failure followed by same-entity read or mutation",
                "risk": "recovery_window_opened",
            })
        else:
            output.append({
                "event_id": f"recovery-decision-{len(output):04d}",
                "level": "recovery_decision",
                "kind": "premature_stop_candidate",
                "source_event_ids": [str(failure.get("event_id") or "")],
                "member_event_ids": [],
                "public_evidence": "failed write has no later same-entity public action in window",
                "risk": "unresolved_recovery",
            })
            continue

        for candidate in recovery_entry:
            kind = None
            evidence = ""
            if candidate.get("operation") == "WRITE" and _same_call(failure, candidate):
                kind = "repeated_failed_mutation"
                evidence = "same public tool, entity references, and arguments after rejection"
            elif candidate.get("operation") == "WRITE" and candidate.get("tool") == failure.get("tool") and _same_entity(failure, candidate):
                kind = "parameter_change_candidate"
                evidence = "same public tool/entity with changed arguments after rejection"
            elif candidate.get("operation") == "WRITE" and not _same_entity(failure, candidate):
                kind = "target_drift_candidate"
                evidence = "later mutation has no shared public entity reference"
            if kind:
                output.append({
                    "event_id": f"recovery-decision-{len(output):04d}",
                    "level": "recovery_decision",
                    "kind": kind,
                    "source_event_ids": [str(failure.get("event_id") or "")],
                    "member_event_ids": [str(candidate.get("event_id") or "")],
                    "public_evidence": evidence,
                    "risk": "candidate_recovery_error",
                })

        has_preview = any(
            event.get("interaction") == "PREVIEW" and _same_entity(failure, event)
            for event in recovery_entry
        )
        committed = [
            event for event in recovery_entry
            if event.get("operation") == "WRITE" and event.get("interaction") == "COMMITTED"
        ]
        if committed and not has_preview:
            output.append({
                "event_id": f"recovery-decision-{len(output):04d}",
                "level": "recovery_decision",
                "kind": "order_violation_candidate",
                "source_event_ids": [str(failure.get("event_id") or "")],
                "member_event_ids": [str(event.get("event_id") or "") for event in committed],
                "public_evidence": "committed same-entity mutation without a public preview in recovery window",
                "risk": "unresolved_precondition_order",
            })
    return output

FUSION_CATEGORY_PRIORITY = {
    "failure_boundary": 6,
    "recovery_decision": 5,
    "material_disagreement": 4,
    "unresolved_evidence": 3,
    "verified_effect": 2,
    "ordinary_observation": 1,
}

FUSION_CATEGORY_CAPS = {
    "material_disagreement": 4,
    "verified_effect": 3,
    "ordinary_observation": 1,
}

def _fusion_category(item: Mapping[str, Any]) -> str:
    tags = set(item.get("risk_tags") or ())
    if "failure_boundary" in tags or item.get("status") == "FAILURE":
        return "failure_boundary"
    if item.get("level") in {"recovery_window", "recovery_decision"}:
        return "recovery_decision"
    if item.get("level") == "transaction_disagreement" or "material_disagreement" in tags:
        return "material_disagreement"
    if tags & {"missing_readback_candidate", "unknown_evidence", "pending_consent_or_commit"}:
        return "unresolved_evidence"
    if item.get("status") == "SUCCESS" and any(
        event.get("event_class") in {"mutation", "verification"}
        for event in item.get("events") or ()
    ):
        return "verified_effect"
    return "ordinary_observation"

def _event_value(item: Mapping[str, Any], round_index: int) -> tuple[int, int, int]:
    """Rank recovery-relevant evidence first and prefer newer evidence on ties."""
    category = _fusion_category(item)
    return (
        FUSION_CATEGORY_PRIORITY[category],
        round_index,
        len(item.get("member_event_ids") or item.get("events") or ()),
    )

def analyze_public_events(trajectory: Mapping[str, Any], tool_schemas: Any = None) -> dict[str, Any]:
    public = public_projection(trajectory)
    events = public_events(public, tool_schemas)
    groups = _make_groups(events)
    windows = _make_recovery_windows(events, groups)
    recovery_decisions = _make_recovery_decision_events(events)
    ledger = {
        "schema_version": "state_trace_event_ledger_v1",
        "task_key": public["task_key"],
        "domain": public["domain"],
        "event_count": len(events),
        "transaction_group_count": len(groups),
        "recovery_window_count": len(windows),
        "raw_events": [_event_payload(event) for event in events],
        "transaction_groups": groups,
        "recovery_windows": windows,
        "recovery_decision_events": recovery_decisions,
        "hierarchy": {
            "atomic": len(events),
            "transaction": len(groups),
            "recovery_window": len(windows),
            "recovery_decision": len(recovery_decisions),
        },
        "status_counts": dict(Counter(group["status"] for group in groups)),
        "public_only": True,
        "outcome_used": False,
    }
    assert_public_payload(ledger)
    return ledger

def compare_public_trajectories(
    previous: Mapping[str, Any], current: Mapping[str, Any], tool_schemas: Any = None,
) -> dict[str, Any]:
    """Build a symmetric public disagreement packet for two rollout rounds."""
    packet = build_disagreement_packet(previous, current, tool_schemas)
    packet["schema_version"] = "state_trace_event_disagreement_v1"
    assert_public_payload(packet)
    return packet

def _bounded_event_payload(item: Mapping[str, Any], max_chars: int = 12000) -> list[dict[str, Any]]:
    """Return public event shapes, never replayable identifiers or values."""
    events = []
    for source in item.get("events") or ():
        event = dict(source)
        arguments = event.get("arguments") or {}
        events.append({
            "event_id": str(event.get("event_id") or ""),
            "position": int(event.get("position", -1)),
            "event_class": str(event.get("event_class") or "other"),
            "operation": str(event.get("operation") or "UNKNOWN"),
            "argument_field_count": len(arguments),
            "argument_field_kinds": sorted(argument_shape(str(key), value) for key, value in arguments.items()),
            "entity_reference_count": len(event.get("entity_refs") or ()),
            "result_category": str(event.get("result") or "UNKNOWN"),
            "interaction": str(event.get("interaction") or ""),
        })
    while events and len(_canonical(events)) > max_chars:
        events.pop()
    return events

def _redact_disagreement(component: Mapping[str, Any]) -> dict[str, Any]:
    """Keep disagreement structure while removing replayable content."""
    output = {
        "component_id": str(component.get("component_id") or ""),
        "kind": str(component.get("kind") or ""),
    }
    transaction_key = component.get("transaction_key")
    if isinstance(transaction_key, list):
        output["transaction_shape"] = [
            str(transaction_key[0]) if transaction_key else "UNKNOWN",
            str(transaction_key[1]) if len(transaction_key) > 1 else "",
            ["entity_ref"] * (len(transaction_key[2]) if len(transaction_key) > 2 and isinstance(transaction_key[2], list) else 0),
        ]
    alternatives = []
    for alternative in component.get("unlabeled_alternatives") or ():
        summary = []
        for item in alternative if isinstance(alternative, list) else ():
            if isinstance(item, Mapping):
                args = item.get("public_arguments") or {}
                summary.append({
                    "schema_role": str(item.get("schema_role") or ""),
                    "public_result_category": str(item.get("public_result_category") or "UNKNOWN"),
                    "argument_field_count": len(args),
                    "argument_field_kinds": sorted(argument_shape(str(key), value) for key, value in args.items()),
                    "entity_reference_count": len(item.get("public_entity_refs") or ()),
                })
            else:
                summary.append("event_shape")
        alternatives.append(summary)
    output["alternative_shapes"] = alternatives
    return output

def fuse_event_ledgers(ledgers: Sequence[Mapping[str, Any]], max_groups: int = 12) -> dict[str, Any]:
    """Fuse prior round ledgers into bounded, source-free public hypotheses."""
    if not ledgers:
        raise ValueError("at least one event ledger is required")
    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    ranked = []
    for round_index, ledger in enumerate(ledgers):
        for group in ledger.get("transaction_groups") or ():
            ranked.append((_event_value(group, round_index), group, round_index))
        for window in ledger.get("recovery_windows") or ():
            ranked.append((_event_value(window, round_index), window, round_index))
        for decision in ledger.get("recovery_decision_events") or ():
            ranked.append((_event_value(decision, round_index), decision, round_index))
        disagreement = ledger.get("cross_round_disagreement") or {}
        for component in disagreement.get("conflict_components") or ():
            item = {
                "level": "transaction_disagreement",
                "status": "MIXED",
                "risk_tags": ["material_disagreement"],
                "entity_refs": [],
                "events": [],
                "disagreement": _redact_disagreement(component),
            }
            ranked.append((_event_value(item, round_index), item, round_index))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    category_counts: Counter[str] = Counter()
    for _, item, round_index in ranked:
        category = _fusion_category(item)
        cap = FUSION_CATEGORY_CAPS.get(category)
        if cap is not None and category_counts[category] >= cap:
            continue
        key = _canonical({
            "events": item.get("events"),
            "risk_tags": item.get("risk_tags"),
            "disagreement": item.get("disagreement"),
        })
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "source_round": round_index,
            "fusion_category": category,
            "level": item.get("level"),
            "status": item.get("status"),
            "risk_tags": list(item.get("risk_tags") or ()),
            "entity_reference_count": len(item.get("entity_refs") or ()),
            "events": _bounded_event_payload(item),
            "source_event_ids": list(item.get("source_event_ids") or ()),
            "member_event_ids": list(item.get("member_event_ids") or ()),
            "public_evidence": item.get("public_evidence"),
            "disagreement": copy.deepcopy(item.get("disagreement")),
        })
        category_counts[category] += 1
        if len(selected) >= max_groups:
            break
    fused = {
        "schema_version": "state_trace_event_fusion_v1",
        "rounds_seen": len(ledgers),
        "high_value_events": selected,
        "category_counts": dict(category_counts),
        "instruction": (
            "Treat these public event summaries as unresolved hypotheses. Re-ground every "
            "object, argument, order, consent decision, and result in the current fresh "
            "environment. Preserve supported effects, resolve recovery risks, and do not "
            "replay a prior action merely because it appears in the summary."
        ),
        "public_only": True,
        "outcome_used": False,
    }
    assert_public_payload(fused)
    return fused

def event_scaling_message(fusion: Mapping[str, Any], round_index: int) -> str:
    assert_public_payload(fusion)
    return (
        f"StateTrace Event-Driven Scaling round {round_index}. This is a public, "
        "outcome-free event ledger from prior independent executions.\n"
        f"{json.dumps(fusion, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}\n"
        "Complete the whole visible task in the current fresh environment. Use the ledger "
        "to locate unresolved recovery decisions, but reacquire facts with legal tools and "
        "ask the user when a choice is theirs. Do not use hidden requirements, evaluator "
        "fields, gold state, or prior candidate outcomes."
    )
