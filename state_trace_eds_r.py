"""Extracted from the experimental state_trace_eds_r.py; algorithm bodies unchanged."""

from __future__ import annotations

from typing import Any, Mapping

HR_RISK_TYPES = {
    "repeated_failed_mutation": "repeated_failed_mutation",
    "parameter_change_candidate": "parameter_change_without_regrounding",
    "target_drift_candidate": "target_drift_without_reacquisition",
    "order_violation_candidate": "order_violation",
    "premature_stop_candidate": "premature_stop",
}

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
