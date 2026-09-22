"""Extracted from the experimental state_trace_sdcr.py; algorithm bodies unchanged."""

from __future__ import annotations

import copy

import json

from collections import defaultdict

from typing import Any, Mapping, Sequence

from state_trace_dor import public_events

from public_evidence import assert_public_payload, public_projection

def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_role": str(event.get("operation") or "UNKNOWN"),
        "tool": str(event.get("tool") or ""),
        "public_entity_refs": sorted(map(str, event.get("entity_refs") or ())),
        "public_arguments": copy.deepcopy(dict(event.get("arguments") or {})),
        "public_result_category": str(event.get("semantic_result") or "UNKNOWN"),
    }

def _transaction_key(event: Mapping[str, Any]) -> str:
    return _canonical([
        str(event.get("operation") or "UNKNOWN"),
        str(event.get("tool") or ""),
        sorted(map(str, event.get("entity_refs") or ())),
    ])

def _grouped_events(events: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[_transaction_key(event)].append(_event_payload(event))
    return dict(grouped)

def _order_signature(events: Sequence[Mapping[str, Any]]) -> list[str]:
    return [_transaction_key(event) for event in events]

def build_disagreement_packet(
    left: Mapping[str, Any], right: Mapping[str, Any], schemas: Any = None,
) -> dict[str, Any]:
    left_public, right_public = public_projection(left), public_projection(right)
    if left_public["task_key"] != right_public["task_key"]:
        raise ValueError("SDCR candidates must describe the same task")
    left_events = public_events(left_public, schemas)
    right_events = public_events(right_public, schemas)
    grouped = [_grouped_events(left_events), _grouped_events(right_events)]

    conflicts = []
    for key in sorted(set(grouped[0]) | set(grouped[1])):
        alternatives = [group.get(key, []) for group in grouped]
        if _canonical(alternatives[0]) == _canonical(alternatives[1]):
            continue
        unique = sorted({_canonical(value) for value in alternatives})
        conflicts.append({
            "component_id": f"transaction-{len(conflicts):04d}",
            "kind": "TRANSACTION_CONTENT",
            "transaction_key": json.loads(key),
            "unlabeled_alternatives": [json.loads(value) for value in unique],
        })

    order_alternatives = [_order_signature(left_events), _order_signature(right_events)]
    if _canonical(order_alternatives[0]) != _canonical(order_alternatives[1]):
        unique_order = sorted({_canonical(value) for value in order_alternatives})
        conflicts.append({
            "component_id": f"transaction-{len(conflicts):04d}",
            "kind": "TRANSACTION_ORDER",
            "unlabeled_alternatives": [json.loads(value) for value in unique_order],
        })

    packet = {
        "schema_version": "state_trace_sdcr_packet_v1",
        "task_key": left_public["task_key"],
        "domain": left_public["domain"],
        "material_disagreement": bool(conflicts),
        "conflict_components": conflicts,
        "public_only": True,
        "candidate_identity_removed": True,
        "candidate_winner_included": False,
    }
    assert_public_payload(packet)
    return packet
