"""Event-credit feedback for iterative Event-Driven Scaling.

Trajectory-level acceptance stays unchanged, but every public comparison also
produces local feedback for the next fresh rollout. Positive credit can only
reference the accepted trajectory and negative credit can only reference the
rejected trajectory. Literal values and results are never replayed.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping, Sequence

from state_trace_effect_consensus import assert_public_payload
from state_trace_event_scaling import analyze_public_events, fuse_event_ledgers
from state_trace_eds_r import _hierarchical_events
from state_trace_eds_r import _frontier_card, build_event_graph


def _redact_recovery_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Keep recovery structure while removing replayable entity identities."""
    all_nodes = list(graph.get("nodes") or ())
    # The full graph is useful for offline audit but is too noisy as a model
    # prompt. Keep only risky nodes (plus a short context suffix when no risk
    # was detected), and retain only edges between retained nodes.
    risky_nodes = [node for node in all_nodes if node.get("risk_tags")]
    selected_nodes = risky_nodes[-8:] if risky_nodes else all_nodes[-4:]
    selected_ids = {str(node.get("event_id") or "") for node in selected_nodes}
    return {
        "schema_version": str(graph.get("schema_version") or "recovery_graph_v1"),
        "nodes": [
            {
                "event_id": str(node.get("event_id") or ""),
                "position": int(node.get("position", -1)),
                "event_class": list(node.get("event_class") or ()),
                "operations": list(node.get("operations") or ()),
                "status": node.get("status"),
                "risk_tags": list(node.get("risk_tags") or ()),
                "member_event_count": len(node.get("member_event_ids") or ()),
            }
            for node in selected_nodes
        ],
        "edges": [
            {
                "source": str(edge.get("source") or ""),
                "target": str(edge.get("target") or ""),
                "relation": str(edge.get("relation") or ""),
            }
            for edge in graph.get("edges") or ()
            if str(edge.get("source") or "") in selected_ids
            and str(edge.get("target") or "") in selected_ids
        ],
        "public_only": True,
        "outcome_used": False,
    }


def _redact_recovery_frontier(cards: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "frontier_event_id": str(card.get("frontier_event_id") or ""),
            "risk_type": str(card.get("risk_type") or "unknown_public_recovery_risk"),
            "preserve_event_count": len(card.get("preserve") or ()),
            "repair_event_count": len(card.get("repair") or ()),
            "verify_event_count": len(card.get("verify") or ()),
            "avoid": list(card.get("avoid") or ())[:3],
            "public_evidence": copy.deepcopy(card.get("public_evidence") or {}),
        }
        for card in cards
    ]


ISSUE_TYPES = (
    "NONE", "FAILURE_RECOVERY", "TARGET_IDENTITY", "PARAMETER_GROUNDING",
    "ACTION_ORDER", "VERIFICATION", "CONSENT", "TASK_COVERAGE", "STOPPING", "OTHER",
)

ISSUE_DIRECTIVES = {
    "NONE": "",
    "FAILURE_RECOVERY": "Do not repeat a publicly rejected action; reacquire the target and preconditions before choosing a different recovery action.",
    "TARGET_IDENTITY": "Reacquire the target identity and bind every mutation to the publicly supported object.",
    "PARAMETER_GROUNDING": "Recompute mutation arguments from current public facts and constraints before acting.",
    "ACTION_ORDER": "Re-establish public preconditions and execute dependent actions in a valid order.",
    "VERIFICATION": "Read back every consequential mutation and reconcile the result before stopping.",
    "CONSENT": "Preserve preview and confirmation semantics; do not commit without the required public consent.",
    "TASK_COVERAGE": "Complete every visible stateful and user-facing obligation without dropping explanation or constraints.",
    "STOPPING": "Stop only after visible obligations and required verification are complete, and avoid post-success mutation.",
    "OTHER": "Avoid the rejected structural event pattern and re-ground the next action from current public evidence.",
}

RECOVERY_KIND_PRIORITY = {
    "repeated_failed_mutation": 0,
    "target_drift_candidate": 1,
    "parameter_change_candidate": 2,
    "order_violation_candidate": 3,
    "premature_stop_candidate": 4,
    "recovery_entry": 5,
}


def select_primary_recovery_decision(
    decisions: Sequence[Mapping[str, Any]],
    raw_events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Choose one stable public intervention: severity first, then earliest."""
    if not decisions:
        return None
    positions = {
        str(event.get("event_id") or ""): int(event.get("position", 10**9))
        for event in raw_events
    }

    def position(decision: Mapping[str, Any]) -> int:
        ids = list(decision.get("source_event_ids") or ()) + list(
            decision.get("member_event_ids") or ()
        )
        return min((positions.get(str(event_id), 10**9) for event_id in ids), default=10**9)

    return min(
        decisions,
        key=lambda decision: (
            RECOVERY_KIND_PRIORITY.get(str(decision.get("kind") or ""), 99),
            position(decision),
            str(decision.get("event_id") or ""),
        ),
    )

SELECT_INSTRUCTION = (
    "Perform a list-wise comparison of all completed public trajectories for the same visible stateful task; "
    "do not score candidates independently, vote by frequency, or prefer the first candidate. Select the "
    "stronger trajectory, then assign "
    "event-level credit for the next independent rollout. Preserve-event IDs must come only from the "
    "selected candidate; avoid-event IDs must come only from the rejected candidate. Credit concrete "
    "public evidence about requested coverage, entity and argument grounding, action order, consent, "
    "tool failures, read-back verification, and stopping. A successful tool receipt alone does not prove "
    "that an event fulfilled the request. Preserve coverage of distinct public entities even when their "
    "actions have the same shape; do not treat one repeated-shaped mutation as satisfying all such actions. "
    "State one narrowly scoped unresolved issue for the next rollout, "
    "or NONE when no public issue remains. Treat every event as a hypothesis for a fresh environment. "
    "Never use hidden requirements, gold state, evaluator labels, or candidate outcomes. Return one "
    "candidate_index and cite public event IDs; optional credit fields may be omitted when unsupported. "
    "Call select_and_credit_events exactly once."
)

# Keep relative trajectory selection deliberately small. Models are much
# more reliable at one list-wise decision than at jointly selecting a winner,
# citing events, assigning credit, and classifying the remaining issue.
LISTWISE_SELECT_INSTRUCTION = (
    "Compare the two completed public trajectories jointly and select the one that best completes the same "
    "visible stateful task. Consider: (1) tangible progress toward every visible request, (2) consistency of "
    "the final public state and user intent, (3) efficient tool use, (4) real loops or repeated actions, "
    "(5) error severity and stability, and (6) StateTrace recovery evidence: correct target and arguments, "
    "valid action order and consent, recovery after public failures, read-back verification, and stopping "
    "without a destructive post-success mutation. Compare candidates directly; do not score them in "
    "isolation, vote by frequency, prefer longer traces, or prefer candidate 0. Use only public messages, "
    "tool calls, tool results, schemas, and event summaries. Never use hidden requirements, gold state, "
    "evaluator labels, or candidate outcomes. Return exactly one choice by calling select_listwise_trajectory."
)


def compact_listwise_retry_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Remove verbose tool-result prose while preserving symmetric public evidence."""
    compact_candidates = []
    for candidate in packet.get("candidates") or ():
        conversation = ((candidate.get("trajectory") or {}).get("conversation") or [])
        user_messages = [
            str(message.get("content") or "")
            for message in conversation
            if message.get("role") == "user" and message.get("content") not in (None, "")
        ]
        final_answer = next((
            str(message.get("content") or "")
            for message in reversed(conversation)
            if message.get("role") == "assistant" and message.get("content") not in (None, "")
        ), "")
        compact_candidates.append({
            "candidate_index": candidate.get("candidate_index"),
            "candidate_role": candidate.get("candidate_role"),
            "visible_user_messages": user_messages,
            "final_assistant_answer": final_answer,
            "public_trace_ir": copy.deepcopy(candidate.get("public_trace_ir") or {}),
        })
    output = {
        "candidates": compact_candidates,
        "public_transaction_disagreement": copy.deepcopy(
            packet.get("public_transaction_disagreement") or []
        ),
        "public_only": True,
        "compact_format_retry": True,
    }
    if "public_receipt_evidence" in packet:
        output["public_receipt_evidence"] = copy.deepcopy(packet["public_receipt_evidence"])
    assert_public_payload(output)
    return output


LATE_REPLACEMENT_BASES = (
    "retain_incumbent",
    "missing_visible_obligation",
    "explicit_failure_recovery",
    "verified_state_restoration",
)


def listwise_selector_tool(
    count: int = 2, *, require_replacement_evidence: bool = False,
) -> dict[str, Any]:
    if count < 2:
        raise ValueError("list-wise selector needs at least two candidates")
    properties: dict[str, Any] = {
        "index": {"type": "integer", "enum": list(range(count))},
        "analysis": {"type": "string"},
    }
    required = ["index", "analysis"]
    if require_replacement_evidence:
        properties.update({
            "replacement_basis": {
                "type": "string", "enum": list(LATE_REPLACEMENT_BASES),
            },
            "proposal_evidence_event_ids": {
                "type": "array", "items": {"type": "string"},
            },
            "incumbent_defect_event_ids": {
                "type": "array", "items": {"type": "string"},
            },
        })
        required.extend([
            "replacement_basis",
            "proposal_evidence_event_ids",
            "incumbent_defect_event_ids",
        ])
    return {
        "name": "select_listwise_trajectory",
        "description": "Select the strongest completed public trajectory by direct comparison.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def parse_listwise_choice(response: Any, count: int = 2) -> dict[str, Any] | None:
    """Parse the paper-style ``index + analysis`` contract across providers."""
    calls = (
        list(response.get("tool_calls") or ())
        if isinstance(response, Mapping)
        else list(getattr(response, "tool_calls", ()) or ())
    )
    candidates: list[Mapping[str, Any]] = []
    explicit_text: list[str] = []
    for call in calls:
        name, raw_arguments = _tool_call_parts(call)
        if name != "select_listwise_trajectory":
            continue
        if isinstance(raw_arguments, str):
            explicit_text.append(raw_arguments)
        arguments = _json_mapping(raw_arguments)
        if arguments is not None:
            candidates.append(arguments)
    content = response.get("content") if isinstance(response, Mapping) else getattr(response, "content", None)
    if isinstance(content, list):
        content = " ".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, Mapping) else str(item)
            for item in content
        )
    if isinstance(content, str):
        explicit_text.append(content)
    content_mapping = _json_mapping(content)
    if content_mapping is not None:
        candidates.append(content_mapping)
    for value in candidates:
        index = _candidate_index(value.get("index", value.get("candidate_index")), count)
        analysis = str(value.get("analysis") or value.get("reason") or "").strip()
        if index is not None and analysis:
            decision = {"candidate_index": index, "reason": analysis}
            if value.get("replacement_basis") in LATE_REPLACEMENT_BASES:
                decision.update({
                    "replacement_basis": value["replacement_basis"],
                    "proposal_evidence_event_ids": _string_list(
                        value.get("proposal_evidence_event_ids")
                    ),
                    "incumbent_defect_event_ids": _string_list(
                        value.get("incumbent_defect_event_ids")
                    ),
                })
            return decision
    # Some OpenAI-compatible providers truncate or prose-wrap otherwise valid
    # tool arguments. Accept only one unambiguous, explicitly labelled index.
    matches: list[int] = []
    for text in explicit_text:
        for pattern in (
            r'"(?:index|candidate_index)"\s*:\s*([0-9]+)',
            r"(?:final\s+choice|selected|choose)\s*[:=]?\s*(?:candidate\s*)?([0-9]+)\b",
        ):
            matches.extend(int(value) for value in re.findall(pattern, text, flags=re.IGNORECASE))
    valid = {value for value in matches if value in range(count)}
    if len(valid) == 1:
        return {
            "candidate_index": valid.pop(),
            "reason": "explicit prose-wrapped list-wise choice",
            "parser": "explicit_index_v1",
        }
    return None


def resolve_listwise_choice(
    response: Any, shown_to_original: Sequence[int],
) -> tuple[int, dict[str, Any]]:
    decision = parse_listwise_choice(response, len(shown_to_original))
    if decision is None:
        return 0, {
            "fallback": True,
            "decision": None,
            "fallback_reason": "invalid_listwise_response",
        }
    shown = int(decision["candidate_index"])
    original = int(shown_to_original[shown])
    mapped = dict(decision)
    mapped.update({
        "selected_original_index": original,
        "rejected_original_indices": [
            int(value) for value in shown_to_original if int(value) != original
        ],
    })
    return original, {"fallback": False, "decision": mapped}


def event_credit_tool(count: int = 2) -> dict[str, Any]:
    if count < 2:
        raise ValueError("event-credit selector needs at least two candidates")
    return {
        "name": "select_and_credit_events",
        "description": "Select one public trajectory and assign local event credit for the next rollout.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_index": {"type": "integer", "enum": list(range(count))},
                "reason": {"type": "string"},
                "supporting_event_ids": {"type": "array", "items": {"type": "string"}},
                "preserve_event_ids": {"type": "array", "items": {"type": "string"}},
                "avoid_event_ids": {"type": "array", "items": {"type": "string"}},
                "unresolved_issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
                "unresolved_issue": {"type": "string"},
            },
            "required": [
                "candidate_index", "reason", "supporting_event_ids",
            ],
        },
    }


def _tool_call_parts(call: Any) -> tuple[Any, Any]:
    """Read both normalized and OpenAI-style nested tool calls."""
    if isinstance(call, Mapping):
        function = call.get("function")
        if isinstance(function, Mapping):
            return function.get("name"), function.get("arguments", {})
        return call.get("name"), call.get("arguments", {})
    function = getattr(call, "function", None)
    if function is not None:
        return getattr(function, "name", None), getattr(function, "arguments", {})
    return getattr(call, "name", None), getattr(call, "arguments", {})


def _json_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _candidate_index(value: Any, count: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str) and value.strip().isdigit():
        candidate = int(value.strip())
    else:
        return None
    return candidate if candidate in range(count) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def parse_event_credit(response: Any, count: int = 2) -> dict[str, Any] | None:
    calls = (
        list(response.get("tool_calls") or ())
        if isinstance(response, Mapping)
        else list(getattr(response, "tool_calls", ()) or ())
    )
    argument_candidates: list[Mapping[str, Any]] = []
    for call in calls:
        name, raw_arguments = _tool_call_parts(call)
        if name != "select_and_credit_events":
            continue
        arguments = _json_mapping(raw_arguments)
        if arguments is not None:
            argument_candidates.append(arguments)

    # Some providers return the requested JSON in assistant content even when
    # a tool schema was supplied. Accept only a complete JSON object, never a
    # winner inferred from free-form prose.
    content = response.get("content") if isinstance(response, Mapping) else getattr(response, "content", None)
    content_mapping = _json_mapping(content)
    if content_mapping is not None:
        if content_mapping.get("name") == "select_and_credit_events":
            nested = _json_mapping(content_mapping.get("arguments"))
            if nested is not None:
                argument_candidates.append(nested)
        else:
            argument_candidates.append(content_mapping)

    for arguments in argument_candidates:
        candidate = _candidate_index(arguments.get("candidate_index"), count)
        reason = str(arguments.get("reason") or "").strip()
        preserve = _string_list(arguments.get("preserve_event_ids"))
        supporting = _string_list(arguments.get("supporting_event_ids")) or list(preserve)
        avoid = _string_list(arguments.get("avoid_event_ids"))
        if candidate is None or not reason or not supporting:
            continue
        raw_issue_type = str(arguments.get("unresolved_issue_type") or "NONE").strip().upper()
        issue = str(arguments.get("unresolved_issue") or "").strip()
        issue_type = raw_issue_type if raw_issue_type in ISSUE_TYPES else ("OTHER" if issue else "NONE")
        if issue_type != "NONE" and not issue:
            issue_type = "NONE"
        issue = str(arguments.get("unresolved_issue") or "").strip()
        return {
            "candidate_index": int(candidate),
            "reason": reason,
            "supporting_event_ids": supporting,
            "preserve_event_ids": preserve,
            "avoid_event_ids": avoid,
            "unresolved_issue_type": str(issue_type),
            "unresolved_issue": issue,
        }
    return None


def resolve_event_credit(
    response: Any,
    shown_to_original: Sequence[int],
    allowed_event_ids: Mapping[int, set[str]],
) -> tuple[int, dict[str, Any]]:
    """Validate candidate and event provenance; invalid feedback keeps the incumbent."""
    decision = parse_event_credit(response, len(shown_to_original))
    if decision is None:
        return 0, {"fallback": True, "decision": None, "fallback_reason": "invalid_event_credit_response"}
    shown = decision["candidate_index"]
    selected_ids = allowed_event_ids.get(shown, set())
    rejected_ids = set().union(*(
        allowed_event_ids.get(index, set())
        for index in range(len(shown_to_original)) if index != shown
    ))
    valid_support = [value for value in decision["supporting_event_ids"] if value in selected_ids]
    if not valid_support:
        return 0, {
            "fallback": True,
            "decision": decision,
            "fallback_reason": "unsupported_selection_evidence",
        }
    dropped = {
        "supporting_event_ids": [value for value in decision["supporting_event_ids"] if value not in selected_ids],
        "preserve_event_ids": [value for value in decision["preserve_event_ids"] if value not in selected_ids],
        "avoid_event_ids": [value for value in decision["avoid_event_ids"] if value not in rejected_ids],
    }
    decision["supporting_event_ids"] = valid_support
    decision["preserve_event_ids"] = [value for value in decision["preserve_event_ids"] if value in selected_ids]
    decision["avoid_event_ids"] = [value for value in decision["avoid_event_ids"] if value in rejected_ids]
    original = int(shown_to_original[shown])
    mapped = copy.deepcopy(decision)
    mapped["selected_original_index"] = original
    mapped["rejected_original_indices"] = [
        int(value) for value in shown_to_original if value != original
    ]
    return original, {
        "fallback": False,
        "decision": mapped,
        "credit_sanitized": any(dropped.values()),
        "dropped_event_ids": dropped,
    }


def _field_name(reference: str) -> str:
    return reference.partition("=")[0].strip()


def _structural_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_event_id": str(event.get("event_id") or ""),
        "operation": str(event.get("operation") or "UNKNOWN"),
        "event_class": str(event.get("event_class") or "other"),
        "tool": str(event.get("tool") or ""),
        "entity_fields": sorted({
            _field_name(str(value)) for value in event.get("entity_refs") or ()
            if _field_name(str(value))
        }),
        "argument_keys": sorted(str(key) for key in (event.get("arguments") or {})),
        "result_category": str(event.get("result") or "UNKNOWN"),
        "interaction": str(event.get("interaction") or ""),
    }


def _events_by_id(trajectory: Mapping[str, Any], schemas: Any) -> dict[str, dict[str, Any]]:
    ledger = analyze_public_events(trajectory, schemas)
    return {
        str(event.get("event_id") or ""): dict(event)
        for event in ledger.get("raw_events") or () if event.get("event_id")
    }


def _infer_issue_type(avoid_events: Sequence[Mapping[str, Any]]) -> str:
    """Turn contradictory NONE-plus-negative-credit output into typed feedback."""
    if any(event.get("result_category") in {"REJECTED", "FAILED", "ERROR"} for event in avoid_events):
        return "FAILURE_RECOVERY"
    if any(event.get("interaction") == "PREVIEW" for event in avoid_events):
        return "CONSENT"
    if any(event.get("event_class") == "mutation" for event in avoid_events):
        return "PARAMETER_GROUNDING"
    if any(event.get("event_class") == "verification" for event in avoid_events):
        return "VERIFICATION"
    return "OTHER" if avoid_events else "NONE"


def _event_shape(event: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return a value-free event identity for cross-environment credit."""
    structural = _structural_event(event)
    return (
        structural["operation"], structural["event_class"], structural["tool"],
        tuple(structural["entity_fields"]), tuple(structural["argument_keys"]),
        structural["result_category"], structural["interaction"],
    )


def _risk_ids_and_tags(
    ledger: Mapping[str, Any], *, tagged_only: bool = False,
) -> tuple[set[str], dict[str, set[str]]]:
    ids: set[str] = set()
    tags_by_id: dict[str, set[str]] = {}
    sources = (
        list(ledger.get("transaction_groups") or ())
        + list(ledger.get("recovery_windows") or ())
        + list(ledger.get("recovery_decision_events") or ())
    )
    for item in sources:
        tags = {str(value) for value in item.get("risk_tags") or ()}
        kind = str(item.get("kind") or "")
        if kind:
            tags.add(kind)
        if tagged_only and not tags:
            continue
        source_ids = list(item.get("source_event_ids") or ()) + list(
            item.get("member_event_ids") or ()
        )
        for event_id in source_ids:
            key = str(event_id)
            if key:
                ids.add(key)
                tags_by_id.setdefault(key, set()).update(tags)
    return ids, tags_by_id


def _issue_from_tags(tags: set[str], avoid_events: Sequence[Mapping[str, Any]]) -> str:
    joined = " ".join(sorted(tags)).lower()
    if "target" in joined or "identity" in joined:
        return "TARGET_IDENTITY"
    if "parameter" in joined or "argument" in joined:
        return "PARAMETER_GROUNDING"
    if "order" in joined or "precondition" in joined:
        return "ACTION_ORDER"
    if "consent" in joined or "preview" in joined:
        return "CONSENT"
    if "readback" in joined or "verification" in joined:
        return "VERIFICATION"
    if "premature" in joined or "post_success" in joined or "stopping" in joined:
        return "STOPPING"
    if "recovery" in joined or "failed" in joined or "repeat" in joined:
        return "FAILURE_RECOVERY"
    return _infer_issue_type(avoid_events)


def derive_event_credit_decision(
    candidates: Sequence[Mapping[str, Any]],
    selected_index: int,
    schemas: Any,
    reason: str = "",
    *,
    failure_grounded: bool = False,
    tagged_risk_only: bool = False,
) -> dict[str, Any]:
    """Derive model-independent forward credit after list-wise selection."""
    if selected_index not in range(len(candidates)):
        raise ValueError("selected_index is outside candidate pool")
    ledgers = [analyze_public_events(candidate, schemas) for candidate in candidates]
    events = [list(ledger.get("raw_events") or ()) for ledger in ledgers]
    rejected_indices = [index for index in range(len(candidates)) if index != selected_index]
    rejected_events = [event for index in rejected_indices for event in events[index]]
    selected_events = events[selected_index]
    selected_shapes = {_event_shape(event) for event in selected_events}
    rejected_shapes = {_event_shape(event) for event in rejected_events}

    selected_risk_ids, _ = _risk_ids_and_tags(
        ledgers[selected_index], tagged_only=tagged_risk_only,
    )
    rejected_risk_ids: set[str] = set()
    rejected_tags: dict[str, set[str]] = {}
    for index in rejected_indices:
        risk_ids, tags = _risk_ids_and_tags(ledgers[index], tagged_only=tagged_risk_only)
        rejected_risk_ids.update(risk_ids)
        for event_id, values in tags.items():
            rejected_tags.setdefault(event_id, set()).update(values)

    def successful(event: Mapping[str, Any]) -> bool:
        return str(event.get("result") or "") not in {"REJECTED", "FAILED", "ERROR"}

    preserve_ranked = sorted(
        (
            event for event in selected_events
            if successful(event) and (
                _event_shape(event) not in rejected_shapes
                or event.get("event_class") in {"mutation", "verification"}
            )
        ),
        key=lambda event: (
            event.get("event_class") == "verification",
            event.get("event_class") == "mutation",
            str(event.get("event_id") or "") not in selected_risk_ids,
            int(event.get("position", -1)),
        ),
        reverse=True,
    )
    avoid_ranked = sorted(
        (
            event for event in rejected_events
            if (
                not successful(event)
                if failure_grounded else (
                    str(event.get("event_id") or "") in rejected_risk_ids
                    or not successful(event)
                    or _event_shape(event) not in selected_shapes
                )
            )
        ),
        key=lambda event: (
            str(event.get("event_id") or "") in rejected_risk_ids,
            not successful(event),
            event.get("event_class") in {"mutation", "verification"},
            int(event.get("position", -1)),
        ),
        reverse=True,
    )
    # Ordering or semantic grounding can decide a comparison even when both
    # traces have identical value-free event shapes. Keep a bounded suffix so
    # the next round still receives ECA without replaying literal values.
    if not preserve_ranked:
        preserve_ranked = [event for event in selected_events if successful(event)][-3:]
    if not avoid_ranked and not failure_grounded:
        avoid_ranked = [
            event for event in rejected_events
            if event.get("event_class") in {"mutation", "verification"}
        ][-3:]

    preserve_ids = [str(event.get("event_id")) for event in preserve_ranked[:6] if event.get("event_id")]
    avoid_ids = [str(event.get("event_id")) for event in avoid_ranked[:6] if event.get("event_id")]
    avoid_structural = [_structural_event(event) for event in avoid_ranked[:6]]
    active_tags = {
        tag for event_id in avoid_ids for tag in rejected_tags.get(event_id, set())
    }
    issue_type = _issue_from_tags(active_tags, avoid_structural)
    return {
        "candidate_index": int(selected_index),
        "reason": str(reason or "list-wise public trajectory comparison"),
        "supporting_event_ids": list(preserve_ids),
        "preserve_event_ids": preserve_ids,
        "avoid_event_ids": avoid_ids,
        "unresolved_issue_type": issue_type,
        "unresolved_issue": ISSUE_DIRECTIVES[issue_type],
        "credit_source": (
            "failure_grounded_public_event_credit_v1" if failure_grounded
            else "tagged_risk_public_event_difference_v1" if tagged_risk_only
            else "deterministic_public_event_difference_v1"
        ),
    }


def _credit_transaction_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    """Align public events without carrying environment-specific literal values."""
    structural = _structural_event(event)
    return (
        structural["operation"], structural["event_class"], structural["tool"],
        tuple(structural["entity_fields"]), tuple(structural["argument_keys"]),
        structural["interaction"],
    )


def _credit_content_signature(event: Mapping[str, Any]) -> str:
    """Keep public values internal to alignment; they are never emitted as credit."""
    return json.dumps({
        "entity_refs": sorted(map(str, event.get("entity_refs") or ())),
        "arguments": dict(event.get("arguments") or {}),
        "result": str(event.get("result") or "UNKNOWN"),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def derive_conflict_localized_event_credit_decision(
    candidates: Sequence[Mapping[str, Any]],
    selected_index: int,
    schemas: Any,
    reason: str = "",
) -> dict[str, Any]:
    """Assign credit only where public evidence localizes a difference or risk.

    A trajectory-level loss does not make every rejected-exclusive event bad.
    Negative credit therefore requires an explicit public failure/risk or a
    content conflict within an aligned transaction. Positive credit is limited
    to risk-free verification and the selected side of those same conflicts.
    """
    if selected_index not in range(len(candidates)):
        raise ValueError("selected_index is outside candidate pool")
    ledgers = [analyze_public_events(candidate, schemas) for candidate in candidates]
    events = [list(ledger.get("raw_events") or ()) for ledger in ledgers]
    selected_events = events[selected_index]
    rejected_indices = [index for index in range(len(candidates)) if index != selected_index]
    rejected_events = [event for index in rejected_indices for event in events[index]]
    rejected_shapes = {_event_shape(event) for event in rejected_events}

    selected_by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    rejected_by_key: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for event in selected_events:
        selected_by_key.setdefault(_credit_transaction_key(event), []).append(event)
    for event in rejected_events:
        rejected_by_key.setdefault(_credit_transaction_key(event), []).append(event)

    _, selected_tags = _risk_ids_and_tags(ledgers[selected_index])
    selected_risk_ids = {
        event_id for event_id, tags in selected_tags.items() if tags
    }
    rejected_risk_ids: set[str] = set()
    rejected_tags: dict[str, set[str]] = {}
    for index in rejected_indices:
        _, tags = _risk_ids_and_tags(ledgers[index])
        for event_id, values in tags.items():
            rejected_tags.setdefault(event_id, set()).update(values)
        rejected_risk_ids.update(
            event_id for event_id, values in tags.items() if values
        )

    failed_results = {"REJECTED", "FAILED", "ERROR"}

    def failed(event: Mapping[str, Any]) -> bool:
        return str(event.get("result") or "") in failed_results

    conflict_keys = {
        key for key in set(selected_by_key) & set(rejected_by_key)
        if {
            _credit_content_signature(event) for event in selected_by_key[key]
        } != {
            _credit_content_signature(event) for event in rejected_by_key[key]
        }
    }

    successful_mutations = [
        event for event in selected_events
        if event.get("event_class") == "mutation" and not failed(event)
    ]

    def is_public_readback(event: Mapping[str, Any]) -> bool:
        if str(event.get("operation") or "") != "READ" or failed(event):
            return False
        fields = set(_structural_event(event)["entity_fields"])
        if not fields:
            return False
        position = int(event.get("position", -1))
        return any(
            int(mutation.get("position", -1)) < position
            and fields.intersection(_structural_event(mutation)["entity_fields"])
            for mutation in successful_mutations
        )

    def is_grounded_observation(event: Mapping[str, Any]) -> bool:
        return (
            str(event.get("operation") or "") == "READ"
            and not failed(event)
            and bool(_structural_event(event)["entity_fields"])
        )

    preserve_ranked = sorted(
        (
            event for event in selected_events
            if not failed(event)
            and str(event.get("event_id") or "") not in selected_risk_ids
            and (
                _event_shape(event) not in rejected_shapes
                or _credit_transaction_key(event) in conflict_keys
                or event.get("event_class") == "mutation"
                or event.get("event_class") == "verification"
                or is_public_readback(event)
                or is_grounded_observation(event)
            )
        ),
        key=lambda event: (
            _credit_transaction_key(event) in conflict_keys,
            is_public_readback(event),
            is_grounded_observation(event),
            event.get("event_class") == "verification",
            event.get("event_class") == "mutation",
            int(event.get("position", -1)),
        ),
        reverse=True,
    )
    avoid_ranked = sorted(
        (
            event for event in rejected_events
            if failed(event)
            or str(event.get("event_id") or "") in rejected_risk_ids
            or _credit_transaction_key(event) in conflict_keys
        ),
        key=lambda event: (
            str(event.get("event_id") or "") in rejected_risk_ids,
            failed(event),
            _credit_transaction_key(event) in conflict_keys,
            int(event.get("position", -1)),
        ),
        reverse=True,
    )

    preserve_ids = [str(event.get("event_id")) for event in preserve_ranked[:6] if event.get("event_id")]
    avoid_ids = [str(event.get("event_id")) for event in avoid_ranked[:6] if event.get("event_id")]
    avoid_structural = [_structural_event(event) for event in avoid_ranked[:6]]
    active_tags = {
        tag for event_id in avoid_ids for tag in rejected_tags.get(event_id, set())
    }
    issue_type = _issue_from_tags(active_tags, avoid_structural)
    return {
        "candidate_index": int(selected_index),
        "reason": str(reason or "list-wise public trajectory comparison"),
        "supporting_event_ids": list(preserve_ids),
        "preserve_event_ids": preserve_ids,
        "avoid_event_ids": avoid_ids,
        "unresolved_issue_type": issue_type,
        "unresolved_issue": ISSUE_DIRECTIVES[issue_type],
        "credit_source": "conflict_localized_public_event_credit_v1",
        "conflict_component_count": len(conflict_keys),
    }


def build_event_credit_state(
    candidates: Sequence[Mapping[str, Any]],
    selected_index: int,
    selector_details: Mapping[str, Any],
    schemas: Any,
    prior_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    decision = selector_details.get("decision") or {}
    if selector_details.get("fallback") or not decision:
        state = {
            "schema_version": "state_trace_event_credit_v1", "available": False,
            "preserve_events": [], "avoid_events": [], "unresolved_issue_type": "NONE",
            "unresolved_issue": "", "public_only": True, "outcome_used": False,
        }
        assert_public_payload(state)
        return state
    selected_events = _events_by_id(candidates[selected_index], schemas)
    rejected_indices = [index for index in range(len(candidates)) if index != selected_index]
    preserve = [
        _structural_event(selected_events[event_id])
        for event_id in decision.get("preserve_event_ids") or () if event_id in selected_events
    ][:6]
    rejected_events = {
        event_id: event
        for index in rejected_indices
        for event_id, event in _events_by_id(candidates[index], schemas).items()
    }
    avoid = [
        _structural_event(rejected_events[event_id])
        for event_id in decision.get("avoid_event_ids") or () if event_id in rejected_events
    ][:6]
    issue_type = str(decision.get("unresolved_issue_type") or "NONE")
    issue_source = "selector"
    if issue_type == "NONE" and avoid:
        issue_type = _infer_issue_type(avoid)
        issue_source = "structural_negative_credit"
    prior_issue = str((prior_state or {}).get("unresolved_issue_type") or "NONE")
    if issue_type == "NONE":
        recurrence = 0
    elif issue_type == prior_issue:
        recurrence = int((prior_state or {}).get("issue_recurrence") or 0) + 1
    else:
        recurrence = 1
    state = {
        "schema_version": "state_trace_event_credit_v1", "available": True,
        "preserve_events": preserve, "avoid_events": avoid,
        "unresolved_issue_type": issue_type,
        "unresolved_issue": ISSUE_DIRECTIVES[issue_type],
        "issue_source": issue_source,
        "issue_recurrence": recurrence,
        "public_only": True, "outcome_used": False,
    }
    assert_public_payload(state)
    return state


def build_eds_ec_frontier(
    incumbent: Mapping[str, Any], schemas: Any,
    prior_disagreement: Mapping[str, Any] | None,
    max_event_groups: int,
    prior_credit_state: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = analyze_public_events(incumbent, schemas)
    if prior_disagreement:
        ledger["cross_round_disagreement"] = copy.deepcopy(dict(prior_disagreement))
        ledger["material_disagreement"] = bool(prior_disagreement.get("material_disagreement"))
    hierarchy = _hierarchical_events(ledger)
    current_rds = [event for event in hierarchy if event.get("level") == "recovery_decision"]
    primary_rds = select_primary_recovery_decision(
        ledger.get("recovery_decision_events") or (),
        ledger.get("raw_events") or (),
    )
    credit = copy.deepcopy(dict(prior_credit_state or {}))
    active = None
    if current_rds:
        event = next(
            (item for item in current_rds
             if str(item.get("event_id") or "") == str((primary_rds or {}).get("event_id") or "")),
            current_rds[0],
        )
        active = {
            "source": "current_public_rds", "issue_type": str(event.get("rds_type") or "public_recovery_risk"),
            "event_id": str(event.get("event_id") or ""),
            "instruction": (
                "Re-ground the affected target, parameters, order, and current result; choose a different "
                "recovery action when public evidence rejects the prior one, then verify it."
            ),
        }
    elif credit.get("available") and credit.get("unresolved_issue_type") != "NONE":
        active = {
            "source": "prior_event_credit", "issue_type": str(credit.get("unresolved_issue_type")),
            "event_id": None, "instruction": str(credit.get("unresolved_issue") or "")[:1200],
        }
    # Keep offline ledger fidelity, but bound the model-facing event summary.
    # Long heterogeneous traces otherwise spend the proposal context on
    # ordinary observations instead of the few recovery-relevant events.
    fusion = fuse_event_ledgers([ledger], max_groups=min(6, max_event_groups))
    fusion.update({
        "schema_version": "state_trace_eds_ec_frontier_v1",
        "event_hierarchy": hierarchy,
        "event_credit_state": credit,
        "event_intervention": {
            "policy": "preserve supported prefix, repair earliest public recovery frontier, verify, then stop",
            "reacquire_current_state": True,
            "replay_prior_calls": False,
        },
        "active_intervention": active,
        "intervention_count": 1 if active else 0,
        "credit_policy": {
            "positive_credit": "accepted-trajectory structure only",
            "negative_credit": "rejected-trajectory warning only",
            "history": "latest validated comparison only",
            "literal_replay": False,
        },
        "task_coverage_policy": [
            "complete every visible user request and constraint",
            "preserve required explanation, confirmation, and interaction semantics",
            "reacquire current objects and values before mutation",
            "verify consequential writes and stop only after the visible task is complete",
        ],
        "instruction": (
            "Execute the entire visible task in the fresh environment. Use the event hierarchy to understand "
            "transactions and recovery episodes. Make the smallest necessary intervention: do not alter "
            "already-supported actions or invent unrelated mutations. If no unresolved credit exists, "
            "preserve the current public task coverage and only re-ground facts. Apply at most the one "
            "active intervention, if present. "
            "Preserve positively credited event structure, avoid negatively credited structure, but never "
            "replay old identifiers, argument values, results, or calls. Re-ground all facts with legal tools."
        ),
        "public_only": True, "outcome_used": False,
    })
    assert_public_payload(fusion)
    return ledger, fusion
