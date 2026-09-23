"""Public dual-obligation arbitration for StateTrace-DOR.

The module consumes completed public trajectories only. It never reads task
outcomes, benchmark requirements, evaluator labels, or environment snapshots.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from trace_ir import extract_trace_ir


FORBIDDEN_FIELDS = {
    "task_completion_pass",
    "state_requirements_met",
    "task_requirements_met",
    "ux_score",
    "gold_action",
    "gold_state",
    "evaluator_label",
    "expected_state",
    "reward",
}

CONSENT_RE = re.compile(
    r"\b(yes|confirm(?:ed)?|go ahead|proceed|process (?:it|the|all)|finali[sz]e|"
    r"make the (?:change|booking)|please (?:cancel|book|change|return|exchange|update|"
    r"process|submit|apply|remove|add)|do (?:it|that)|looks (?:good|right|correct))\b",
    re.IGNORECASE,
)


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _call_name(call: Mapping[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
    return str(call.get("name") or function.get("name") or "")


def _call_args(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
    value = call.get("arguments", function.get("arguments", {}))
    parsed = _json(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _result_status(value: Any) -> str:
    parsed = _json(value)
    if isinstance(parsed, Mapping):
        status = str(parsed.get("status") or "").lower()
        if status == "preview":
            return "PREVIEW"
        if parsed.get("success") is False or parsed.get("ok") is False or parsed.get("error"):
            return "REJECTED"
        if status in {"error", "failed", "rejected", "invalid"}:
            return "REJECTED"
    text = _result_text(parsed).lower()
    if any(token in text for token in ("timeout", "timed out", "deadline exceeded")):
        return "UNKNOWN"
    if re.search(r"\b(error|failed|failure|rejected|invalid|forbidden|not allowed)\b", text):
        return "REJECTED"
    return "ACKNOWLEDGED" if text else "UNKNOWN"


def _entity_refs(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    refs = []
    for key, value in arguments.items():
        lower = str(key).lower()
        if value is None or isinstance(value, (Mapping, list, tuple)):
            continue
        if any(token in lower for token in ("id", "order", "booking", "customer", "flight", "product", "item", "rental")):
            refs.append(f"{key}={value}")
    return tuple(sorted(set(refs)))


def _changed_slots(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    slots = []
    for key in arguments:
        lower = str(key).lower()
        if lower == "confirm" or any(token in lower for token in ("id", "order", "booking", "customer", "flight", "product", "item", "rental")):
            continue
        slots.append(str(key))
    return tuple(sorted(slots))


def _nearest_user(conversation: Sequence[Mapping[str, Any]], turn_index: int) -> tuple[int | None, str]:
    for index in range(turn_index - 1, -1, -1):
        message = conversation[index]
        if message.get("role") == "user":
            return index, str(message.get("content") or "")
    return None, ""


def _tool_results(conversation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in conversation:
        if message.get("role") not in {"tool", "function"}:
            continue
        key = str(message.get("tool_call_id") or message.get("name") or "")
        if key:
            results[key] = message.get("content", message.get("result"))
    return results


def public_events(trajectory: Mapping[str, Any], tool_schemas: Any = None) -> list[dict[str, Any]]:
    """Return public events enriched with consent and preview semantics."""
    conversation = list(trajectory.get("conversation") or ())
    trace = extract_trace_ir(trajectory, tool_schemas)
    results = _tool_results(conversation)
    raw_calls: list[tuple[int, Mapping[str, Any]]] = []
    for turn_index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if isinstance(call, Mapping):
                raw_calls.append((turn_index, call))
    if len(raw_calls) != len(trace.get("events") or ()):
        raise ValueError("TraceIR/public-call count mismatch")

    output = []
    for event, (turn_index, call) in zip(trace["events"], raw_calls):
        arguments = _call_args(call)
        call_id = str(call.get("id") or call.get("tool_call_id") or "")
        result = call.get("result")
        if result is None:
            result = results.get(call_id, results.get(_call_name(call)))
        user_index, user_text = _nearest_user(conversation, turn_index)
        explicit_consent = bool(CONSENT_RE.search(user_text))
        semantic = _result_status(result)
        confirm = arguments.get("confirm") if isinstance(arguments.get("confirm"), bool) else None
        if semantic == "PREVIEW":
            interaction = "PREVIEW"
        elif event.get("operation") == "WRITE" and semantic == "ACKNOWLEDGED":
            interaction = "COMMITTED"
        else:
            interaction = "NONE"
        output.append({
            **event,
            "arguments": arguments,
            "entity_refs": list(_entity_refs(arguments)),
            "changed_slots": list(_changed_slots(arguments)),
            "semantic_result": semantic,
            "interaction_status": interaction,
            "confirm_argument": confirm,
            "nearest_user_index": user_index,
            "explicit_user_consent": explicit_consent,
        })
    return output


def compile_request_obligations(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile candidate-independent public interaction obligations."""
    obligations = []
    for index, message in enumerate(trajectory.get("conversation") or ()):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        if CONSENT_RE.search(text):
            obligations.append({
                "obligation_id": f"consent-user-{index:04d}",
                "type": "CONSENT",
                "state": "supported",
                "public_message_index": index,
                "public_span": text,
            })
    return obligations


def _same_target(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_refs = set(map(str, left.get("entity_refs") or ()))
    right_refs = set(map(str, right.get("entity_refs") or ()))
    return bool(left_refs and right_refs and left_refs & right_refs)


def _same_effect(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return left.get("tool") == right.get("tool") and _same_target(left, right)


def _later(events: Sequence[Mapping[str, Any]], event: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    index = int(event.get("call_index", -1))
    return (candidate for candidate in events if int(candidate.get("call_index", -1)) > index)


def _is_successful_write(event: Mapping[str, Any]) -> bool:
    return (
        event.get("operation") == "WRITE"
        and event.get("semantic_result") == "ACKNOWLEDGED"
        and event.get("interaction_status") == "COMMITTED"
    )


def find_defects(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find high-precision public recovery or consent defects."""
    defects: list[dict[str, Any]] = []
    for event in events:
        if event.get("semantic_result") == "REJECTED" and event.get("operation") == "WRITE":
            recovered = any(
                _is_successful_write(candidate)
                and _same_effect(event, candidate)
                and candidate.get("arguments") != event.get("arguments")
                for candidate in _later(events, event)
            )
            if not recovered:
                defects.append({
                    "defect_id": f"recovery-{event['event_id']}",
                    "type": "UNRESOLVED_REJECTION",
                    "source_event_id": event["event_id"],
                    "tool": event.get("tool"),
                    "entity_refs": list(event.get("entity_refs") or ()),
                    "changed_slots": list(event.get("changed_slots") or ()),
                    "rejected_arguments": event.get("arguments") or {},
                })
        if event.get("interaction_status") == "COMMITTED" and event.get("confirm_argument") is True:
            matching_previews = [
                candidate for candidate in events
                if int(candidate.get("call_index", -1)) < int(event.get("call_index", -1))
                and candidate.get("interaction_status") == "PREVIEW"
                and _same_effect(candidate, event)
            ]
            # A confirm argument is also an API commit flag. Without an earlier
            # public preview, whether extra conversational consent was required
            # is unknown and must not be labeled as a violation.
            if matching_previews and not event.get("explicit_user_consent"):
                defects.append({
                    "defect_id": f"consent-{event['event_id']}",
                    "type": "UNCONSENTED_COMMIT",
                    "source_event_id": event["event_id"],
                    "tool": event.get("tool"),
                    "entity_refs": list(event.get("entity_refs") or ()),
                    "changed_slots": list(event.get("changed_slots") or ()),
                })
        if event.get("interaction_status") == "PREVIEW" and event.get("explicit_user_consent"):
            committed = any(
                _is_successful_write(candidate)
                and _same_effect(event, candidate)
                and candidate.get("confirm_argument") is True
                for candidate in _later(events, event)
            )
            if not committed:
                defects.append({
                    "defect_id": f"pending-{event['event_id']}",
                    "type": "CONSENTED_PREVIEW_NOT_COMMITTED",
                    "source_event_id": event["event_id"],
                    "tool": event.get("tool"),
                    "entity_refs": list(event.get("entity_refs") or ()),
                    "changed_slots": list(event.get("changed_slots") or ()),
                })

    for previous, current in zip(events, events[1:]):
        if (
            previous.get("semantic_result") == "REJECTED"
            and previous.get("tool") == current.get("tool")
            and previous.get("arguments") == current.get("arguments")
        ):
            defects.append({
                "defect_id": f"repeat-{current['event_id']}",
                "type": "EXACT_FAILED_REPEAT",
                "source_event_id": previous["event_id"],
                "repeat_event_id": current["event_id"],
                "tool": previous.get("tool"),
                "entity_refs": list(previous.get("entity_refs") or ()),
                "changed_slots": list(previous.get("changed_slots") or ()),
                "rejected_arguments": previous.get("arguments") or {},
            })
    return defects


def _effect_keys(events: Sequence[Mapping[str, Any]]) -> set[tuple[str, tuple[str, ...]]]:
    return {
        (str(event.get("tool")), tuple(sorted(map(str, event.get("entity_refs") or ()))))
        for event in events
        if _is_successful_write(event) and event.get("entity_refs")
    }


def _certificate(defect: Mapping[str, Any], challenger_events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        event for event in challenger_events
        if event.get("tool") == defect.get("tool")
        and set(map(str, event.get("entity_refs") or ())) & set(map(str, defect.get("entity_refs") or ()))
    ]
    resolution = None
    if defect["type"] in {"UNRESOLVED_REJECTION", "EXACT_FAILED_REPEAT"}:
        resolution = next((
            event for event in candidates
            if _is_successful_write(event)
            and event.get("arguments") != defect.get("rejected_arguments")
        ), None)
    elif defect["type"] == "CONSENTED_PREVIEW_NOT_COMMITTED":
        resolution = next((
            event for event in candidates
            if _is_successful_write(event)
            and event.get("confirm_argument") is True
            and event.get("explicit_user_consent")
        ), None)
    elif defect["type"] == "UNCONSENTED_COMMIT":
        resolution = next((
            event for event in candidates
            if _is_successful_write(event)
            and event.get("confirm_argument") is True
            and event.get("explicit_user_consent")
        ), None)
        if resolution is None:
            preview = next((
                event for event in candidates
                if event.get("interaction_status") == "PREVIEW"
                and event.get("confirm_argument") is not True
            ), None)
            if preview is not None:
                later_unconsented_commit = any(
                    _is_successful_write(event)
                    and _same_effect(preview, event)
                    and event.get("confirm_argument") is True
                    and not event.get("explicit_user_consent")
                    and int(event.get("call_index", -1)) > int(preview.get("call_index", -1))
                    for event in candidates
                )
                if not later_unconsented_commit:
                    resolution = preview
    if resolution is None:
        return None
    return {
        "defect_id": defect["defect_id"],
        "defect_type": defect["type"],
        "anchor_source_event_id": defect["source_event_id"],
        "challenger_resolution_event_id": resolution["event_id"],
        "tool": defect.get("tool"),
        "entity_refs": sorted(set(defect.get("entity_refs") or ()) & set(resolution.get("entity_refs") or ())),
        "anchor_changed_slots": list(defect.get("changed_slots") or ()),
        "challenger_changed_slots": list(resolution.get("changed_slots") or ()),
        "same_entity": True,
        "public_only": True,
    }


def compare_candidate(
    anchor: Mapping[str, Any], challenger: Mapping[str, Any], tool_schemas: Any = None,
) -> dict[str, Any]:
    """Return a non-compensatory public dominance decision."""
    anchor_events = public_events(anchor, tool_schemas)
    challenger_events = public_events(challenger, tool_schemas)
    anchor_defects = find_defects(anchor_events)
    challenger_defects = find_defects(challenger_events)
    certificates = [
        certificate for defect in anchor_defects
        if (certificate := _certificate(defect, challenger_events)) is not None
    ]

    anchor_effects = _effect_keys(anchor_events)
    challenger_effects = _effect_keys(challenger_events)
    resolved_anchor_effects = {
        (str(certificate.get("tool")), tuple(sorted(map(str, certificate.get("entity_refs") or ()))))
        for certificate in certificates
        if certificate.get("defect_type") != "UNCONSENTED_COMMIT"
    }
    lost_effects = sorted(anchor_effects - challenger_effects - resolved_anchor_effects)
    anchor_defect_types = {str(defect["type"]) for defect in anchor_defects}
    new_defects = [
        defect for defect in challenger_defects
        if str(defect["type"]) not in anchor_defect_types
    ]
    dominates = bool(certificates) and not lost_effects and not new_defects
    return {
        "dominates": dominates,
        "anchor_defects": anchor_defects,
        "challenger_defects": challenger_defects,
        "certificates": certificates,
        "lost_supported_effects": [list(item) for item in lost_effects],
        "new_defects": new_defects,
        "anchor_request_obligations": compile_request_obligations(anchor),
        "challenger_request_obligations": compile_request_obligations(challenger),
        "public_only": True,
    }


def arbitrate(
    anchor: Mapping[str, Any], challengers: Sequence[Mapping[str, Any]], tool_schemas: Any = None,
) -> tuple[int, dict[str, Any]]:
    """Select anchor index 0 unless exactly one challenger strictly dominates."""
    comparisons = [compare_candidate(anchor, challenger, tool_schemas) for challenger in challengers]
    dominant = [index + 1 for index, comparison in enumerate(comparisons) if comparison["dominates"]]
    selected = dominant[0] if len(dominant) == 1 else 0
    return selected, {
        "selected_index": selected,
        "dominant_indices": dominant,
        "abstention_reason": None if selected else ("no_dominant_challenger" if not dominant else "multiple_dominant_challengers"),
        "comparisons": comparisons,
        "public_only": True,
        "online_outcome_used": False,
    }


def assert_public_payload(value: Any) -> None:
    """Reject accidental outcome/evaluator fields in a DOR payload."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden DOR field: {key}")
            assert_public_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_public_payload(child)
