"""Extracted from the experimental state_trace_dor.py; algorithm bodies unchanged."""

from __future__ import annotations

import json

import re

from typing import Any, Mapping, Sequence

from trace_ir import extract_trace_ir

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
