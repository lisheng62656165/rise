"""Deterministic public evidence extraction for StateTrace-RE.

This module deliberately does not inspect benchmark scores, task definitions,
or hidden environment state. It only consumes the visible conversation and
optional public tool schemas.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


READ_HINTS = ("get", "read", "fetch", "search", "list", "lookup", "query", "view", "check", "validate", "calculate")
WRITE_HINTS = (
    "add", "apply", "book", "cancel", "change", "create", "delete", "exchange",
    "file", "modify", "process", "redeem", "refund", "remove", "reserve", "set",
    "submit", "update",
)
TIMEOUT_HINTS = ("timeout", "timed out", "time out", "deadline exceeded", "connection reset")
REJECTION_HINTS = ("invalid", "rejected", "forbidden", "not allowed", "cannot", "failed", "error")


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _call_name(call: Mapping[str, Any]) -> str:
    return str(call.get("name") or call.get("function", {}).get("name") or "")


def _call_args(call: Mapping[str, Any]) -> Any:
    args = call.get("arguments")
    if args is None and isinstance(call.get("function"), Mapping):
        args = call["function"].get("arguments")
    return _json(args or {})


def _call_id(call: Mapping[str, Any]) -> str:
    return str(call.get("id") or call.get("tool_call_id") or "")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _result_failed(result: Any) -> bool:
    if isinstance(result, Mapping):
        if result.get("success") is False or result.get("ok") is False:
            return True
        return bool(result.get("error") or result.get("exception"))
    text = _text(result).lower()
    if not text or any(hint in text for hint in TIMEOUT_HINTS):
        return False
    return bool(re.search(r"\b(error|failed|failure|rejected|invalid|forbidden)\b", text))


def _result_timeout(result: Any) -> bool:
    text = _text(result).lower()
    return any(hint in text for hint in TIMEOUT_HINTS)


def _operation(name: str, args: Any, schema: Mapping[str, Any] | None) -> str:
    lower = name.lower()
    if lower in {"stop", "terminate", "finish", "submit"}:
        return "STOP"
    if any(lower.startswith(hint) or f"_{hint}" in lower for hint in READ_HINTS):
        return "READ"
    if any(lower.startswith(hint) or f"_{hint}" in lower for hint in WRITE_HINTS):
        return "WRITE"
    if schema:
        description = _text(schema).lower()
        if any(token in description for token in ("create", "update", "delete", "modify", "write", "apply", "redeem", "file a", "book a", "cancel")):
            return "WRITE"
        if any(token in description for token in ("read", "retrieve", "look up", "search", "list", "check", "validate")):
            return "READ"
    return "UNKNOWN"


def _entity_refs(args: Any) -> list[str]:
    if not isinstance(args, Mapping):
        return []
    refs: list[str] = []
    for key, value in args.items():
        key_lower = str(key).lower()
        if value is None or isinstance(value, (Mapping, list, tuple)):
            continue
        if any(token in key_lower for token in ("id", "order", "booking", "customer", "flight", "product", "item")):
            refs.append(f"{key}={value}")
    return sorted(set(refs))


def _tool_results(conversation: list[Mapping[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for message in conversation:
        if message.get("role") not in {"tool", "function"}:
            continue
        key = str(message.get("tool_call_id") or message.get("name") or "")
        if key:
            results[key] = message.get("content", message.get("result"))
    return results


def extract_trace_ir(
    trajectory: Mapping[str, Any],
    tool_schemas: Mapping[str, Mapping[str, Any]] | list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return public, deterministic TraceIR for a trajectory row."""
    conversation = list(trajectory.get("conversation") or [])
    tool_results = _tool_results(conversation)
    schemas: dict[str, Mapping[str, Any]] = {}
    if isinstance(tool_schemas, Mapping):
        schemas = dict(tool_schemas)
    elif isinstance(tool_schemas, list):
        schemas = {_call_name(schema): schema for schema in tool_schemas}

    calls: list[dict[str, Any]] = []
    for turn_index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        for call_index, raw_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(raw_call, Mapping):
                continue
            name = _call_name(raw_call)
            args = _call_args(raw_call)
            call_id = _call_id(raw_call)
            result = raw_call.get("result")
            if result is None:
                result = tool_results.get(call_id)
            if result is None:
                result = tool_results.get(name)
            calls.append({
                "turn_index": turn_index,
                "call_index": call_index,
                "name": name,
                "arguments": args,
                "result": result,
                "entity_refs": _entity_refs(args),
                "schema": schemas.get(name),
            })

    events: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        result = call["result"]
        if result is None or _result_timeout(result):
            status = "UNKNOWN"
            evidence = "unknown"
        elif _result_failed(result):
            status = "DETERMINISTIC_REJECTION"
            evidence = "contradicted"
        else:
            status = "ACKNOWLEDGED"
            evidence = "unknown"
        operation = _operation(call["name"], call["arguments"], call["schema"])
        if operation == "READ" and result is not None and not _result_failed(result) and not _result_timeout(result):
            status = "READBACK_CONFIRMED"
            evidence = "confirmed"
        events.append({
            "event_id": f"call-{index:04d}",
            "turn_index": call["turn_index"],
            "call_index": index,
            "tool": call["name"],
            "operation": operation,
            "entity_refs": call["entity_refs"],
            "arguments": call["arguments"],
            "result_status": status,
            "evidence_status": evidence,
            "readback_event_ids": [],
        })

    for index, event in enumerate(events):
        if event["operation"] != "WRITE" or event["result_status"] != "ACKNOWLEDGED":
            continue
        for later in events[index + 1:]:
            if later["operation"] == "READ" and later["result_status"] == "READBACK_CONFIRMED":
                if not event["entity_refs"] or set(event["entity_refs"]) & set(later["entity_refs"]):
                    event["readback_event_ids"].append(later["event_id"])
                    event["result_status"] = "READBACK_CONFIRMED"
                    event["evidence_status"] = "confirmed"
                    break

    diagnostics: list[dict[str, Any]] = []
    for index in range(1, len(events)):
        prior, current = events[index - 1], events[index]
        same_call = (
            prior["tool"] == current["tool"]
            and prior["arguments"] == current["arguments"]
        )
        if same_call and prior["result_status"] == "DETERMINISTIC_REJECTION":
            diagnostics.append({"type": "EXACT_FAILED_REPEAT", "event_ids": [prior["event_id"], current["event_id"]]})

    if events and events[-1]["result_status"] == "DETERMINISTIC_REJECTION":
        diagnostics.append({"type": "UNVERIFIED_TERMINATION_AFTER_EXPLICIT_FAILURE", "event_ids": [events[-1]["event_id"]]})

    return {
        "task_key": trajectory.get("task_key"),
        "events": events,
        "public_recovery_diagnostics": diagnostics,
        "tool_call_count": len(events),
        "public_only": True,
    }


def public_trajectory(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """Strip scoring/runtime fields while retaining visible conversation."""
    output: dict[str, Any] = {"conversation": []}
    for message in trajectory.get("conversation") or []:
        clean: dict[str, Any] = {"role": message.get("role")}
        if message.get("content") not in (None, ""):
            clean["content"] = message["content"]
        if message.get("tool_calls"):
            clean["tool_calls"] = json.loads(json.dumps(message["tool_calls"], default=str))
        if message.get("tool_call_id"):
            clean["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            clean["name"] = message["name"]
        output["conversation"].append(clean)
    return output


def recovery_summary(trace_ir: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "events": trace_ir.get("events", []),
        "public_recovery_diagnostics": trace_ir.get("public_recovery_diagnostics", []),
    }
