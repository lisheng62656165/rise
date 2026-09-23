"""Public-trajectory semantic selector for the OccuBench EDS-ECA adapter."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Mapping, Sequence


ISSUE_TYPES = (
    "NONE",
    "FAILURE_RECOVERY",
    "TARGET_IDENTITY",
    "PARAMETER_GROUNDING",
    "ACTION_ORDER",
    "VERIFICATION",
    "TASK_COVERAGE",
    "STOPPING",
    "OTHER",
)

SELECTOR_INSTRUCTION = """You compare two completed agent trajectories for the same visible task.
Use only the visible task instruction, public agent actions, environment observations, and responses.
Select the trajectory that more completely satisfies the visible request while preserving correct entity
identity, grounded parameters and calculations, valid action order, and consistency with tool results.
An environment word such as 'rejected' can be a correct task outcome; interpret it in context. More
actions, longer prose, completion language, or fewer negative words are not evidence of correctness.
Penalize unsupported claims, omitted task obligations, repeated failed actions without new evidence,
wrong-target actions, contradictions, and stopping without required public evidence. Never use a hidden
verification plan, evaluator result, reward, or candidate source identity. Cite event IDs only from the
candidate to which each credit belongs, and call select_and_credit_events exactly once.
The supporting_event_ids and preserve_event_ids must contain only event IDs from the selected candidate.
The avoid_event_ids must contain only event IDs from the rejected candidate. Include at least one
supporting event ID; preserve_event_ids and avoid_event_ids may be empty when no safe credit applies."""


def selector_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "select_and_credit_events",
            "description": "Select the stronger public trajectory and provide event credit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_index": {"type": "integer", "enum": [0, 1]},
                    "reason": {"type": "string"},
                    "supporting_event_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "preserve_event_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "avoid_event_ids": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "unresolved_issue_type": {
                        "type": "string", "enum": list(ISSUE_TYPES),
                    },
                    "unresolved_issue": {"type": "string"},
                },
                "required": [
                    "candidate_index",
                    "reason",
                    "supporting_event_ids",
                    "preserve_event_ids",
                    "avoid_event_ids",
                    "unresolved_issue_type",
                    "unresolved_issue",
                ],
            },
        },
    }


def _bounded(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 16] + "...[truncated]"


def trajectory_events(trajectory: str, namespace: str) -> list[dict[str, Any]]:
    """Convert public OccuBench text into candidate-scoped semantic events."""
    events: list[dict[str, Any]] = []
    prefixes = (
        ("[Agent Action]", "ACTION", 1800),
        ("[Environment Observation]", "OBSERVATION", 3200),
        ("[Agent Response]", "RESPONSE", 2200),
    )
    current: list[str] = []
    current_kind: str | None = None
    current_limit = 0

    def flush() -> None:
        if current_kind is None:
            return
        event_id = f"{namespace}-e{len(events):03d}"
        events.append({
            "event_id": event_id,
            "event_type": current_kind,
            "content": _bounded("\n".join(current), current_limit),
        })

    for raw_line in str(trajectory or "").splitlines():
        stripped = raw_line.strip()
        matched = next((item for item in prefixes if stripped.startswith(item[0])), None)
        if matched:
            flush()
            prefix, current_kind, current_limit = matched
            current = [stripped[len(prefix):].lstrip(" :")]
        elif current_kind is not None and stripped:
            current.append(stripped)
    flush()
    return events


def _risk_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = []
    verification = []
    for event in events:
        text = str(event.get("content") or "").lower()
        if event.get("event_type") == "OBSERVATION" and any(
            token in text for token in ("error", "failed", "timeout", "denied", "invalid", "not found")
        ):
            failed.append(str(event["event_id"]))
        if any(token in text for token in ("verify", "check", "status", "read back", "confirm")):
            verification.append(str(event["event_id"]))
    return {
        "public_failure_candidate_ids": failed[-6:],
        "public_verification_candidate_ids": verification[-6:],
        "event_count": len(events),
    }


def build_selector_packet(
    task: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    proposal: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], list[int], dict[int, set[str]]]:
    """Build a source-blind packet with collision-free event provenance."""
    candidates = [incumbent, proposal]
    order = [0, 1]
    random.Random(seed).shuffle(order)
    shown_candidates = []
    allowed: dict[int, set[str]] = {}
    for shown_index, original_index in enumerate(order):
        namespace = f"candidate-{shown_index}"
        events = trajectory_events(candidates[original_index].get("trajectory", ""), namespace)
        allowed[shown_index] = {str(event["event_id"]) for event in events}
        shown_candidates.append({
            "candidate_index": shown_index,
            "events": events,
            "public_recovery_diagnostics": _risk_summary(events),
        })
    packet = {
        "task": {
            "scenario": _bounded(task.get("task_scenario_name"), 300),
            "visible_instruction": _bounded(task.get("agent_instruction"), 8000),
        },
        "candidates": shown_candidates,
        "public_only": True,
        "candidate_identity_removed": True,
        "candidate_winner_included": False,
    }
    return packet, order, allowed


def _tool_calls(response: Any) -> list[Any]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    return list(getattr(message, "tool_calls", None) or [])


def parse_selector_response(response: Any) -> dict[str, Any] | None:
    for call in _tool_calls(response):
        function = getattr(call, "function", None)
        if getattr(function, "name", None) != "select_and_credit_events":
            continue
        arguments = getattr(function, "arguments", None)
        try:
            value = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, ValueError):
            return None
        if not isinstance(value, Mapping):
            return None
        index = value.get("candidate_index")
        issue_type = value.get("unresolved_issue_type")
        fields = ("supporting_event_ids", "preserve_event_ids", "avoid_event_ids")
        if index not in (0, 1) or issue_type not in ISSUE_TYPES:
            return None
        if not str(value.get("reason") or "").strip():
            return None
        if any(
            not isinstance(value.get(field), list)
            or not all(isinstance(item, str) and item for item in value[field])
            for field in fields
        ):
            return None
        if not value["supporting_event_ids"]:
            return None
        if issue_type != "NONE" and not str(value.get("unresolved_issue") or "").strip():
            return None
        return dict(value)
    return None


def resolve_selector_response(
    response: Any,
    order: Sequence[int],
    allowed: Mapping[int, set[str]],
) -> tuple[int, dict[str, Any]]:
    """Map a validated displayed choice back to incumbent/proposal index."""
    decision = parse_selector_response(response)
    if decision is None:
        return 0, {"fallback": True, "fallback_reason": "invalid_selector_response"}
    shown = int(decision["candidate_index"])
    selected_ids = allowed.get(shown, set())
    rejected_ids = set().union(*(allowed.get(i, set()) for i in allowed if i != shown))
    for field, valid, reason in (
        ("supporting_event_ids", selected_ids, "unsupported_selection_evidence"),
        ("preserve_event_ids", selected_ids, "preserve_credit_wrong_candidate"),
        ("avoid_event_ids", rejected_ids, "avoid_credit_wrong_candidate"),
    ):
        if not set(decision[field]).issubset(valid):
            return 0, {
                "fallback": True,
                "fallback_reason": reason,
                "decision": decision,
            }
    original = int(order[shown])
    return original, {
        "fallback": False,
        "decision": decision,
        "selected_original_index": original,
    }


def credit_guidance(details: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded, value-free guidance for the next fresh rollout."""
    if details.get("fallback"):
        return {"available": False, "issue_type": "NONE", "instruction": ""}
    decision = details.get("decision") or {}
    issue_type = str(decision.get("unresolved_issue_type") or "NONE")
    directives = {
        "FAILURE_RECOVERY": "After a rejected action, reacquire evidence and use a different supported recovery.",
        "TARGET_IDENTITY": "Reacquire and preserve the intended target before acting.",
        "PARAMETER_GROUNDING": "Recompute action parameters from current public evidence.",
        "ACTION_ORDER": "Re-establish public preconditions and execute dependent actions in order.",
        "VERIFICATION": "Verify consequential effects against a fresh public observation.",
        "TASK_COVERAGE": "Retain every visible requested output, constraint, calculation, and state effect.",
        "STOPPING": "Stop only when visible obligations are complete; avoid post-success actions.",
        "OTHER": "Resolve the remaining public inconsistency without replaying prior values.",
        "NONE": "",
    }
    return {
        "available": True,
        "issue_type": issue_type,
        "instruction": directives[issue_type],
        "preserve_event_count": len(decision.get("preserve_event_ids") or ()),
        "avoid_event_count": len(decision.get("avoid_event_ids") or ()),
    }
