"""Extracted from the experimental state_trace_pool_select.py; algorithm bodies unchanged."""

from __future__ import annotations

import re

from typing import Any, Mapping

from state_trace_eds_ec import resolve_listwise_choice

from trace_ir import extract_trace_ir, public_trajectory, recovery_summary

OAGENTS_BON_LISTWISE_INSTRUCTION = (
    "Compare four independently completed public trajectories for the same visible "
    "task and select exactly one final trajectory. Use only the supplied public "
    "conversation, tool calls, tool results, and tool schemas. Prefer concrete "
    "completion of the visible request, consistency with the final public state, "
    "correct arguments and targets, valid ordering, and fewer unresolved tool "
    "errors or contradictions. Do not use hidden requirements, evaluator labels, "
    "rewards, private state, candidate scores, event labels, recovery annotations, "
    "or trajectory length as a quality signal. Do not vote by frequency and do not "
    "prefer candidate 0. Return exactly one candidate index and a concise analysis "
    "by calling select_listwise_trajectory."
)

def build_packet(candidates: list[Mapping[str, Any]], schemas: Any, order: list[int]) -> dict[str, Any]:
    traces = [extract_trace_ir(row, schemas) for row in candidates]
    return {
        "candidates": [{
            "candidate_index": shown,
            "trajectory": public_trajectory(candidates[original]),
            "public_trace_ir": recovery_summary(traces[original]),
        } for shown, original in enumerate(order)],
        "public_only": True,
    }


def packet_event_ids(packet: Mapping[str, Any]) -> dict[int, set[str]]:
    """Extract auditable event IDs from a public selector packet."""
    allowed: dict[int, set[str]] = {}
    for candidate in packet.get("candidates") or ():
        shown = candidate.get("candidate_index")
        if not isinstance(shown, int):
            continue
        trace = candidate.get("public_trace_ir") or {}
        event_ids = {
            str(event.get("event_id"))
            for event in trace.get("events") or ()
            if event.get("event_id")
        }
        for diagnostic in trace.get("public_recovery_diagnostics") or ():
            event_ids.update(str(value) for value in diagnostic.get("event_ids") or ())
        allowed[shown] = event_ids
    return allowed

def resolve_oagents_choice(
    response: Any, shown_to_original: list[int],
) -> tuple[int, dict[str, Any]]:
    """Resolve strict tool output, then explicit prose-wrapped index output."""
    original, details = resolve_listwise_choice(response, shown_to_original)
    if not details.get("fallback"):
        return original, details
    content = response.get("content") if isinstance(response, Mapping) else getattr(response, "content", None)
    if isinstance(content, list):
        content = " ".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, Mapping) else str(item)
            for item in content
        )
    text = str(content or "")
    patterns = (
        r'"(?:index|candidate_index)"\s*:\s*([0-9]+)',
        r"(?:selected|choose|choice)\s+(?:candidate\s*)?([0-9]+)\b",
        r"\bcandidate\s+([0-9]+)\s+(?:is|has|best|strongest)",
    )
    matches = []
    for pattern in patterns:
        matches.extend(int(value) for value in re.findall(pattern, text, flags=re.IGNORECASE))
    valid = {value for value in matches if value in range(len(shown_to_original))}
    if len(valid) != 1:
        return original, details
    shown = valid.pop()
    selected = int(shown_to_original[shown])
    return selected, {
        "fallback": False,
        "decision": {
            "candidate_index": shown,
            "selected_original_index": selected,
            "reason": "explicit prose-wrapped list-wise choice",
            "parser": "oagents_explicit_index_v1",
        },
    }
