"""Public, evaluator-independent evidence for DeepPlanning selection."""
from __future__ import annotations

import re
from typing import Any

DAY_RE = re.compile(r"(?:^|\n)\s*(?:day|第\s*)\s*(\d+)", re.IGNORECASE)
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
TRAVEL_TOOLS = {"query_flight_info", "query_train_info", "query_route_info", "query_hotel_info"}


def _plan_text(text: str) -> str:
    match = re.search(r"<plan>(.*?)</plan>", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else (text or "")


def public_plan_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    """Summarize only visible plan text and tool results, never benchmark outcomes."""
    raw_text = str(candidate.get("final_answer") or "")
    text = _plan_text(raw_text)
    days = sorted({int(x) for x in DAY_RE.findall(text)})
    events = candidate.get("events", [])
    failures = sum(str(e.get("result_category")) == "FAILURE" for e in events)
    supported = sum(
        str(e.get("tool") or "") in TRAVEL_TOOLS
        and str(e.get("result_category")) == "SUCCESS" for e in events
    )
    return {
        "schema_signals": {
            "has_plan_block": bool(re.search(r"<plan>.*?</plan>", raw_text, re.IGNORECASE | re.DOTALL)),
            "day_count": len(days), "day_numbers": days[:14],
            "has_time_entries": bool(TIME_RE.search(text)), "has_daily_structure": bool(days),
        },
        "public_grounding": {"successful_travel_lookups": supported,
                             "tool_event_count": len(events), "visible_failure_events": failures},
        "risk_signals": {"empty_final_answer": not bool(text.strip()),
                          "tool_failure_present": failures > 0,
                          "unverified_plan_signal": bool(text.strip()) and supported == 0},
        "public_only": True, "outcome_used": False,
    }


def adapter_frontier(candidate: dict[str, Any], credit: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "deepplanning_public_evidence": public_plan_evidence(candidate),
        "adapter_instruction": (
            "For DeepPlanning travel candidates, inspect public plan structure and grounding first: "
            "parseable daily plans, time entries, successful visible travel lookups, and visible tool "
            "failures. Use these as evidence only; never infer evaluator labels or hidden state."
        ),
        "event_credit_state": credit or {"available": False},
        "public_only": True, "outcome_used": False,
    }
