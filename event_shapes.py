"""Extracted from the experimental state_trace_eds_eca_v2.py; algorithm bodies unchanged."""

from __future__ import annotations

from typing import Any

def argument_shape(key: str, value: Any) -> str:
    """Map dataset-specific argument names to a small public type shape."""
    lowered = key.lower()
    if isinstance(value, bool):
        return "boolean"
    if any(token in lowered for token in ("date", "time", "day", "month")):
        return "temporal"
    if any(token in lowered for token in (
        "id", "ref", "code", "number", "sku", "entity", "object", "item",
        "order", "booking", "reservation", "product", "user", "account",
    )):
        return "identifier"
    if any(token in lowered for token in ("price", "fee", "cost", "total", "amount", "count", "qty")):
        return "numeric"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "numeric"
    return "text"
