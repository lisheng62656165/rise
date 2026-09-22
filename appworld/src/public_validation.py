"""Checks that selector packets contain public evidence only."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FORBIDDEN_SELECTOR_KEYS = {
    "evaluation",
    "evaluation_success",
    "goal_completion",
    "ground_truth",
    "private_data",
    "reward",
    "success",
}


def reject_forbidden_fields(value: Any, path: str = "root") -> None:
    """Reject evaluator, reward, and hidden-state fields before an API request."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_SELECTOR_KEYS:
                raise ValueError(f"forbidden selector field at {path}.{key}")
            reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_fields(child, f"{path}[{index}]")
