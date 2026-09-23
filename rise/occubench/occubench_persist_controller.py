#!/usr/bin/env python3
"""Conservative, online-visible PERSIST-ACE controller for OccuBench."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


MUTATION_TERMS = {
    "activate", "add", "advance", "approve", "assign", "book", "cancel", "close", "configure",
    "control", "create", "deactivate", "delete", "execute", "install", "issue", "manage",
    "initialize", "modify", "open", "pay", "record", "refill", "register", "remove", "renew", "resolve",
    "restore", "return", "set", "start", "stop", "submit", "toggle", "transfer", "update",
}
READ_TERMS = {
    "access", "analyze", "assess", "calculate", "check", "compute", "diagnose", "fetch",
    "find", "get", "inspect", "list", "lookup", "monitor", "query", "read", "retrieve",
    "search", "test", "validate", "verify", "view",
}
MUTATION_DESCRIPTION_TERMS = {
    "advances", "changes", "controls", "creates", "deletes", "executes", "initializes",
    "initiates", "modifies", "sets", "submits", "updates", "writes",
}
READ_DESCRIPTION_TERMS = {
    "calculates", "checks", "fetches", "lists", "reads", "retrieves", "returns", "searches",
    "validates", "verifies", "views",
}
POLLING_DESCRIPTION_TERMS = {
    "active safety", "current", "metrics", "monitor", "performance data", "progress",
    "real-time", "remaining", "sensor", "starts or checks", "status", "telemetry", "time",
    "verification test",
}
EXPLICIT_ERROR_PATTERNS = {
    "500 error", "connection refused", "exception", "http 500", "internal error",
    "rate limit", "service unavailable", "temporarily unavailable", "timed out", "timeout",
    "try again",
}


def normalize_observation(text: str) -> str:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return re.sub(r"\s+", " ", text).strip()
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_arguments(value: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return re.sub(r"\s+", " ", value).strip()
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def observation_has_explicit_error(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        if parsed.get("error") not in (None, "", False):
            return True
        status_code = parsed.get("status_code", parsed.get("http_status"))
        if isinstance(status_code, int) and status_code >= 500:
            return True
        status = str(parsed.get("status", "")).strip().lower().replace(" ", "_")
        if status in {"service_unavailable", "internal_error", "timeout"}:
            return True
    lower = text.lower()
    return any(pattern in lower for pattern in EXPLICIT_ERROR_PATTERNS)


def observation_is_failure(text: str) -> bool:
    if observation_has_explicit_error(text):
        return True
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        status = str(parsed.get("status", "")).strip().lower().replace(" ", "_")
        return status in {"error", "failed", "failure", "rejected", "denied"}
    lower = text.lower()
    return " not found" in lower or lower.startswith("error:")


def words(value: str) -> set[str]:
    return {part for part in value.lower().replace("-", "_").split("_") if part}


def unwrap_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    schema = schema or {}
    return schema.get("function", schema)


def classify_tool(schema: Optional[Dict[str, Any]], tool_name: str) -> str:
    function = unwrap_schema(schema)
    name_words = words(tool_name)
    description_words = words(str(function.get("description", "")).replace(" ", "_"))
    if name_words & MUTATION_TERMS or description_words & MUTATION_DESCRIPTION_TERMS:
        return "write_or_state_change"
    if name_words & READ_TERMS or description_words & READ_DESCRIPTION_TERMS:
        return "read_only"
    return "unknown_high_risk"


def is_dynamic_polling(schema: Optional[Dict[str, Any]]) -> bool:
    description = str(unwrap_schema(schema).get("description", "")).lower()
    return any(term in description for term in POLLING_DESCRIPTION_TERMS)


def build_schema_registry(tools: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for schema in tools:
        function = unwrap_schema(schema)
        name = function.get("name")
        if name:
            registry[str(name)] = schema
    return registry


@dataclass
class ControllerDecision:
    step: int
    trigger: str
    decision: str
    reason: str
    proposal_tool: str
    proposal_arguments: str
    candidate_tool: Optional[str]
    candidate_arguments: Optional[str]
    prior_same_effect_count: int
    prior_same_action_effect_count: int
    prior_same_family_effect_count: int
    prior_family_failure_streak: int
    trigger_scope: str
    proposal_risk: str
    gates: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrictPersistController:
    """One-shot conservative controller; uncertainty always executes Raw proposal."""

    def __init__(
        self,
        tools: Iterable[Dict[str, Any]],
        *,
        repeat_threshold: int = 3,
        max_length_continuations: int = 1,
    ) -> None:
        self.schemas = build_schema_registry(tools)
        self.repeat_threshold = repeat_threshold
        self.max_length_continuations = max_length_continuations
        self.effect_counts: Counter[Tuple[str, str, str]] = Counter()
        self.family_effect_counts: Counter[Tuple[str, str]] = Counter()
        self.action_effect_streaks: Dict[Tuple[str, str], int] = {}
        self.family_effect_streaks: Dict[str, int] = {}
        self.family_failure_streaks: Dict[str, int] = {}
        self.last_observation_by_action: Dict[Tuple[str, str], str] = {}
        self.last_observation_by_family: Dict[str, str] = {}
        self.last_effect_was_explicit_error: Dict[Tuple[str, str], bool] = {}
        self.last_family_effect_was_explicit_error: Dict[str, bool] = {}
        self.length_continuations = 0
        self.audit: List[ControllerDecision] = []

    def record_effect(self, tool_name: str, tool_arguments: str, observation: str) -> None:
        arguments = canonical_arguments(tool_arguments)
        normalized = normalize_observation(observation)
        action = (tool_name, arguments)
        previous_action = self.last_observation_by_action.get(action)
        previous_family = self.last_observation_by_family.get(tool_name)
        self.action_effect_streaks[action] = (
            self.action_effect_streaks.get(action, 0) + 1
            if previous_action == normalized
            else 1
        )
        self.family_effect_streaks[tool_name] = (
            self.family_effect_streaks.get(tool_name, 0) + 1
            if previous_family == normalized
            else 1
        )
        self.family_failure_streaks[tool_name] = (
            self.family_failure_streaks.get(tool_name, 0) + 1
            if observation_is_failure(observation)
            else 0
        )
        self.effect_counts[(tool_name, arguments, normalized)] += 1
        self.family_effect_counts[(tool_name, normalized)] += 1
        self.last_observation_by_action[(tool_name, arguments)] = normalized
        self.last_observation_by_family[tool_name] = normalized
        self.last_effect_was_explicit_error[(tool_name, arguments)] = observation_has_explicit_error(observation)
        self.last_family_effect_was_explicit_error[tool_name] = observation_has_explicit_error(observation)

    def inspect_proposal(
        self,
        *,
        step: int,
        tool_name: str,
        tool_arguments: str,
    ) -> Optional[ControllerDecision]:
        arguments = canonical_arguments(tool_arguments)
        action = (tool_name, arguments)
        latest_action = self.last_observation_by_action.get(action)
        latest_family = self.last_observation_by_family.get(tool_name)
        if latest_family is None:
            return None

        prior_action_count = self.action_effect_streaks.get(action, 0)
        prior_family_count = self.family_effect_streaks.get(tool_name, 0)
        prior_failure_streak = self.family_failure_streaks.get(tool_name, 0)
        prior_count = max(prior_action_count, prior_family_count, prior_failure_streak)
        productive_retry = (
            prior_failure_streak > 0
            and prior_failure_streak <= 2
        )
        if prior_count < self.repeat_threshold and not productive_retry:
            return None

        if prior_action_count >= self.repeat_threshold:
            trigger_scope = "exact_action_effect_streak"
        elif prior_family_count >= self.repeat_threshold:
            trigger_scope = "tool_family_effect_streak"
        else:
            trigger_scope = "tool_family_failure_streak"

        schema = self.schemas.get(tool_name)
        risk = classify_tool(schema, tool_name)
        dynamic_polling = is_dynamic_polling(schema)
        gates = {
            "online_visible": True,
            "progress_deficient": prior_count >= self.repeat_threshold,
            "productive_retry_protected": productive_retry,
            "read_only_proposal": risk == "read_only",
            "not_dynamic_polling": not dynamic_polling,
            "grounded_recovery_candidate": False,
            "bounded_one_shot": True,
            "no_unsafe_write": risk == "read_only",
            "positive_expected_utility": False,
        }

        if productive_retry:
            trigger = "explicit_error_retry"
            reason = "protect_productive_retry"
        elif risk != "read_only":
            trigger = "repeated_identical_effect"
            reason = "abstain_unsafe_or_unknown_proposal_family"
        elif dynamic_polling:
            trigger = "repeated_identical_effect"
            reason = "protect_dynamic_polling"
        else:
            trigger = "repeated_identical_effect"
            reason = "abstain_no_validated_grounded_recovery_card"

        decision = ControllerDecision(
            step=step,
            trigger=trigger,
            decision="execute_original",
            reason=reason,
            proposal_tool=tool_name,
            proposal_arguments=arguments,
            candidate_tool=None,
            candidate_arguments=None,
            prior_same_effect_count=prior_count,
            prior_same_action_effect_count=prior_action_count,
            prior_same_family_effect_count=prior_family_count,
            prior_family_failure_streak=prior_failure_streak,
            trigger_scope=trigger_scope,
            proposal_risk=risk,
            gates=gates,
        )
        self.audit.append(decision)
        return decision

    def allow_length_continuation(self, finish_reason: Optional[str], has_tool_calls: bool) -> bool:
        if finish_reason != "length" or has_tool_calls:
            return False
        if self.length_continuations >= self.max_length_continuations:
            return False
        self.length_continuations += 1
        return True
