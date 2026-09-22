"""APPWorld-specific public evidence adapter for StateTrace-EDS-ECA.

The EDS-ECA controller remains dataset agnostic.  APPWorld needs a small
adapter because API documentation is verbose and a single ReAct code block
may contain several API calls with one shared execution output.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
import copy


_API_CALL_RE = re.compile(r"^apis\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)$")
_STRONG_FAILURE_RE = re.compile(
    r"(?:execution\s+failed|traceback\s*\(.*?\)|exception\s*:|"
    r"response\s+status\s+code\s+is\s+[45]\d\d|\bHTTP\s+[45]\d\d\b)",
    re.IGNORECASE | re.DOTALL,
)
_AUTH_FAILURE_RE = re.compile(r"\b(?:unauthorized|forbidden)\b|\b(?:401|403)\b", re.IGNORECASE)
_NOT_FOUND_RE = re.compile(r"\b(?:not found|does not exist)\b", re.IGNORECASE)
_DOC_APIS = {"show_api_doc", "show_api_descriptions"}
_READ_PREFIXES = ("show_", "search_", "get_", "list_", "find_", "check_", "lookup_")
_MUTATE_PREFIXES = (
    "create_", "update_", "delete_", "add_", "remove_", "send_", "approve_",
    "reject_", "set_", "mark_", "modify_", "move_", "book_", "cancel_",
    "complete_", "transfer_", "pay_", "subscribe_", "unsubscribe_",
)


def _jsonable_ast(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def _is_api_doc(app: str, api: str) -> bool:
    return app == "api_docs" or api in _DOC_APIS


def _operation_kind(api: str) -> str:
    if api.startswith(_READ_PREFIXES):
        return "read"
    if api.startswith(_MUTATE_PREFIXES):
        return "mutation"
    if api in {"login", "logout", "signup", "verify_account"}:
        return "auth"
    return "unknown"


def _argument_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def extract_appworld_api_calls(code: str) -> list[dict[str, Any]]:
    """Extract API calls in source order, without executing model code."""

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return []
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _API_CALL_RE.match(ast.unparse(node.func))
    ]
    nodes.sort(key=lambda node: (node.lineno, node.col_offset, getattr(node, "end_lineno", node.lineno)))
    calls: list[dict[str, Any]] = []
    for node in nodes:
        match = _API_CALL_RE.match(ast.unparse(node.func))
        assert match is not None
        calls.append(
            {
                "app": match.group(1),
                "api": match.group(2),
                "positional": [_jsonable_ast(value) for value in node.args],
                "arguments": {
                    item.arg: _jsonable_ast(item.value)
                    for item in node.keywords
                    if item.arg is not None
                },
            }
        )
    return calls


def _compact_output(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 48] + "...[public output truncated]"


def _failure_kind(output: str, app: str, api: str) -> str:
    if _is_api_doc(app, api):
        return "none"
    if _AUTH_FAILURE_RE.search(output) and (
        _STRONG_FAILURE_RE.search(output) or "status code" in output.lower()
    ):
        return "auth_failure"
    if _STRONG_FAILURE_RE.search(output):
        return "runtime_failure"
    if _NOT_FOUND_RE.search(output) and ("exception" in output.lower() or "status code" in output.lower()):
        return "not_found"
    return "none"


def _result_shape(output: str) -> tuple[str, int | None, bool, bool]:
    try:
        value = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return ("text" if output else "empty", None, False, False)
    if isinstance(value, list):
        return ("array", len(value), any(isinstance(x, Mapping) and "id" in x for x in value), False)
    if isinstance(value, Mapping):
        has_id = any(key == "id" or str(key).endswith("_id") for key in value)
        has_error = bool(value.get("error"))
        return ("object", 1, has_id, has_error)
    return (type(value).__name__, 1, False, False)


def _public_event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    output = str(event.get("explicit_result") or "")
    is_doc = bool(event.get("is_api_documentation"))
    if is_doc:
        excerpt = "API documentation call; documentation text omitted from selector packet."
    elif event.get("failure_signal"):
        excerpt = _compact_output(output, 760)
    else:
        excerpt = _compact_output(output, 520)
    summary = {
        "event_id": str(event.get("event_id") or ""),
        "step_id": str(event.get("step_id") or ""),
        "batch_call_index": int(event.get("batch_call_index") or 1),
        "batch_size": int(event.get("batch_size") or 1),
        "app": str(event.get("app") or ""),
        "api": str(event.get("api") or ""),
        "operation": str(event.get("operation") or "unknown"),
        "argument_keys": sorted(str(key) for key in (event.get("arguments") or {})),
        "argument_types": dict(event.get("argument_types") or {}),
        "positional_arity": len(event.get("positional") or []),
        "failure_signal": bool(event.get("failure_signal")),
        "failure_kind": str(event.get("failure_kind") or "none"),
        "failure_scope": str(event.get("failure_scope") or "none"),
        "result_shape": str(event.get("result_shape") or "unknown"),
        "result_cardinality": event.get("result_cardinality"),
        "public_id_present": bool(event.get("public_id_present")),
        "result_has_error_field": bool(event.get("result_has_error_field")),
        "is_api_documentation": is_doc,
        "public_result_excerpt": excerpt,
    }
    if not is_doc:
        summary["arguments"] = event.get("arguments") or {}
    return summary


def _focus_events(events: Sequence[Mapping[str, Any]], limit: int = 36) -> list[dict[str, Any]]:
    if len(events) <= limit:
        return [_public_event_summary(event) for event in events]
    keep: set[int] = set(range(min(4, len(events))))
    keep.update(range(max(0, len(events) - 8), len(events)))
    for index, event in enumerate(events):
        if event.get("failure_signal") or event.get("operation") == "mutation":
            keep.update(range(max(0, index - 1), min(len(events), index + 2)))
    indices = sorted(keep)
    if len(indices) > limit:
        indices = indices[:limit]
    return [_public_event_summary(events[index]) for index in indices]


def _trajectory_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "event_count": len(events),
        "failure_signal_count": sum(bool(event.get("failure_signal")) for event in events),
        "mutation_count": sum(event.get("operation") == "mutation" for event in events),
        "read_count": sum(event.get("operation") == "read" for event in events),
        "verification_like_count": sum(
            event.get("operation") == "read" and index > 0 and events[index - 1].get("operation") == "mutation"
            for index, event in enumerate(events)
        ),
        "documentation_call_count": sum(bool(event.get("is_api_documentation")) for event in events),
        "batch_step_count": sum(int(event.get("batch_size") or 1) > 1 for event in events),
    }


def compile_appworld_public_events(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compile public events with conservative, APPWorld-aware failure labels."""

    events: list[dict[str, Any]] = []
    for step_index, step in enumerate(trace, start=1):
        output = _compact_output(step.get("execution_output"))
        calls = extract_appworld_api_calls(str(step.get("code") or ""))
        batch_size = len(calls)
        for call_index, call in enumerate(calls, start=1):
            kind = _failure_kind(output, call["app"], call["api"])
            result_shape, cardinality, public_id, has_error = _result_shape(output)
            event = {
                "event_id": f"s{step_index:02d}c{call_index:02d}",
                "step_id": f"s{step_index:02d}",
                "batch_call_index": call_index,
                "batch_size": batch_size,
                **call,
                "operation": _operation_kind(call["api"]),
                "explicit_result": output,
                "failure_kind": kind,
                "failure_signal": kind != "none",
                "failure_scope": "call" if batch_size == 1 else ("step" if kind != "none" else "none"),
                "explicit_failure": kind != "none" and batch_size == 1,
                "attributed_failure": kind != "none" and batch_size == 1,
                "result_shape": result_shape,
                "result_cardinality": cardinality,
                "public_id_present": public_id,
                "result_has_error_field": has_error,
                "is_api_documentation": _is_api_doc(call["app"], call["api"]),
            }
            event["argument_types"] = {
                str(key): _argument_type(value) for key, value in event["arguments"].items()
            }
            events.append(event)
    return events


def appworld_selector_packet(
    task_instruction: str,
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact packet while retaining public evidence and event IDs."""

    return {
        "task_instruction": task_instruction,
        "candidate_A": {
            "summary": _trajectory_summary(left),
            "public_events": _focus_events(left),
        },
        "candidate_B": {
            "summary": _trajectory_summary(right),
            "public_events": _focus_events(right),
        },
    }


def appworld_flat_selector_packet(
    task_instruction: str,
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose chronological atomic events without hierarchy-level aggregates."""
    return {
        "task_instruction": task_instruction,
        "candidate_A": {"atomic_events": _focus_events(left)},
        "candidate_B": {"atomic_events": _focus_events(right)},
        "event_representation": "flat_atomic_events",
    }


def namespace_appworld_events(events: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    """Give selector-visible events candidate-local IDs for safe credit mapping."""
    result = []
    for event in events:
        item = copy.deepcopy(dict(event))
        source_id = str(item.get("event_id") or "")
        item["source_event_id"] = source_id
        item["event_id"] = f"{label}:{source_id}"
        result.append(item)
    return result


def appworld_event_frontier(
    incumbent: Sequence[Mapping[str, Any]],
    prior_disagreement: Mapping[str, Any] | None = None,
    prior_credit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a literal-free accepted-incumbent frontier for a fresh proposal."""
    structural = []
    for event in incumbent:
        structural.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "step_id": str(event.get("step_id") or ""),
                "app": str(event.get("app") or ""),
                "api": str(event.get("api") or ""),
                "operation": str(event.get("operation") or "unknown"),
                "argument_keys": sorted(str(k) for k in (event.get("arguments") or {})),
                "argument_types": dict(event.get("argument_types") or {}),
                "failure_kind": str(event.get("failure_kind") or "none"),
                "failure_scope": str(event.get("failure_scope") or "none"),
                "result_shape": str(event.get("result_shape") or "unknown"),
                "result_cardinality": event.get("result_cardinality"),
                "is_api_documentation": bool(event.get("is_api_documentation")),
            }
        )
    failures = [e for e in structural if e["failure_kind"] != "none"]
    mutations = [e for e in structural if e["operation"] == "mutation"]
    reads = [e for e in structural if e["operation"] == "read"]
    active = None
    if failures:
        active = {
            "source": "current_public_appworld_failure",
            "issue_type": "FAILURE_RECOVERY",
            "event_id": failures[-1]["event_id"],
            "instruction": "Re-ground the failed target and preconditions, choose a different legal action, then verify.",
        }
    elif prior_credit and prior_credit.get("available") and prior_credit.get("unresolved_issue_type") not in {None, "NONE"}:
        active = {
            "source": "prior_event_credit",
            "issue_type": str(prior_credit.get("unresolved_issue_type")),
            "instruction": str((prior_credit.get("active_intervention") or {}).get("instruction") or "Re-ground the next action from public facts."),
        }
    return {
        "schema_version": "appworld_state_trace_eds_eca_frontier_v1",
        "event_hierarchy": {
            "documentation": sum(e["is_api_documentation"] for e in structural),
            "reads": len(reads),
            "mutations": len(mutations),
            "failures": len(failures),
            "events": structural[-36:],
        },
        "active_intervention": active,
        "prior_disagreement": copy.deepcopy(dict(prior_disagreement or {})),
        "event_credit_state": copy.deepcopy(dict(prior_credit or {})),
        "event_intervention": {
            "preserve_supported_prefix": True,
            "reacquire_current_state": True,
            "replay_prior_calls": False,
            "max_active_interventions": 1,
        },
        "public_only": True,
        "outcome_used": False,
        "literal_replay": False,
    }


def appworld_atomic_frontier(
    incumbent: Sequence[Mapping[str, Any]],
    prior_credit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a literal-free flat frontier for the no-hierarchy ablation."""
    atomic_events = []
    for event in incumbent[-36:]:
        atomic_events.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "app": str(event.get("app") or ""),
                "api": str(event.get("api") or ""),
                "operation": str(event.get("operation") or "unknown"),
                "argument_keys": sorted(str(k) for k in (event.get("arguments") or {})),
                "argument_types": dict(event.get("argument_types") or {}),
                "failure_signal": bool(event.get("failure_signal")),
                "result_shape": str(event.get("result_shape") or "unknown"),
            }
        )
    return {
        "schema_version": "appworld_state_trace_flat_atomic_frontier_v1",
        "atomic_events": atomic_events,
        "event_credit_state": copy.deepcopy(dict(prior_credit or {})),
        "public_only": True,
        "outcome_used": False,
        "literal_replay": False,
    }


def appworld_preserve_only_frontier(
    incumbent: Sequence[Mapping[str, Any]],
    prior_credit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose only non-failure structure for the no-RDS ablation."""
    events = []
    for event in incumbent:
        if event.get("failure_signal"):
            continue
        events.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "app": str(event.get("app") or ""),
                "api": str(event.get("api") or ""),
                "operation": str(event.get("operation") or "unknown"),
                "argument_keys": sorted(str(k) for k in (event.get("arguments") or {})),
                "argument_types": dict(event.get("argument_types") or {}),
                "result_shape": str(event.get("result_shape") or "unknown"),
                "is_api_documentation": bool(event.get("is_api_documentation")),
            }
        )
    return {
        "schema_version": "appworld_state_trace_preserve_only_frontier_v1",
        "event_hierarchy": {"successful_events": events[-36:]},
        "active_intervention": None,
        "event_credit_state": copy.deepcopy(dict(prior_credit or {})),
        "event_intervention": {
            "preserve_supported_prefix": True,
            "rds_avoidance": False,
            "require_fresh_verification": True,
        },
        "public_only": True,
        "outcome_used": False,
        "literal_replay": False,
    }


def appworld_rds_only_frontier(
    incumbent: Sequence[Mapping[str, Any]],
    prior_credit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose only failure/RDS structure for the no-preserve ablation."""
    failures = []
    for event in incumbent:
        if not event.get("failure_signal"):
            continue
        failures.append(
            {
                "event_id": str(event.get("event_id") or ""),
                "app": str(event.get("app") or ""),
                "api": str(event.get("api") or ""),
                "operation": str(event.get("operation") or "unknown"),
                "argument_keys": sorted(str(k) for k in (event.get("arguments") or {})),
                "argument_types": dict(event.get("argument_types") or {}),
                "failure_kind": str(event.get("failure_kind") or "none"),
                "failure_scope": str(event.get("failure_scope") or "none"),
            }
        )
    active = None
    if failures:
        active = {
            "source": "current_public_appworld_failure",
            "issue_type": "FAILURE_RECOVERY",
            "event_id": failures[-1]["event_id"],
            "instruction": "Re-ground the failed target and preconditions, choose a different legal action, then verify.",
        }
    elif prior_credit and prior_credit.get("active_intervention"):
        active = copy.deepcopy(prior_credit["active_intervention"])
    return {
        "schema_version": "appworld_state_trace_rds_only_frontier_v1",
        "event_hierarchy": {"recovery_risk_events": failures[-36:]},
        "active_intervention": active,
        "event_credit_state": copy.deepcopy(dict(prior_credit or {})),
        "event_intervention": {
            "preserve_supported_prefix": False,
            "rds_avoidance": True,
            "require_fresh_verification": True,
        },
        "public_only": True,
        "outcome_used": False,
        "literal_replay": False,
    }


def appworld_public_disagreement(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare public structural transaction traces before paying for a selector."""
    def signature(events: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
        return [
            (
                e.get("app"), e.get("api"), e.get("operation"),
                tuple(sorted((e.get("arguments") or {}).keys())),
                tuple(sorted((e.get("argument_types") or {}).items())),
                e.get("failure_kind"), e.get("failure_scope"),
                e.get("result_shape"), e.get("result_cardinality"),
            )
            for e in events
        ]
    left_sig, right_sig = signature(left), signature(right)
    return {
        "material_disagreement": left_sig != right_sig,
        "left_event_count": len(left),
        "right_event_count": len(right),
        "left_failure_count": sum(e.get("failure_signal") for e in left),
        "right_failure_count": sum(e.get("failure_signal") for e in right),
        "left_mutation_count": sum(e.get("operation") == "mutation" for e in left),
        "right_mutation_count": sum(e.get("operation") == "mutation" for e in right),
    }
