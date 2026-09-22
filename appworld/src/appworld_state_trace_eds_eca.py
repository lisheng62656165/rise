from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .public_validation import reject_forbidden_fields


ISSUE_TYPES = (
    "NONE", "FAILURE_RECOVERY", "TARGET_IDENTITY", "PARAMETER_GROUNDING",
    "ACTION_ORDER", "VERIFICATION", "CONSENT", "TASK_COVERAGE", "STOPPING", "OTHER",
)

ISSUE_DIRECTIVES = {
    "NONE": "",
    "FAILURE_RECOVERY": "Do not repeat a rejected call. Reacquire its target and preconditions before choosing a different recovery action.",
    "TARGET_IDENTITY": "Reacquire the target identity and bind each mutation to a publicly supported object.",
    "PARAMETER_GROUNDING": "Recompute mutation arguments from current public facts and constraints before acting.",
    "ACTION_ORDER": "Re-establish public preconditions and execute dependent actions in a valid order.",
    "VERIFICATION": "Read back consequential mutations when a public read API is available before stopping.",
    "CONSENT": "Preserve preview and confirmation semantics; do not commit without required public consent.",
    "TASK_COVERAGE": "Complete every visible request and constraint without dropping required explanation or interaction.",
    "STOPPING": "Stop only after visible obligations are complete, and avoid post-success mutation.",
    "OTHER": "Avoid the rejected structural event pattern and re-ground the next action from current public evidence.",
}

SELECTOR_INSTRUCTION = """You are the StateTrace-EDS-ECA public-evidence selector.
Compare two completed AppWorld trajectories for the same visible request. Use only the
task instruction and public event summaries. Prefer grounded targets and arguments,
valid authentication and action order, explicit recovery, read-back after mutations,
visible task coverage, and no collateral mutation. Do not prefer a trajectory merely
because it is shorter. Documentation text is not a failure. A batched step has shared
failure scope; do not blame every call individually. Event credit is structural guidance
for a fresh environment: never copy literal identifiers, values, results, or calls.
Do not infer hidden requirements, evaluator labels, rewards, or success. Return one
short JSON object only, with no markdown and a reason of at most 12 words:
{"choice":"A" or "B","supporting_event_ids":["..."],"preserve_event_ids":["..."],"avoid_event_ids":["..."],"unresolved_issue_type":"NONE|FAILURE_RECOVERY|TARGET_IDENTITY|PARAMETER_GROUNDING|ACTION_ORDER|VERIFICATION|CONSENT|TASK_COVERAGE|STOPPING|OTHER","reason":"..."}
Use at most 3 IDs in each ID list."""

EXECUTION_INSTRUCTION = """StateTrace-EDS-ECA event-driven execution.
Solve the entire visible AppWorld task independently in this fresh environment. The
attached event credit contains structural hypotheses from earlier public trajectories.
Reacquire every object, identifier, value, and precondition with legal APIs. Preserve
useful event structure and avoid rejected structure, but never replay old literal values
or assume an earlier result still holds. Apply at most the one active intervention. Inspect
API documentation, avoid repeating rejected calls without new evidence, verify
consequential mutations when possible, and call supervisor.complete_task only after the
visible task is complete."""


@dataclass(frozen=True)
class CreditDecision:
    choice: str
    supporting_event_ids: tuple[str, ...]
    preserve_event_ids: tuple[str, ...]
    avoid_event_ids: tuple[str, ...]
    unresolved_issue_type: str
    reason: str
    fallback: bool = False


def selector_packet(task_instruction: str, left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    packet = {
        "task_instruction": task_instruction,
        "candidate_A": {"public_events": copy.deepcopy(list(left))},
        "candidate_B": {"public_events": copy.deepcopy(list(right))},
    }
    reject_forbidden_fields(packet)
    return packet


def selector_messages(packet: Mapping[str, Any]) -> list[dict[str, str]]:
    reject_forbidden_fields(packet)
    return [
        {"role": "system", "content": SELECTOR_INSTRUCTION},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=True, sort_keys=True)},
    ]


def parse_credit_decision(text: str, left_ids: set[str], right_ids: set[str]) -> CreditDecision:
    value = None
    decoder = json.JSONDecoder()
    raw_text = text or ""
    for match in re.finditer(r"\{", raw_text):
        try:
            candidate, _ = decoder.raw_decode(raw_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "choice" in candidate:
            value = candidate
            break
    if value is None:
        return CreditDecision("A", (), (), (), "NONE", "malformed selector response", True)
    choice = str(value.get("choice") or "").upper()
    issue = str(value.get("unresolved_issue_type") or "")
    fields = ("supporting_event_ids", "preserve_event_ids", "avoid_event_ids")
    if choice not in {"A", "B"} or issue not in ISSUE_TYPES:
        return CreditDecision("A", (), (), (), "NONE", "invalid selector fields", True)
    if any(not isinstance(value.get(field), list) or not all(isinstance(x, str) for x in value[field]) for field in fields):
        return CreditDecision("A", (), (), (), "NONE", "invalid credit lists", True)
    selected_ids, rejected_ids = (left_ids, right_ids) if choice == "A" else (right_ids, left_ids)
    supporting = tuple(value["supporting_event_ids"])
    preserve = tuple(value["preserve_event_ids"])
    avoid = tuple(value["avoid_event_ids"])
    if not supporting or not set(supporting).issubset(selected_ids):
        return CreditDecision("A", (), (), (), "NONE", "unsupported selection evidence", True)
    if not set(preserve).issubset(selected_ids) or not set(avoid).issubset(rejected_ids):
        return CreditDecision("A", (), (), (), "NONE", "cross-candidate event credit", True)
    return CreditDecision(choice, supporting, preserve, avoid, issue, str(value.get("reason") or ""), False)


def _structural_event(event: Mapping[str, Any]) -> dict[str, Any]:
    projected = {
        "source_event_id": str(event.get("event_id") or ""),
        "app": str(event.get("app") or ""),
        "api": str(event.get("api") or ""),
        "argument_keys": sorted(str(key) for key in (event.get("arguments") or {})),
        "argument_types": dict(event.get("argument_types") or {}),
        "positional_arity": len(event.get("positional") or []),
        "explicit_failure": bool(event.get("explicit_failure")),
    }
    for key in (
        "operation", "failure_kind", "failure_scope", "result_shape",
        "result_cardinality", "public_id_present", "result_has_error_field",
        "is_api_documentation", "batch_size", "batch_call_index",
    ):
        if key in event:
            projected[key] = event[key]
    return projected


def build_credit_state(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], decision: CreditDecision
) -> dict[str, Any]:
    if decision.fallback:
        return {"available": False, "preserve_events": [], "avoid_events": [], "unresolved_issue_type": "NONE", "active_intervention": None, "public_only": True}
    selected, rejected = (left, right) if decision.choice == "A" else (right, left)
    selected_by_id = {str(event.get("event_id")): event for event in selected}
    rejected_by_id = {str(event.get("event_id")): event for event in rejected}
    issue = decision.unresolved_issue_type
    if issue == "NONE" and decision.avoid_event_ids:
        issue = "FAILURE_RECOVERY" if any(rejected_by_id[event_id].get("explicit_failure") for event_id in decision.avoid_event_ids) else "OTHER"
    active = None if issue == "NONE" else {"issue_type": issue, "instruction": ISSUE_DIRECTIVES[issue]}
    state = {
        "available": True,
        "preserve_events": [_structural_event(selected_by_id[x]) for x in decision.preserve_event_ids if x in selected_by_id][:6],
        "avoid_events": [_structural_event(rejected_by_id[x]) for x in decision.avoid_event_ids if x in rejected_by_id][:6],
        "unresolved_issue_type": issue,
        "active_intervention": active,
        "public_only": True,
        "literal_replay": False,
    }
    reject_forbidden_fields(state)
    return state


def proposal_messages(base_messages: Sequence[Mapping[str, Any]], credit: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    messages = copy.deepcopy([dict(message) for message in base_messages])
    messages[0]["content"] = (
        str(messages[0].get("content") or "")
        + "\n\n" + EXECUTION_INSTRUCTION
        + f"\n\nCurrent stage: {stage}. Public structural event credit:\n"
        + json.dumps(credit, ensure_ascii=True, sort_keys=True)
    )
    return messages


def public_event_ids(events: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(event.get("event_id")) for event in events if event.get("event_id")}
