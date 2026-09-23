"""OccuBench adapter for the original StateBench EDS-ECA algorithm.

Only the environment boundary is benchmark-specific: OccuBench's text trace
is converted into StateBench's public conversation representation. Event
analysis, disagreement detection, event credit, and frontier construction are
then delegated to the StateBench implementation.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent / "vendor" / "statebench"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state_trace_eds_ec import (  # noqa: E402
    SELECT_INSTRUCTION,
    build_eds_ec_frontier,
    build_event_credit_state,
    event_credit_tool,
    resolve_event_credit,
)
from state_trace_event_scaling import (  # noqa: E402
    analyze_public_events,
    compare_public_trajectories,
    event_scaling_message,
)
from state_trace_pool_select import (  # noqa: E402
    build_packet as build_statebench_packet,
    packet_event_ids,
)


MARKER_RE = re.compile(
    r"(?m)^\[(Agent Response|Agent Action|Environment Observation)\]"
    r"(?: \(from ([^)]+)\))?:\s*"
)
ACTION_RE = re.compile(
    r"^Call tool\s+`([^`]+)`\s+with arguments:\s*(.*)$",
    re.DOTALL,
)
READ_TOOL_HINTS = (
    "get", "retrieve", "search", "list", "lookup", "query", "scan", "inspect",
    "check", "read", "fetch", "calculate", "compute", "assess",
)
WRITE_TOOL_HINTS = (
    "update", "set", "submit", "create", "delete", "remove", "modify", "change",
    "apply", "run", "execute", "record", "log", "finalize", "close", "open",
    "approve", "reject", "assign", "allocate", "transfer", "schedule", "issue",
    "cancel", "book", "reserve", "send", "process",
)


def _json_or_text(value: str) -> Any:
    text = value.strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _public_arguments(value: Any) -> Any:
    # Distinguish a requested tool argument from evaluator target-state metadata.
    if isinstance(value, Mapping):
        return {
            ("public_tool_target_state" if key == "target_state" else key):
            _public_arguments(child) for key, child in value.items()
        }
    if isinstance(value, list):
        return [_public_arguments(child) for child in value]
    return value


def occubench_public_row(
    task: Mapping[str, Any],
    stage_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one OccuBench stage into a StateBench-compatible public row."""
    trajectory = str(stage_row.get("trajectory") or "")
    matches = list(MARKER_RE.finditer(trajectory))
    conversation: list[dict[str, Any]] = [{
        "role": "user",
        "content": str(task.get("agent_instruction") or ""),
    }]
    pending_calls: list[dict[str, Any]] = []
    call_count = 0

    for index, match in enumerate(matches):
        kind = match.group(1)
        source_tool = str(match.group(2) or "")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(trajectory)
        content = trajectory[match.end():end].strip()

        if kind == "Agent Response":
            if content:
                conversation.append({"role": "assistant", "content": content})
            continue

        if kind == "Agent Action":
            parsed = ACTION_RE.match(content)
            if parsed is None:
                continue
            tool_name = parsed.group(1).strip()
            arguments = _json_or_text(parsed.group(2))
            if not isinstance(arguments, Mapping):
                arguments = {"_raw_arguments": arguments}
            call_id = f"occubench-call-{call_count:04d}"
            call_count += 1
            call = {
                "id": call_id,
                "name": tool_name,
                "arguments": _public_arguments(arguments),
            }
            conversation.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [call],
            })
            pending_calls.append(call)
            continue

        result = _json_or_text(content)
        call = pending_calls.pop(0) if pending_calls else None
        if call is not None:
            call["result"] = result
            call_id = str(call["id"])
            tool_name = str(call["name"])
        else:
            call_id = f"occubench-orphan-{index:04d}"
            tool_name = source_tool
        conversation.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": source_tool or tool_name,
            "content": result,
        })

    return {
        "task_key": f"occubench::{int(task['task_id'])}",
        "domain": str(task.get("domain") or task.get("category") or "occubench"),
        "conversation": conversation,
    }


def statebench_tool_schemas(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose OccuBench action definitions in the schema form StateBench accepts."""
    schemas = []
    for source in config.get("action_set_definitions") or ():
        if not isinstance(source, Mapping) or not source.get("name"):
            continue
        item = dict(source)
        normalized = re.sub(r"[^a-z0-9]+", "_", str(item["name"]).lower()).strip("_")
        first = normalized.partition("_")[0]
        description = str(item.get("description") or "")
        lower_description = description.lower()
        write_description = any(token in lower_description for token in (
            "update", "record", "submit", "create", "delete", "modify", "change",
            "execute", "initiate", "finalize", "store", "save", "log ", "write",
        ))
        if first in WRITE_TOOL_HINTS or write_description:
            role = "This action updates or writes environment state."
        elif first in READ_TOOL_HINTS:
            role = "This action reads or retrieves environment state."
        else:
            role = ""
        item["description"] = f"{description} {role}".strip()
        schemas.append(item)
    return schemas


def analyze_stage(
    task: Mapping[str, Any],
    stage_row: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return analyze_public_events(occubench_public_row(task, stage_row), schemas)


def compare_stages(
    task: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    proposal: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return compare_public_trajectories(
        occubench_public_row(task, incumbent),
        occubench_public_row(task, proposal),
        schemas,
    )


def build_generation_guidance(
    task: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    prior_disagreement: Mapping[str, Any] | None,
    prior_credit_state: Mapping[str, Any] | None,
    round_index: int,
    max_event_groups: int = 12,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    public_incumbent = occubench_public_row(task, incumbent)
    ledger, frontier = build_eds_ec_frontier(
        public_incumbent,
        schemas,
        prior_disagreement,
        max_event_groups,
        prior_credit_state,
    )
    return ledger, frontier, event_scaling_message(frontier, round_index)


def build_selector_packet(
    task: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    proposal: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[
    dict[str, Any],
    list[int],
    dict[int, set[str]],
    list[dict[str, Any]],
]:
    candidates = [
        occubench_public_row(task, incumbent),
        occubench_public_row(task, proposal),
    ]
    order = [0, 1]
    random.Random(seed).shuffle(order)
    packet = build_statebench_packet(candidates, schemas, order)
    packet.update({
        "candidate_identity_removed": True,
        "candidate_winner_included": False,
    })
    return packet, order, packet_event_ids(packet), candidates


def openai_event_credit_tool(count: int = 2) -> dict[str, Any]:
    """Wrap StateBench's tool declaration for an OpenAI-compatible endpoint."""
    return {"type": "function", "function": event_credit_tool(count)}


def resolve_selector(
    response: Any,
    order: Sequence[int],
    allowed_event_ids: Mapping[int, set[str]],
) -> tuple[int, dict[str, Any]]:
    choices = getattr(response, "choices", None)
    if choices:
        response = getattr(choices[0], "message", response)
    raw_calls = (
        list(response.get("tool_calls") or ())
        if isinstance(response, Mapping)
        else list(getattr(response, "tool_calls", ()) or ())
    )
    normalized_calls = []
    for call in raw_calls:
        function = (
            call.get("function")
            if isinstance(call, Mapping)
            else getattr(call, "function", None)
        )
        if function is not None:
            name = (
                function.get("name")
                if isinstance(function, Mapping)
                else getattr(function, "name", None)
            )
            arguments = (
                function.get("arguments")
                if isinstance(function, Mapping)
                else getattr(function, "arguments", None)
            )
        else:
            name = call.get("name") if isinstance(call, Mapping) else getattr(call, "name", None)
            arguments = call.get("arguments") if isinstance(call, Mapping) else getattr(call, "arguments", None)
        normalized_calls.append({"name": name, "arguments": arguments})
    return resolve_event_credit(
        {"tool_calls": normalized_calls}, order, allowed_event_ids,
    )


def build_credit_state(
    public_candidates: Sequence[Mapping[str, Any]],
    selected_index: int,
    selector_details: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return build_event_credit_state(
        public_candidates,
        selected_index,
        selector_details,
        schemas,
        prior_state,
    )
