import json
from types import SimpleNamespace

from occubench_eds_eca_selector import (
    build_selector_packet,
    credit_guidance,
    resolve_selector_response,
    trajectory_events,
)


def _response(arguments):
    function = SimpleNamespace(
        name="select_and_credit_events",
        arguments=json.dumps(arguments),
    )
    call = SimpleNamespace(function=function)
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _trajectory(tool, result):
    return (
        f"[Agent Action]: Call tool `{tool}` with arguments: {{\"id\": \"X-1\"}}\n"
        f"[Environment Observation] (from {tool}): {result}\n"
        "[Agent Response]: Finished after checking the public result.\n"
    )


def test_events_use_candidate_scoped_ids():
    left = trajectory_events(_trajectory("check_order", "ok"), "candidate-0")
    right = trajectory_events(_trajectory("check_order", "ok"), "candidate-1")
    assert left[0]["event_id"] == "candidate-0-e000"
    assert right[0]["event_id"] == "candidate-1-e000"
    assert {row["event_id"] for row in left}.isdisjoint(
        {row["event_id"] for row in right}
    )


def test_packet_excludes_hidden_verification_plan_and_outcomes():
    task = {
        "task_scenario_name": "Order review",
        "agent_instruction": "Correctly reject an invalid order.",
        "verification_plan": "SECRET GOLD",
    }
    incumbent = {"trajectory": _trajectory("reject_order", "rejected")}
    proposal = {"trajectory": _trajectory("approve_order", "approved")}
    packet, order, allowed = build_selector_packet(task, incumbent, proposal, 7)
    text = json.dumps(packet)
    assert "SECRET GOLD" not in text
    assert "is_correct" not in text
    assert sorted(order) == [0, 1]
    assert allowed[0].isdisjoint(allowed[1])


def test_valid_semantic_selection_maps_shuffled_candidate():
    task = {"task_scenario_name": "x", "agent_instruction": "reject invalid"}
    incumbent = {"trajectory": _trajectory("reject_order", "rejected")}
    proposal = {"trajectory": _trajectory("approve_order", "approved")}
    packet, order, allowed = build_selector_packet(task, incumbent, proposal, 9)
    shown = order.index(1)
    rejected = 1 - shown
    selected_id = sorted(allowed[shown])[0]
    rejected_id = sorted(allowed[rejected])[0]
    response = _response({
        "candidate_index": shown,
        "reason": "The visible instruction requires rejection.",
        "supporting_event_ids": [selected_id],
        "preserve_event_ids": [selected_id],
        "avoid_event_ids": [rejected_id],
        "unresolved_issue_type": "NONE",
        "unresolved_issue": "",
    })
    selected, details = resolve_selector_response(response, order, allowed)
    assert selected == 1
    assert details["fallback"] is False


def test_cross_candidate_credit_fails_closed_to_incumbent():
    task = {"task_scenario_name": "x", "agent_instruction": "do x"}
    row = {"trajectory": _trajectory("check", "ok")}
    _, order, allowed = build_selector_packet(task, row, row, 3)
    shown = order.index(1)
    other = 1 - shown
    wrong_id = sorted(allowed[other])[0]
    response = _response({
        "candidate_index": shown,
        "reason": "looks stronger",
        "supporting_event_ids": [wrong_id],
        "preserve_event_ids": [],
        "avoid_event_ids": [],
        "unresolved_issue_type": "NONE",
        "unresolved_issue": "",
    })
    selected, details = resolve_selector_response(response, order, allowed)
    assert selected == 0
    assert details["fallback_reason"] == "unsupported_selection_evidence"


def test_credit_guidance_is_structural_not_selector_prose():
    details = {
        "fallback": False,
        "decision": {
            "unresolved_issue_type": "TASK_COVERAGE",
            "unresolved_issue": "Repeat secret ID X-1",
            "preserve_event_ids": ["candidate-0-e001"],
            "avoid_event_ids": ["candidate-1-e001"],
        },
    }
    value = credit_guidance(details)
    assert value["available"] is True
    assert value["issue_type"] == "TASK_COVERAGE"
    assert "X-1" not in str(value)
