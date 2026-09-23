import json
import pytest
import threading
from types import SimpleNamespace

from occubench_statebench_eds_eca import (
    SELECT_INSTRUCTION,
    analyze_stage,
    build_credit_state,
    build_generation_guidance,
    build_selector_packet,
    compare_stages,
    resolve_selector,
    statebench_tool_schemas,
)


TASK = {
    "task_id": 7,
    "domain": "orders",
    "agent_instruction": "Update order X-1, verify it, and stop.",
    "verification_plan": "SECRET GOLD",
}
CONFIG = {
    "action_set_definitions": [
        {"name": "get_order", "description": "Retrieve an order", "parameters": {"type": "object"}},
        {"name": "update_order", "description": "Update an order", "parameters": {"type": "object"}},
        {"name": "run_mental_rep", "description": "Runs a simulation", "parameters": {"type": "object"}},
        {"name": "scan_inventory", "description": "Returns inventory", "parameters": {"type": "object"}},
    ]
}


@pytest.mark.parametrize("failures", [0, 1, 3])
@pytest.mark.parametrize("design", ["original", "atts-eca-pairwise-v1"])
def test_selector_validation_retry(monkeypatch, failures, design):
    import run_occubench_eds_eca_mimo as runner
    calls, closed = [], []
    semaphore = threading.BoundedSemaphore(1)
    ns = SimpleNamespace(
        selector_seed=12, selector_api_key_env="TEST_SELECTOR_KEY",
        selector_base_url="http://unused", agent_base_url="http://unused",
        selector_extra_body_json="", agent_extra_body_json="",
        selector_retries=0, selector_retry_delay=0, selector_model="test",
        agent_model="test", selector_max_tokens=1200,
        selector_validation_retries=2, selector_semaphore=semaphore,
        selector_design=design,
    )
    monkeypatch.setenv("TEST_SELECTOR_KEY", "dummy")
    monkeypatch.setattr(runner, "build_statebench_selector_packet", lambda *a: (
        {"instruction": "public only"}, [1, 0], {0: {"c0-e0"}, 1: {"c1-e0"}}, []))
    monkeypatch.setattr(runner.base, "make_client", lambda *a: SimpleNamespace(
        _client=SimpleNamespace(close=lambda: closed.append(True))))
    monkeypatch.setattr(runner.base, "log_response", lambda *a, **kw: None)

    def complete(*args, **kwargs):
        calls.append(json.loads(json.dumps(kwargs)))
        return _openai_response({
            "candidate_index": 0, "reason": "Public evidence supports this candidate",
            "supporting_event_ids": ["c1-e0" if len(calls) <= failures else "c0-e0"],
            "preserve_event_ids": [], "avoid_event_ids": [],
            "unresolved_issue_type": "NONE", "unresolved_issue": "",
        })

    monkeypatch.setattr(runner.base, "create_chat_completion_with_retry", complete)
    if failures == 3:
        with pytest.raises(RuntimeError, match="selector_validation_exhausted"):
            runner.select_with_statebench_eds_eca(ns, TASK, {}, {}, [], 1)
        assert len(calls) == 3
    else:
        selected, details, _ = runner.select_with_statebench_eds_eca(ns, TASK, {}, {}, [], 1)
        assert selected == 1
        assert not details["fallback"]
        assert len(details["validation_attempts"]) == failures + 1
    assert closed == [True]
    assert semaphore.acquire(blocking=False)
    semaphore.release()
    assert "SECRET GOLD" not in json.dumps(calls)
    assert calls[0]['tools'][0]['function']['parameters']['properties']['candidate_index']['enum'] == [0, 1]
    if design == 'atts-eca-pairwise-v1':
        assert 'exactly TWO' in calls[0]['messages'][0]['content']
        assert 'EDS-ECA evidence and credit' in calls[0]['messages'][0]['content']
    if failures:
        assert "unsupported_selection_evidence" in calls[1]["messages"][-1]["content"]
        assert calls[1]["stream"] is False


def _trace(update_result='{"success": true}'):
    return {
        "trajectory": (
            '[Agent Action]: Call tool `get_order` with arguments: {"order_id":"X-1"}\n'
            '[Environment Observation] (from get_order): {"status":"OPEN"}\n'
            '[Agent Action]: Call tool `update_order` with arguments: '
            '{"order_id":"X-1","status":"CLOSED"}\n'
            f'[Environment Observation] (from update_order): {update_result}\n'
            '[Agent Action]: Call tool `get_order` with arguments: {"order_id":"X-1"}\n'
            '[Environment Observation] (from get_order): {"status":"CLOSED"}\n'
            '[Agent Response]: Done.\n'
        )
    }


def test_explicit_nonstream_overrides_stream_environment(monkeypatch):
    import run_occubench_base as base
    monkeypatch.setenv("OCCUBENCH_STREAM", "1")
    calls = []
    expected = object()
    def create(**kwargs):
        calls.append(kwargs)
        return expected
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = base.create_chat_completion_with_retry(
        client, max_retries=0, retry_delay=0, stream=False, model="test", messages=[])
    assert result is expected
    assert calls[0]["stream"] is False


def _response(arguments):
    return {"tool_calls": [{"name": "select_and_credit_events", "arguments": arguments}]}


def _openai_response(arguments):
    function = SimpleNamespace(
        name="select_and_credit_events",
        arguments=json.dumps(arguments),
    )
    message = SimpleNamespace(tool_calls=[SimpleNamespace(function=function)])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_occubench_trace_uses_statebench_schema_aware_hierarchy():
    schemas = statebench_tool_schemas(CONFIG)
    ledger = analyze_stage(TASK, _trace(), schemas)
    assert ledger["event_count"] == 3
    assert ledger["hierarchy"]["atomic"] == 3
    assert ledger["raw_events"][0]["operation"] == "READ"
    assert ledger["raw_events"][1]["operation"] == "WRITE"
    assert ledger["raw_events"][1]["result"] == "ACKNOWLEDGED"


def test_public_target_state_argument_is_not_evaluator_metadata():
    from occubench_statebench_eds_eca import occubench_public_row
    from state_trace_effect_consensus import assert_public_payload
    stage = _trace()
    stage['trajectory'] = stage['trajectory'].replace('"status":"CLOSED"', '"target_state":"CLOSED"', 1)
    row = occubench_public_row(TASK, stage)
    assert row['conversation'][3]['tool_calls'][0]['arguments']['public_tool_target_state'] == 'CLOSED'
    analyze_stage(TASK, stage, statebench_tool_schemas(CONFIG))
    assert 'SECRET GOLD' not in json.dumps(row)
    with pytest.raises(ValueError):
        assert_public_payload({'target_state': 'SECRET GOLD'})
    with pytest.raises(ValueError):
        assert_public_payload({'gold_state': 'SECRET GOLD'})


def test_occubench_schema_adapter_marks_state_roles_without_gold_data():
    schemas = statebench_tool_schemas(CONFIG)
    descriptions = {row["name"]: row["description"] for row in schemas}
    assert "writes environment state" in descriptions["run_mental_rep"]
    assert "reads or retrieves" in descriptions["scan_inventory"]


def test_material_disagreement_uses_statebench_transaction_comparison():
    schemas = statebench_tool_schemas(CONFIG)
    same = compare_stages(TASK, _trace(), _trace(), schemas)
    changed = compare_stages(
        TASK,
        _trace(),
        _trace('{"success": false, "error": "invalid status"}'),
        schemas,
    )
    assert same["material_disagreement"] is False
    assert changed["material_disagreement"] is True


def test_selector_packet_and_credit_match_statebench_contract():
    schemas = statebench_tool_schemas(CONFIG)
    packet, order, allowed, candidates = build_selector_packet(
        TASK, _trace(), _trace('{"success": false}'), schemas, seed=9,
    )
    assert "SECRET GOLD" not in json.dumps(packet)
    assert sorted(order) == [0, 1]
    selected_shown = order.index(0)
    rejected_shown = 1 - selected_shown
    selected_id = sorted(allowed[selected_shown])[0]
    rejected_id = sorted(allowed[rejected_shown])[0]
    selected, details = resolve_selector(_response({
        "candidate_index": selected_shown,
        "reason": "The selected trace has stronger public state evidence.",
        "supporting_event_ids": [selected_id],
        "preserve_event_ids": [selected_id],
        "avoid_event_ids": [rejected_id],
        "unresolved_issue_type": "FAILURE_RECOVERY",
        "unresolved_issue": "Re-ground the failed update.",
    }), order, allowed)
    assert selected == 0
    assert details["fallback"] is False
    credit = build_credit_state(candidates, selected, details, schemas, None)
    assert credit["available"] is True
    assert credit["preserve_events"]
    assert credit["avoid_events"]
    assert "operation" in credit["preserve_events"][0]


def test_openai_tool_call_is_normalized_to_statebench_contract():
    schemas = statebench_tool_schemas(CONFIG)
    _, order, allowed, _ = build_selector_packet(
        TASK, _trace(), _trace('{"success": false}'), schemas, seed=9,
    )
    selected_shown = order.index(1)
    rejected_shown = 1 - selected_shown
    selected_id = sorted(allowed[selected_shown])[0]
    rejected_id = sorted(allowed[rejected_shown])[0]
    selected, details = resolve_selector(_openai_response({
        "candidate_index": selected_shown,
        "reason": "The proposal has stronger public evidence.",
        "supporting_event_ids": [selected_id],
        "preserve_event_ids": [selected_id],
        "avoid_event_ids": [rejected_id],
        "unresolved_issue_type": "NONE",
        "unresolved_issue": "",
    }), order, allowed)
    assert selected == 1
    assert details["fallback"] is False


def test_generation_guidance_is_the_statebench_frontier():
    schemas = statebench_tool_schemas(CONFIG)
    ledger, frontier, message = build_generation_guidance(
        TASK, _trace(), schemas, None, None, round_index=2,
    )
    assert ledger["schema_version"] == "state_trace_event_ledger_v1"
    assert frontier["schema_version"] == "state_trace_eds_ec_frontier_v1"
    assert "StateTrace Event-Driven Scaling round 2" in message
    assert "hidden requirements" in message
    assert "select_and_credit_events" in SELECT_INSTRUCTION
