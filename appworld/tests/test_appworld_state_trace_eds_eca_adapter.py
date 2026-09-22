import json

from src.appworld_state_trace_eds_eca_adapter import (
    appworld_atomic_frontier,
    appworld_event_frontier,
    appworld_flat_selector_packet,
    appworld_preserve_only_frontier,
    appworld_public_disagreement,
    appworld_rds_only_frontier,
    appworld_selector_packet,
    compile_appworld_public_events,
    namespace_appworld_events,
)


def _trace(code: str, output: str):
    return [{"code": code, "execution_output": output}]


def test_api_documentation_words_do_not_create_failure_signal():
    output = json.dumps(
        {
            "description": "This API reports failure details and invalid input examples.",
            "parameters": [],
        }
    )
    events = compile_appworld_public_events(
        _trace("apis.api_docs.show_api_doc(app_name='mail', api_name='send')", output)
    )
    assert len(events) == 1
    assert events[0]["is_api_documentation"]
    assert not events[0]["failure_signal"]
    assert not events[0]["explicit_failure"]


def test_batched_failure_is_step_scoped_and_calls_keep_source_order():
    events = compile_appworld_public_events(
        _trace(
            "apis.mail.login(user='u'); apis.mail.send(to='x')",
            "Execution failed: Response status code is 401",
        )
    )
    assert [event["api"] for event in events] == ["login", "send"]
    assert all(event["batch_size"] == 2 for event in events)
    assert all(event["failure_signal"] for event in events)
    assert all(not event["explicit_failure"] for event in events)
    assert all(event["failure_scope"] == "step" for event in events)


def test_selector_packet_omits_verbose_documentation_but_keeps_public_summary():
    output = "failure " * 2000
    events = compile_appworld_public_events(
        _trace("apis.api_docs.show_api_doc(app_name='mail', api_name='send')", output)
    )
    packet = appworld_selector_packet("send mail", events, events)
    serialized = json.dumps(packet)
    assert "documentation text omitted" in serialized
    assert len(serialized) < 5000
    assert packet["candidate_A"]["summary"]["documentation_call_count"] == 1


def test_faithful_adapter_keeps_candidate_local_event_provenance():
    events = compile_appworld_public_events(_trace("apis.mail.send(to='x')", "ok"))
    left = namespace_appworld_events(events, "A")
    right = namespace_appworld_events(events, "B")
    assert left[0]["event_id"] == "A:s01c01"
    assert right[0]["event_id"] == "B:s01c01"
    assert left[0]["source_event_id"] == right[0]["source_event_id"]


def test_frontier_and_disagreement_are_public_and_structural():
    left = compile_appworld_public_events(_trace("apis.mail.send(to='x')", "ok"))
    right = compile_appworld_public_events(_trace("apis.mail.search(query='x')", "ok"))
    frontier = appworld_event_frontier(left)
    disagreement = appworld_public_disagreement(left, right)
    assert disagreement["material_disagreement"]
    assert frontier["public_only"] and not frontier["outcome_used"]
    assert "arguments" not in json.dumps(frontier)


def test_flat_event_ablation_has_no_event_hierarchy_or_aggregate_summary():
    events = compile_appworld_public_events(_trace("apis.mail.send(to='x')", "ok"))
    packet = appworld_flat_selector_packet("send mail", events, events)
    frontier = appworld_atomic_frontier(events)
    assert "summary" not in packet["candidate_A"]
    assert "atomic_events" in packet["candidate_A"]
    assert "event_hierarchy" not in frontier
    assert frontier["literal_replay"] is False


def test_no_rds_frontier_contains_only_nonfailure_events():
    good = compile_appworld_public_events(_trace("apis.mail.send(to='x')", "ok"))
    bad = compile_appworld_public_events(_trace("apis.mail.delete(id='x')", "Execution failed: HTTP 500"))
    frontier = appworld_preserve_only_frontier(good + bad)
    events = frontier["event_hierarchy"]["successful_events"]
    assert [event["api"] for event in events] == ["send"]
    assert frontier["active_intervention"] is None
    assert frontier["event_intervention"]["rds_avoidance"] is False


def test_no_preserve_frontier_contains_only_failure_events():
    good = compile_appworld_public_events(_trace("apis.mail.send(to='x')", "ok"))
    bad = compile_appworld_public_events(_trace("apis.mail.delete(id='x')", "Execution failed: HTTP 500"))
    frontier = appworld_rds_only_frontier(good + bad)
    events = frontier["event_hierarchy"]["recovery_risk_events"]
    assert [event["api"] for event in events] == ["delete"]
    assert frontier["active_intervention"]["issue_type"] == "FAILURE_RECOVERY"
    assert frontier["event_intervention"]["preserve_supported_prefix"] is False
