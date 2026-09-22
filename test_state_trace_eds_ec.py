from state_trace_eds_ec import (
    build_eds_ec_frontier,
    build_event_credit_state,
    derive_event_credit_decision,
    parse_listwise_choice,
    resolve_listwise_choice,
)


# 函数作用：构造 paper-aligned selector 的最小 index + analysis 回复。
def response(selected=1):
    return {"tool_calls": [{"name": "select_listwise_trajectory", "arguments": {
        "index": selected,
        "analysis": "candidate verifies the requested update",
    }}]}


# 函数作用：构造包含公开工具调用与结果的最小候选轨迹。
def trajectory(task_key, result="ok"):
    return {
        "task_key": task_key,
        "domain": "customer_support",
        "conversation": [
            {"role": "user", "content": "Fix order O-1."},
            {"role": "assistant", "tool_calls": [{"id": "c0", "function": {
                "name": "update_order", "arguments": '{"order_id":"O-1","status":"fixed"}',
            }}]},
            {"role": "tool", "tool_call_id": "c0", "content": result},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {
                "name": "get_order", "arguments": '{"order_id":"O-1"}',
            }}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
    }


# 函数作用：验证新 selector 能解析二选一结果并还原随机展示顺序。
def test_listwise_selector_maps_shuffled_candidate_order():
    selected, details = resolve_listwise_choice(response(1), [1, 0])
    assert selected == 0
    assert details["fallback"] is False
    assert details["decision"]["selected_original_index"] == 0


# 函数作用：验证 selector 不再直接生成事件信用字段。
def test_listwise_parser_requires_analysis():
    value = response()
    value["tool_calls"][0]["arguments"]["analysis"] = ""
    assert parse_listwise_choice(value) is None


# 函数作用：验证前向信用会删除可重放参数值和工具结果正文。
def test_credit_state_removes_literal_values_and_results():
    left = trajectory("statebench::customer_support::x", result="rejected secret")
    right = trajectory("statebench::customer_support::x", result="accepted secret")
    details = {"fallback": False, "decision": {
        "preserve_event_ids": ["call-0001"], "avoid_event_ids": ["call-0000"],
        "unresolved_issue_type": "VERIFICATION", "unresolved_issue": "verify the object",
        "reason": "Order O-1 and accepted secret look stronger",
    }}
    state = build_event_credit_state([left, right], 1, details, None)
    text = str(state)
    assert "O-1" not in text
    assert "accepted secret" not in text
    assert "rejected secret" not in text
    assert state["preserve_events"][0]["argument_keys"] == ["order_id"]


# 函数作用：验证事件信用由程序根据 winner 和公开轨迹差异确定，而不是模型直接提供。
def test_deterministic_credit_uses_selected_candidate():
    left = trajectory("statebench::customer_support::x", result="rejected")
    right = trajectory("statebench::customer_support::x", result="accepted")
    decision = derive_event_credit_decision([left, right], 1, None, "selected public evidence")
    assert decision["candidate_index"] == 1
    assert decision["credit_source"] == "deterministic_public_event_difference_v1"


# 函数作用：验证负面证据能在 selector 未分类时推断恢复问题类型。
def test_negative_credit_induces_typed_issue_when_selector_says_none():
    left = trajectory("statebench::customer_support::x", result="rejected")
    right = trajectory("statebench::customer_support::x", result="accepted")
    details = {"fallback": False, "decision": {
        "preserve_event_ids": ["call-0001"], "avoid_event_ids": ["call-0000"],
        "unresolved_issue_type": "NONE", "unresolved_issue": "", "reason": "selected is better",
    }}
    state = build_event_credit_state([left, right], 1, details, None)
    assert state["unresolved_issue_type"] == "FAILURE_RECOVERY"
    assert state["issue_source"] == "structural_negative_credit"
    assert "selected is better" not in str(state)


# 函数作用：验证 generation frontier 只暴露一个主要恢复干预点。
def test_frontier_exposes_one_active_intervention(monkeypatch):
    ledger = {
        "task_key": "statebench::customer_support::x", "domain": "customer_support",
        "transaction_groups": [], "recovery_windows": [],
        "recovery_decision_events": [], "raw_events": [],
    }
    monkeypatch.setattr("state_trace_eds_ec.analyze_public_events", lambda row, schemas: dict(ledger))
    monkeypatch.setattr("state_trace_eds_ec._hierarchical_events", lambda row: [])
    monkeypatch.setattr(
        "state_trace_eds_ec.fuse_event_ledgers",
        lambda ledgers, max_groups: {"high_value_events": [], "public_only": True, "outcome_used": False},
    )
    _, frontier = build_eds_ec_frontier(
        {"task_key": "x"}, None, None, 12,
        {"available": True, "unresolved_issue_type": "TASK_COVERAGE", "unresolved_issue": "retain explanation"},
    )
    assert frontier["intervention_count"] == 1
    assert frontier["active_intervention"]["source"] == "prior_event_credit"
    assert frontier["credit_policy"]["history"] == "latest validated comparison only"
