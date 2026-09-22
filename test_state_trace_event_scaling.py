import copy

from state_trace_event_scaling import analyze_public_events, compare_public_trajectories, fuse_event_ledgers


# 函数作用：构造事件分层测试使用的最小公开轨迹。
def trajectory():
    return {
        "task_key": "statebench::customer_support::demo",
        "domain": "customer_support",
        "task_completion_pass": 1,
        "state_requirements_met": 1,
        "conversation": [
            {"role": "user", "content": "Please fix order O-1."},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read_order", "arguments": '{"order_id":"O-1"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'},
            {"role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "update_order", "arguments": '{"order_id":"O-1","status":"fixed"}'}}]},
            {"role": "tool", "tool_call_id": "c2", "content": '{"error":"rejected"}'},
            {"role": "assistant", "tool_calls": [{"id": "c3", "function": {"name": "update_order", "arguments": '{"order_id":"O-1","status":"fixed"}'}}]},
            {"role": "tool", "tool_call_id": "c3", "content": '{"error":"rejected"}'},
        ],
    }


# 函数作用：验证公开轨迹会形成原子、事务、恢复窗口和恢复决策层次。
def test_event_ledger_has_hierarchical_public_events():
    ledger = analyze_public_events(trajectory())
    assert ledger["event_count"] == 3
    assert ledger["transaction_group_count"] >= 2
    assert ledger["recovery_window_count"] >= 1
    assert any("failure_boundary" in group["risk_tags"] for group in ledger["transaction_groups"])
    assert ledger["public_only"] is True
    assert ledger["outcome_used"] is False


# 函数作用：验证跨轮融合容量受限，且不会携带 evaluator 结果字段。
def test_fusion_is_bounded_and_drops_outcome_fields():
    ledger = analyze_public_events(trajectory())
    fused = fuse_event_ledgers([ledger, copy.deepcopy(ledger)], max_groups=2)
    assert len(fused["high_value_events"]) <= 2
    assert fused["public_only"] is True
    assert fused["outcome_used"] is False
    text = str(fused)
    assert "task_completion_pass" not in text
    assert "state_requirements_met" not in text


# 函数作用：验证融合 frontier 会脱敏对象标识和参数字面值。
def test_fusion_redacts_replayable_public_values():
    ledger = analyze_public_events(trajectory())
    right = copy.deepcopy(trajectory())
    right["conversation"][5]["tool_calls"][0]["function"]["arguments"] = (
        '{"order_id":"O-999","status":"other"}'
    )
    disagreement = compare_public_trajectories(trajectory(), right)
    ledger["cross_round_disagreement"] = disagreement
    fused = fuse_event_ledgers([ledger], max_groups=8)
    text = str(fused)
    assert "O-1" not in text
    assert "fixed" not in text
    assert "argument_field_kinds" in text
    assert "entity_reference_count" in text


# 函数作用：验证恢复决策层能够区分多种公开 RDS 风险类型。
def test_recovery_decision_layer_types_public_risks():
    ledger = analyze_public_events(trajectory())
    decisions = ledger["recovery_decision_events"]
    kinds = {item["kind"] for item in decisions}
    assert "recovery_entry" in kinds
    assert "repeated_failed_mutation" in kinds
    assert ledger["hierarchy"]["recovery_decision"] == len(decisions)
    assert all(item["risk"] != "strict_frd" for item in decisions)


# 函数作用：验证两轨迹事务分歧对称，且不包含 winner 或评分。
def test_cross_round_disagreement_is_symmetric_and_outcome_free():
    left = trajectory()
    right = copy.deepcopy(left)
    right["conversation"][5]["tool_calls"][0]["function"]["arguments"] = (
        '{"order_id":"O-1","status":"pending"}'
    )
    packet = compare_public_trajectories(left, right)
    assert packet["material_disagreement"] is True
    assert packet["candidate_identity_removed"] is True
    assert packet["candidate_winner_included"] is False
    assert any(item["kind"] == "TRANSACTION_CONTENT" for item in packet["conflict_components"])
    assert "task_completion_pass" not in str(packet)


# 函数作用：构造带层级、状态和风险标签的最小融合事件。
def _fusion_item(level, status="MIXED", tags=(), event_id="e1"):
    return {
        "level": level,
        "status": status,
        "risk_tags": list(tags),
        "entity_refs": ["O-1"],
        "events": [{"event_id": event_id, "event_class": "mutation"}],
        "member_event_ids": [event_id],
    }


# 函数作用：验证融合排序优先恢复证据，其次是分歧和普通成功事件。
def test_fusion_prioritizes_recovery_evidence_over_disagreement_and_success():
    ledger = {
        "transaction_groups": [
            _fusion_item("transaction", "SUCCESS", event_id="success"),
            _fusion_item("transaction", "FAILURE", ["failure_boundary"], "failure"),
        ],
        "recovery_windows": [],
        "recovery_decision_events": [
            _fusion_item("recovery_decision", tags=["candidate_recovery_error"], event_id="recovery")
        ],
        "cross_round_disagreement": {
            "conflict_components": [{"component_id": "conflict"}],
        },
    }
    fused = fuse_event_ledgers([ledger], max_groups=4)
    categories = [item["fusion_category"] for item in fused["high_value_events"]]
    assert categories == [
        "failure_boundary",
        "recovery_decision",
        "material_disagreement",
        "verified_effect",
    ]


# 函数作用：验证事件价值相同时优先保留更新轮次的公开证据。
def test_fusion_prefers_recent_evidence_on_ties():
    old = {
        "transaction_groups": [_fusion_item("transaction", "FAILURE", ["failure_boundary"], "old")],
        "recovery_windows": [], "recovery_decision_events": [],
    }
    recent = {
        "transaction_groups": [_fusion_item("transaction", "FAILURE", ["failure_boundary"], "recent")],
        "recovery_windows": [], "recovery_decision_events": [],
    }
    fused = fuse_event_ledgers([old, recent], max_groups=2)
    assert [item["source_round"] for item in fused["high_value_events"]] == [1, 0]


# 函数作用：验证事务分歧不能占满 frontier 配额并挤掉恢复证据。
def test_disagreement_quota_cannot_fill_the_fusion_budget():
    ledger = {
        "transaction_groups": [], "recovery_windows": [], "recovery_decision_events": [],
        "cross_round_disagreement": {
            "conflict_components": [{"component_id": f"c{index}"} for index in range(10)],
        },
    }
    fused = fuse_event_ledgers([ledger], max_groups=8)
    assert len(fused["high_value_events"]) == 4
    assert fused["category_counts"] == {"material_disagreement": 4}
