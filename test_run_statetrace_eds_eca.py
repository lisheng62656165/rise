import pytest

from run_statetrace_eds_eca_paper import ROUND_NAMES, build_selector_packet, validate_anchor


# 函数作用：验证正式算法固定执行 A、B、C、D 四轮。
def test_round_names_cover_the_supported_iterative_budget():
    assert ROUND_NAMES == tuple("ABCD")


# 函数作用：验证 EDS-ECA 只接受 Vanilla 轨迹作为初始 A。
def test_primary_anchor_must_be_vanilla():
    validate_anchor({"task_key": "x", "inference_policy": "vanilla"}, "vanilla")
    with pytest.raises(ValueError, match="must be a Vanilla"):
        validate_anchor({"task_key": "x", "inference_policy": "other"}, "vanilla")
    with pytest.raises(ValueError, match="must be a Vanilla"):
        validate_anchor({
            "task_key": "x",
            "inference_policy": "vanilla",
            "state_trace_arc_selected_source": "anchor",
        }, "vanilla")


# 函数作用：验证正式 selector packet 自动删除离线评分，并保留随机展示索引。
def test_selector_packet_contains_public_evidence_only():
    candidate = {
        "task_key": "statebench::demo::1", "domain": "demo",
        "conversation": [{"role": "user", "content": "do the task"}],
        "task_completion_pass": 1, "ux_score": 5,
    }
    packet = build_selector_packet([candidate, candidate], None, [1, 0])
    assert [item["candidate_index"] for item in packet["candidates"]] == [0, 1]
    assert "task_completion_pass" not in str(packet)
    assert "ux_score" not in str(packet)
    assert packet["public_only"] is True
