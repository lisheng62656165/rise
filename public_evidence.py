"""EDS-ECA 的公开证据投影和防泄漏边界。

Every online event, disagreement, frontier, and selector packet passes through
this boundary. It retains only public conversation/tool material and rejects
已知的 evaluator、reward、gold-state 和 hidden-target 字段。
"""

from __future__ import annotations

from typing import Any, Mapping

from trace_ir import public_trajectory


FORBIDDEN_FIELDS = {
    "task_completion_pass", "state_requirements_met", "task_requirements_met",
    "ux_score", "state_diff", "gold_action", "gold_state", "expected_state",
    "evaluator_label", "reward", "milestones", "target_state",
}


# 函数作用：递归拒绝包含隐藏要求、gold state 或 evaluator 结果的在线 payload。
def assert_public_payload(value: Any, path: str = "root") -> None:
    """Fail immediately when an online payload contains evaluator-only data."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden field at {path}.{key}")
            assert_public_payload(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_public_payload(child, f"{path}[{index}]")


# 函数作用：从轨迹记录中仅投影在线方法允许使用的公开对话与元数据。
def public_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a stored trajectory to the public fields used by the method."""
    projection = {
        "task_key": str(row.get("task_key") or ""),
        "domain": str(row.get("domain") or ""),
        **public_trajectory(row),
    }
    assert_public_payload(projection)
    return projection
