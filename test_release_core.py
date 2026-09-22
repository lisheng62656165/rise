import json

import pytest

from artifact_io import read_rows
from public_evidence import assert_public_payload, public_projection


# 函数作用：验证轨迹读取器跳过汇总文件，并使用 task_key 建立索引。
def test_read_rows_skips_summaries_and_indexes_task_keys(tmp_path):
    (tmp_path / "task.json").write_text(
        json.dumps({"task_key": "statebench::demo::1", "conversation": []}),
        encoding="utf-8",
    )
    (tmp_path / "run_summary.json").write_text("{}", encoding="utf-8")
    rows = read_rows(tmp_path)
    assert list(rows) == ["statebench::demo::1"]


# 函数作用：验证公开信息守卫会拒绝嵌套出现的 evaluator 评分字段。
def test_public_guard_rejects_evaluator_fields():
    with pytest.raises(ValueError, match="forbidden field"):
        assert_public_payload({"nested": {"task_completion_pass": 1}})


# 函数作用：验证公开投影保留对话信息，并删除任务完成度结果。
def test_public_projection_keeps_public_conversation_only():
    row = {
        "task_key": "statebench::demo::1",
        "domain": "demo",
        "conversation": [{"role": "user", "content": "do the task"}],
        "task_completion_pass": 1,
    }
    projected = public_projection(row)
    assert projected["task_key"] == row["task_key"]
    assert "task_completion_pass" not in projected
