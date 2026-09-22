"""EDS-ECA 使用的按 task key 加载 JSON 实验产物模块。

The production loop receives Vanilla A trajectories as immutable input
artifacts. This module scans one directory, ignores reports/errors, validates
检查每个 JSON 是否包含唯一 task key，然后返回 runner 使用的映射。它不读取
评分、不选择候选，只负责稳定加载 anchor，避免引入历史实验代码。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IGNORED_NAMES = {"summary.json", "score_summary.json", "report.json", "report_scores.json"}


# 函数作用：读取目录中的任务 JSON，跳过汇总文件，并按 task_key 建立索引。
def read_rows(directory: Path) -> dict[str, dict[str, Any]]:
    """Load one completed trajectory per task key from an artifact directory."""
    rows = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in IGNORED_NAMES or path.name.endswith(
            (".error.json", "summary.json", "report.json", "report_scores.json")
        ):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        key = str(row.get("task_key") or "")
        if not key or key in rows:
            raise ValueError(f"invalid or duplicate task_key in {path}")
        rows[key] = row
    return rows
