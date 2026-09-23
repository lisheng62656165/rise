"""在 rollout 完成后离线评分，生成过程不接触 gold data。

本文件位于在线 StateTrace-EDS-ECA 调用图之外。
在 A/B'/C'/D' 全部生成和选择结束后，脚本检查运行边界标记，为 worker/domain
构造 task 和可选 UX judge，调用官方 score_one，检查指标是否完整写入并生成
独立 summary。原始轨迹不会改写，评分不会反馈到事件分析、selector、信用或
incumbent 更新。
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DOMAINS = ("customer_support", "shopping_assistant", "travel")
REQUIRED_SCORE_FIELDS = (
    "task_completion_pass",
    "state_requirements_met",
    "task_requirements_met",
    "ux_score",
)


def bundled_benchmark_root() -> Path:
    """Return the benchmark bundled beside this release script."""
    return Path(__file__).resolve().parent / "benchmark"


# 函数作用：解析离线 MIMO 评分阶段的输入、输出和 UX 配置。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statebench-root", type=Path, default=bundled_benchmark_root())
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default=None,
                        help="Recorded for reproducibility; task files are taken from input-dir")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--with-ux", action="store_true")
    parser.add_argument("--ux-only", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument('--judge-seed', type=int, default=88002)
    return parser.parse_args()


# 函数作用：读取待评分目录中的原始任务轨迹并按任务键索引。
def load_raw_trajectories(input_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"run_summary.json", "score_summary.json"} or path.name.endswith(".error.json"):
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_runtime_boundary(row, path)
        rows.append((path, row))
    return rows


# 函数作用：确认评分输入仍满足运行期公开信息边界。
def validate_runtime_boundary(row: dict[str, Any], path: Path) -> None:
    domain = row.get("domain")
    if domain not in DOMAINS:
        raise ValueError(f"{path}: invalid domain {domain!r}")
    if not row.get("task_id") or not isinstance(row.get("conversation"), list):
        raise ValueError(f"{path}: missing task_id or conversation")
    if row.get("agent_runtime_context_exposed") is not False:
        raise ValueError(f"{path}: runtime context exposure was not explicitly disabled")
    if row.get("gold_fields_exposed_to_agent") is not False:
        raise ValueError(f"{path}: gold-field exposure was not explicitly disabled")


# 函数作用：将单任务评分或汇总结果写入 JSON 文件。
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


# 函数作用：判断已有评分文件是否缺少本次要求的完成度或 UX 字段。
def missing_score_fields(path: Path, *, with_ux: bool, ux_only: bool = False) -> list[str]:
    if not path.exists():
        return ["score_file"]
    row = json.loads(path.read_text(encoding="utf-8"))
    fields = ("ux_score",) if ux_only else REQUIRED_SCORE_FIELDS if with_ux else REQUIRED_SCORE_FIELDS[:-1]
    return [field for field in fields if row.get(field) is None]


# 函数作用：并发调度所有缺失的离线评分任务并生成汇总。
def main() -> None:
    args = parse_args()
    if args.with_ux and args.ux_only:
        raise ValueError("--with-ux and --ux-only are mutually exclusive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not ((os.environ.get("MIMO_API_KEY") or "").strip() or (os.environ.get("MIMO_API_KEYS_FILE") or "").strip()):
        raise ValueError("Set MIMO_API_KEY or MIMO_API_KEYS_FILE")

    sys.path.insert(0, str(args.statebench_root))
    from reliable_mimo_client import ReliableMiMoClient as MiMoClient
    from state_bench.domain import get_domain_config
    from state_bench.paths import domain_tasks_dir
    from state_bench.scoring import TaskRequirementsJudge, UXQualityJudge
    from state_bench.scripts.score import score_one

    # 阶段 1：只读取已完成的 raw artifact，并检查元数据是否明确记录了
    # 无泄漏运行边界。
    rows = load_raw_trajectories(args.input_dir)
    if not rows:
        raise ValueError('No trajectories to score')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thread_state = threading.local()

    # 函数作用：根据 StateBench 领域构造对应的任务完成度评审器。
    def judges_for(domain_name: str) -> tuple[Any, Any, Any]:
        """Keep one key-pooled client per scoring worker thread."""
        if not hasattr(thread_state, "client"):
            thread_state.client = MiMoClient.from_env()
            thread_state.client.seed = args.judge_seed
            thread_state.judges = {}
        if domain_name not in thread_state.judges:
            domain = get_domain_config(domain_name)
            task_judge = TaskRequirementsJudge(
                client=thread_state.client,
                prompts_dir=domain.prompts_dir,
                system_prompt=domain.judge_system_prompt,
                reasoning_effort=args.reasoning_effort,
            )
            ux_judge = None
            if args.with_ux or args.ux_only:
                ux_judge = UXQualityJudge(
                    client=thread_state.client,
                    prompts_dir=domain.prompts_dir,
                    system_prompt=domain.judge_system_prompt,
                    reasoning_effort=args.reasoning_effort,
                )
            thread_state.judges[domain_name] = (task_judge, ux_judge)
        task_judge, ux_judge = thread_state.judges[domain_name]
        return thread_state.client, task_judge, ux_judge

    results: list[dict[str, Any]] = []
    # 阶段 2：并发评分。每个 worker 线程拥有自己的 API client，并缓存 domain
    # judge；score 文件按任务保存，可断点续跑而不重复评分已完成轨迹。
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        task_roots = {
            domain_name: args.statebench_root / "state_bench" / "domains" / domain_name / "tasks"
            for domain_name in DOMAINS
        }
        # 函数作用：对一条完整轨迹计算 Task Completion，并按配置补充 UX 评分。
        def score_row(input_path: Path, row: dict[str, Any], output_path: Path) -> dict[str, Any]:
            client, task_judge, ux_judge = judges_for(row["domain"])
            existing_scores = {}
            if args.ux_only and output_path.exists():
                previous = json.loads(output_path.read_text(encoding='utf-8'))
                existing_scores = {k: v for k, v in previous.items()
                                   if k.startswith(('task_requirements_', 'state_requirements_'))
                                   or k == 'task_completion_pass'}
            call_args = [
                input_path,
                task_roots[row["domain"]]
                if task_roots[row["domain"]].exists()
                else domain_tasks_dir(row["domain"]),
                task_judge,
                ux_judge,
                output_path,
                None,
                row["domain"],
                not args.ux_only,
            ]
            if "judge_metadata" in inspect.signature(score_one).parameters:
                call_args.append({
                    "judge_model": client.model_name,
                    "judge_client_name": type(client).__name__,
                    "judge_reasoning_effort": args.reasoning_effort,
                    "scoring_boundary": "offline_after_trajectory",
                })
            result = score_one(*call_args)
            if result.get('status') == 'OK':
                scored = json.loads(output_path.read_text(encoding='utf-8'))
                scored.update(existing_scores)
                scored.update(judge_model=client.model_name, judge_seed=args.judge_seed,
                              judge_base_url=str(client._client.base_url),
                              scoring_boundary='offline_after_trajectory')
                write_json(output_path, scored)
            return result

        for input_path, row in rows:
            domain_name = row["domain"]
            output_path = args.output_dir / input_path.name
            if not missing_score_fields(output_path, with_ux=args.with_ux, ux_only=args.ux_only):
                results.append({"task_key": row["task_key"], "status": "existing"})
                continue
            future = executor.submit(score_row, input_path, row, output_path)
            futures[future] = (row["task_key"], output_path)
        for future in as_completed(futures):
            task_key, output_path = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'status': 'ERR', 'error': type(error).__name__}
            result["task_key"] = task_key
            if result.get("status") == "OK":
                missing = missing_score_fields(
                    output_path, with_ux=args.with_ux, ux_only=args.ux_only
                )
                if missing:
                    output_path.unlink(missing_ok=True)
                    result = {
                        "task_key": task_key,
                        "task_id": result.get("task_id"),
                        "status": "ERR",
                        "error": f"incomplete score fields: {missing}",
                    }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    # 阶段 3：只记录运行完成情况。指标汇总和 Vanilla/方法 paired 统计由后续
    # analysis 脚本完成，避免和单任务评分职责混在一起。
    summary = {
        "input_trajectories": len(rows),
        "scored": sum(row["status"] == "OK" for row in results),
        "existing": sum(row["status"] == "existing" for row in results),
        "errors": sum(row["status"] not in {"OK", "existing"} for row in results),
        "workers": args.workers,
        "with_ux": args.with_ux,
        "results": sorted(results, key=lambda row: row["task_key"]),
    }
    write_json(args.output_dir / "score_summary.json", summary)
    if summary['errors']:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
