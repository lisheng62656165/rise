"""为 StateTrace-EDS-ECA 生成同配置 Vanilla A，并提供共享运行工具。

完整方法必须从未经事件指导和 selector 处理的 Vanilla 轨迹开始。本文件读取
StateBench 固定 70/30/50 划分，在 fresh 环境中使用同一 MIMO 配置执行任务，
保存公开 conversation、工具调用结果和基础机制统计。它还向主 runner 提供
split 读取、公开调用统计和原子 JSON 写入函数。这里不包含任何历史 policy、
Best-of-K、FRD repair controller 或事件选择逻辑；生成结果只作为后续 A anchor。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DOMAINS = ("customer_support", "shopping_assistant", "travel")
SPLIT_SIZES = {"train": 70, "dev": 30, "test": 50}


def bundled_benchmark_root() -> Path:
    """Return the benchmark bundled beside this release script."""
    return Path(__file__).resolve().parent / "benchmark"


# 函数作用：解析 Vanilla A 生成所需的 StateBench、split、任务与并发参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Vanilla anchors for StateTrace-EDS-ECA.")
    parser.add_argument("--statebench-root", type=Path, default=bundled_benchmark_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_SIZES), default="train")
    parser.add_argument("--cohort-file", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks-per-domain", type=int, default=0)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument('--seed', type=int, default=int(os.environ.get('MIMO_SEED', '53403')))
    return parser.parse_args()


# 函数作用：读取并校验 StateBench 70/30/50 固定划分中的任务 ID。
def read_split_ids(root: Path, domain: str, split: str) -> list[str]:
    path = root / "state_bench" / "domains" / domain / "splits" / "train_dev_70_30.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["splits"][split]
    expected = SPLIT_SIZES[split]
    if len(values) != expected or len(values) != len(set(values)):
        raise ValueError(f"invalid prescribed {split} split for {domain}")
    return values


# 函数作用：按 split、cohort 或显式 task 参数枚举 Vanilla A 任务。
def selected_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.cohort_file:
        payload = json.loads(args.cohort_file.read_text(encoding="utf-8"))
        result: list[tuple[str, str]] = []
        for value in payload.get("tasks") or ():
            domain, task_id = str(value).split("::", 1)
            if domain not in DOMAINS:
                raise ValueError(f"invalid cohort task: {value}")
            if (domain, task_id) not in result:
                result.append((domain, task_id))
        if not result:
            raise ValueError("cohort is empty")
        return result

    explicit = set()
    for value in args.task:
        domain, task_id = value.split("::", 1)
        if domain not in DOMAINS:
            raise ValueError(f"invalid --task: {value}")
        explicit.add((domain, task_id))
    result = []
    for domain in DOMAINS:
        task_ids = read_split_ids(args.statebench_root, domain, args.split)
        chosen = [task_id for task_id in task_ids if not explicit or (domain, task_id) in explicit]
        if args.tasks_per_domain:
            chosen = chosen[:args.tasks_per_domain]
        result.extend((domain, task_id) for task_id in chosen)
    if explicit and set(result) != explicit:
        raise ValueError(f"tasks not in split: {sorted(explicit - set(result))}")
    return result


# 函数作用：把对象序列化为键顺序稳定的 JSON，供公开调用比较。
def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


# 函数作用：根据公开工具返回判断该调用是否明确失败。
def tool_result_failed(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return bool(value.get("error") or value.get("exception") or value.get("success") is False)
    return False


# 函数作用：统计轨迹中公开可见的调用、失败和相同结果重复行为。
def public_mechanisms(conversation: list[dict[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for turn_index, message in enumerate(conversation):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            calls.append({"turn_index": turn_index, **call})
    failures = [index for index, call in enumerate(calls) if tool_result_failed(call.get("result"))]
    repeats = []
    for index, call in enumerate(calls):
        for prior_index in range(index - 1, -1, -1):
            prior = calls[prior_index]
            if (prior.get("name"), canonical_json(prior.get("arguments") or {})) != (
                call.get("name"), canonical_json(call.get("arguments") or {})
            ):
                continue
            if canonical_json(prior.get("result")) == canonical_json(call.get("result")):
                repeats.append({"prior_call_index": prior_index, "call_index": index})
            break
    return {
        "tool_call_count": len(calls),
        "tool_failure_count": len(failures),
        "first_tool_failure_index": failures[0] if failures else None,
        "exact_same_result_repeat_count": len(repeats),
        "exact_same_result_repeats": repeats,
    }


# 函数作用：在 fresh StateBench 环境中运行一个不带方法指导的 Vanilla A。
def run_one(args: argparse.Namespace, domain_name: str, task_id: str) -> dict[str, Any]:
    started = time.monotonic()
    sys.path.insert(0, str(args.statebench_root))
    from agents.mimo_agent import MiMoAgent
    from reliable_mimo_client import ReliableMiMoClient
    from state_bench.domain import get_domain_config
    from state_bench.env_loader import load_task_environment
    from state_bench.orchestrator import run_task
    from state_bench.paths import domain_tasks_dir
    from state_bench.schemas import TaskDefinition

    domain = get_domain_config(domain_name)
    task = TaskDefinition.load(domain_tasks_dir(domain_name) / f"{task_id}.json")
    env_data, _ = load_task_environment(domain, task)
    env = domain.environment_class(env_data.deep_copy(), now=task.now)
    system_prompt = domain.agent_system_prompt.format(now=task.now, user_id=task.user_id)
    agent_client = ReliableMiMoClient.from_env()
    simulator_client = ReliableMiMoClient.from_env()
    agent_client.seed = args.seed
    simulator_client.seed = args.seed
    agent = MiMoAgent(
        agent_client, system_prompt, domain.tool_schemas, env.tool_handlers,
        runtime_context=None,
    )
    try:
        trajectory = run_task(
            task, env_data, task.user_id, client=None, domain=domain, agent=agent,
            env=env, simulator_client=simulator_client,
            trajectory_metadata={
                "paper_line": "state_trace_eds_eca_vanilla_anchor",
                "phase": f"vanilla_anchor_{args.split}",
                "split": args.split,
                "domain": domain_name,
                "agent_model": agent_client.model_name,
                "agent_runtime_context_exposed": False,
                "gold_fields_exposed_to_agent": False,
                "inference_policy": "vanilla",
                "generation_seed": args.seed,
            },
        )
    finally:
        for client in (agent_client, simulator_client):
            close = getattr(getattr(client, "_client", None), "close", None)
            if close is not None:
                close()
    row = trajectory.to_dict()
    row.update(public_mechanisms(trajectory.conversation))
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    row["task_key"] = f"statebench::{domain_name}::{task_id}"
    return row


# 函数作用：使用临时文件原子写出 UTF-8 JSON，避免中断留下半文件。
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


# 函数作用：并发生成缺失的 Vanilla anchors，并记录成功、错误和运行配置。
def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.tasks_per_domain < 0:
        raise ValueError("--tasks-per-domain must be non-negative")
    if not ((os.environ.get("MIMO_API_KEY") or "").strip() or (os.environ.get("MIMO_API_KEYS_FILE") or "").strip()):
        raise ValueError("Set MIMO_API_KEY or MIMO_API_KEYS_FILE")
    jobs = selected_jobs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        job for job in jobs
        if not (args.output_dir / f"{job[0]}__{job[1]}.json").exists()
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, args, *job): job for job in pending}
        for future in as_completed(futures):
            domain, task_id = futures[future]
            output_path = args.output_dir / f"{domain}__{task_id}.json"
            try:
                row = future.result()
                write_json(output_path, row)
                result = {"task_key": row["task_key"], "status": "ok"}
            except BaseException as error:
                result = {
                    "task_key": f"statebench::{domain}::{task_id}",
                    "status": "error",
                    "error": str(error),
                    "traceback": traceback.format_exc()[-8000:],
                }
                write_json(args.output_dir / f"{domain}__{task_id}.error.json", result)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(args.output_dir / "run_summary.json", {
        "method": "Vanilla anchor for StateTrace-EDS-ECA",
        "split": args.split,
        "requested_tasks": len(jobs),
        "pending_tasks": len(pending),
        "workers": args.workers,
        "ok": sum(row["status"] == "ok" for row in results),
        "errors": sum(row["status"] == "error" for row in results),
        "results": results,
    })
    if any(row['status'] == 'error' for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
