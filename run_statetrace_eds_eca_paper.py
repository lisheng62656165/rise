"""运行 StateTrace-EDS-ECA 的 StateBench 正式实现。

Vanilla A 是唯一初始轨迹。每轮从 accepted incumbent 的公开事件 frontier 在
fresh environment 中生成一个 proposal；存在 material disagreement 时，list-wise
selector 二选一；程序再从公开事件差异生成 ECA 信用，指导下一轮。最终输出可能
来自 A/B'/C'/D'，而不是强制输出最后一条 proposal。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from artifact_io import read_rows
from phase0_runner import DOMAINS, public_mechanisms, read_split_ids, selected_jobs, write_json
from public_evidence import assert_public_payload, public_projection
from state_trace_eds_ec import (
    LISTWISE_SELECT_INSTRUCTION,
    build_eds_ec_frontier,
    build_event_credit_state,
    derive_event_credit_decision,
    listwise_selector_tool,
    resolve_listwise_choice,
)
from state_trace_event_scaling import analyze_public_events, compare_public_trajectories, event_scaling_message
from state_trace_pool_select import build_packet as build_public_packet


ROUND_NAMES = tuple("ABCD")


def bundled_benchmark_root() -> Path:
    """Return the benchmark bundled beside this release script."""
    return Path(__file__).resolve().parent / "benchmark"


def validate_anchor(row: Mapping[str, Any], anchor_kind: str = "vanilla") -> None:
    """确保初始 A 是原始 Vanilla，而非历史 selector 或其他方法的输出。"""
    if anchor_kind != "vanilla":
        raise ValueError("StateTrace-EDS-ECA only supports a Vanilla anchor")
    if row.get("inference_policy") != "vanilla" or row.get("state_trace_arc_selected_source") is not None:
        raise ValueError(f"A must be a Vanilla trajectory: {row.get('task_key')}")


def build_selector_packet(candidates: list[Mapping[str, Any]], schemas: Any, order: list[int]) -> dict[str, Any]:
    """构造随机顺序、仅含公开轨迹和事件摘要的二选一 selector 输入。"""
    if sorted(order) != list(range(len(candidates))):
        raise ValueError("order must be a permutation of candidate indices")
    packet = build_public_packet(candidates, schemas, order)
    packet['public_transaction_disagreement'] = [compare_public_trajectories(*candidates, schemas)]
    assert_public_payload(packet)
    return packet


def parse_args() -> argparse.Namespace:
    """解析正式算法所需参数；不暴露历史方法、gate 或 ablation 开关。"""
    parser = argparse.ArgumentParser(description="Run paper-aligned StateTrace-EDS-ECA on StateBench")
    parser.add_argument("--statebench-root", type=Path, default=bundled_benchmark_root())
    parser.add_argument("--anchor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--cohort-file", type=Path)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks-per-domain", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True, help="B' seed; C'/D' use seed+1/seed+2")
    parser.add_argument("--selector-seed", type=int, default=77101)
    parser.add_argument("--max-event-groups", type=int, default=12)
    parser.add_argument("--round-count", type=int, choices=(4,), default=4)
    parser.add_argument("--selector-retries", type=int, default=2)
    return parser.parse_args()


def close_client(client: Any) -> None:
    """关闭每个任务拥有的 HTTP transport，避免并发下积累半关闭连接。"""
    transport = getattr(client, "_client", None)
    close = getattr(transport, "close", None)
    if close is not None:
        close()


def main() -> None:
    """加载 Vanilla anchors，并发执行三轮生成、选择和事件信用更新。"""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.tasks_per_domain < 0:
        raise ValueError("--tasks-per-domain must be non-negative")
    if not ((os.environ.get("MIMO_API_KEY") or "").strip() or (os.environ.get("MIMO_API_KEYS_FILE") or "").strip()):
        raise ValueError("Set MIMO_API_KEY or MIMO_API_KEYS_FILE")
    sys.path.insert(0, str(args.statebench_root))

    from agents.mimo_agent import MiMoAgent
    from reliable_mimo_client import ReliableMiMoClient
    from state_bench.domain import get_domain_config
    from state_bench.env_loader import load_task_environment
    from state_bench.orchestrator import run_task
    from state_bench.paths import domain_tasks_dir
    from state_bench.schemas import TaskDefinition

    anchors = read_rows(args.anchor_dir)
    jobs = selected_jobs(args)
    if args.tasks_per_domain:
        jobs = [
            job for domain_name in DOMAINS
            for job in [item for item in jobs if item[0] == domain_name][:args.tasks_per_domain]
        ]
    split_jobs = [(domain, task_id) for domain in DOMAINS for task_id in read_split_ids(args.statebench_root, domain, args.split)]
    ordinals = {job: index for index, job in enumerate(split_jobs)}
    keys = {f"statebench::{domain}::{task_id}" for domain, task_id in jobs}
    missing = keys - set(anchors)
    if missing:
        raise ValueError({"missing_anchor": sorted(missing)})
    for key in keys:
        validate_anchor(anchors[key])

    domains = {name: get_domain_config(name) for name, _ in jobs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "stage_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def generate_proposal(domain: Any, task: Any, system_prompt: str, injected: str, proposal_seed: int, metadata: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        """在全新 StateBench 环境中生成一条完整 proposal 轨迹。"""
        env_data, _ = load_task_environment(domain, task)
        env = domain.environment_class(env_data.deep_copy(), now=task.now)
        client = ReliableMiMoClient.from_env()
        simulator_client = ReliableMiMoClient.from_env()
        client.seed = proposal_seed
        simulator_client.seed = proposal_seed
        agent = MiMoAgent(client, system_prompt, domain.tool_schemas, env.tool_handlers, runtime_context=None)
        original_prepare = agent.prepare_conversation
        agent.prepare_conversation = lambda conversation, prepare=original_prepare, message=injected: agent.inject_system_message(prepare(conversation), message, before_last_user=True)
        started = time.monotonic()
        try:
            trajectory = run_task(
                task, env_data, task.user_id, client=None, domain=domain, agent=agent,
                env=env, simulator_client=simulator_client,
                trajectory_metadata=dict(metadata),
            )
        finally:
            close_client(client)
            close_client(simulator_client)
        row = trajectory.to_dict()
        row['agent_model'] = client.model_name
        row['generation_seed'] = proposal_seed
        row.update(public_mechanisms(trajectory.conversation))
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return row, int(getattr(agent.token_usage, "total_tokens", 0) or 0)

    def select_pair(domain: Any, system_prompt: str, candidates: list[Mapping[str, Any]], order: list[int], decision_seed: int) -> tuple[int, dict[str, Any], int]:
        """调用模型做二选一；格式错误有界重试，耗尽后保留 proposal 等待续跑。"""
        packet = build_selector_packet(candidates, domain.tool_schemas, order)
        tool = listwise_selector_tool(2)
        client = ReliableMiMoClient.from_env()
        client.seed = decision_seed
        selector = MiMoAgent(client, system_prompt, [tool], {}, runtime_context=None)
        client.max_tokens = int(os.environ.get('MIMO_SELECTOR_MAX_TOKENS', client.max_tokens))
        try:
            for attempt in range(args.selector_retries + 1):
                try:
                    response = selector.generate_next_turn(
                        system_prompt=f"{system_prompt}\n\n{LISTWISE_SELECT_INSTRUCTION}",
                        conversation=[{"role": "user", "content": json.dumps(packet, ensure_ascii=False)}],
                        tools=[tool],
                    )
                except (ValueError, TypeError):
                    if attempt == args.selector_retries:
                        raise
                    continue
                selected, details = resolve_listwise_choice(response, order)
                if not details.get('fallback'):
                    details['attempts'] = attempt + 1
                    break
            else:
                raise ValueError('invalid selector after bounded retries; resume the saved proposal')
        finally:
            close_client(client)
        return selected, details, int(getattr(selector.token_usage, "total_tokens", 0) or 0)

    def run_one(domain_name: str, task_id: str) -> dict[str, Any]:
        """对一个 task 执行 A→B′→C′→D′，并持续维护 accepted incumbent。"""
        key = f"statebench::{domain_name}::{task_id}"
        output_path = args.output_dir / f"{domain_name}__{task_id}.json"
        if output_path.exists():
            row = json.loads(output_path.read_text(encoding="utf-8"))
            return {"task_key": key, "status": "existing", "final_origin": row.get("state_trace_eds_eca_final_origin")}
        domain = domains[domain_name]
        task = TaskDefinition.load(domain_tasks_dir(domain_name) / f"{task_id}.json")
        system_prompt = domain.agent_system_prompt.format(now=task.now, user_id=task.user_id)
        incumbent = copy.deepcopy(anchors[key])
        incumbent_origin = "A"
        prior_disagreement: dict[str, Any] | None = None
        credit_state: dict[str, Any] = {}
        stages: list[dict[str, Any]] = []
        proposal_tokens_total = selector_tokens_total = 0
        started = time.monotonic()

        for stage_index in range(1, args.round_count):
            accepted_label = ROUND_NAMES[stage_index]
            proposal_label = f"{accepted_label}'"
            checkpoint_path = checkpoint_dir / f"{domain_name}__{task_id}__{accepted_label}.json"
            if checkpoint_path.exists():
                stage = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                stages.append(stage)
                incumbent = stage["accepted_trajectory"]
                incumbent_origin = str(stage["accepted_origin"])
                prior_disagreement = stage.get("accepted_public_disagreement")
                credit_state = dict(stage.get("event_credit_feedback") or {})
                proposal_tokens_total += int(stage.get("proposal_tokens", 0))
                selector_tokens_total += int(stage.get("selector_tokens", 0))
                continue

            incumbent_ledger, frontier = build_eds_ec_frontier(
                incumbent, domain.tool_schemas, prior_disagreement,
                args.max_event_groups, credit_state,
            )
            proposal_seed = args.seed + stage_index - 1
            proposal_path = checkpoint_dir / f'{domain_name}__{task_id}__{accepted_label}.proposal.json'
            if proposal_path.exists():
                saved_proposal = json.loads(proposal_path.read_text(encoding='utf-8'))
                proposal, proposal_tokens = saved_proposal['trajectory'], saved_proposal['tokens']
            else:
                proposal, proposal_tokens = generate_proposal(
                domain, task, system_prompt,
                event_scaling_message(frontier, stage_index + 1), proposal_seed,
                {
                    "method": "StateTrace-EDS-ECA", "split": args.split,
                    "task_key": key, "proposal_label": proposal_label,
                    "proposal_seed": proposal_seed, "event_scaling_public_only": True,
                    "online_gold_or_evaluator_used": False,
                    "domain": domain_name,
                    "agent_runtime_context_exposed": False,
                    "gold_fields_exposed_to_agent": False,
                },
                )
                proposal['task_key'] = key
                write_json(proposal_path, {'trajectory': proposal, 'tokens': proposal_tokens})
            proposal["task_key"] = key
            proposal_tokens_total += proposal_tokens
            disagreement = compare_public_trajectories(incumbent, proposal, domain.tool_schemas)
            candidates = [incumbent, proposal]
            selector_tokens = 0

            if disagreement.get("material_disagreement"):
                decision_seed = args.selector_seed + ordinals[(domain_name, task_id)] * 3 + stage_index
                order = [0, 1]
                random.Random(decision_seed).shuffle(order)
                selected, selector_details, selector_tokens = select_pair(domain, system_prompt, candidates, order, decision_seed)
                if not selector_details.get("fallback"):
                    model_decision = dict(selector_details.get("decision") or {})
                    credit_decision = derive_event_credit_decision(
                        candidates, selected, domain.tool_schemas, str(model_decision.get("reason") or ""),
                    )
                    credit_decision.update({"selected_original_index": selected, "rejected_original_indices": [1 - selected]})
                    selector_details = {
                        **selector_details, "decision": credit_decision,
                        "selection_decision": model_decision,
                        "selector_protocol": "listwise_plus_deterministic_event_credit_v1",
                    }
            else:
                selected = 0
                selector_details = {
                    "fallback": True, "decision": None,
                    "fallback_reason": "no_material_public_transaction_disagreement",
                    "selector_protocol": "listwise_plus_deterministic_event_credit_v1",
                }
            selector_tokens_total += selector_tokens
            accepted_proposal = selected == 1 and not selector_details.get("fallback")
            credit_state = build_event_credit_state(candidates, selected, selector_details, domain.tool_schemas, credit_state)
            if accepted_proposal:
                incumbent = copy.deepcopy(proposal)
                incumbent_origin = proposal_label
                prior_disagreement = disagreement
            else:
                incumbent = copy.deepcopy(incumbent)
                prior_disagreement = None
            stage = {
                "accepted_stage": accepted_label, "proposal_label": proposal_label,
                "accepted_origin": incumbent_origin, "proposal_accepted": accepted_proposal,
                "incumbent_event_ledger": incumbent_ledger,
                "generation_frontier": frontier, "public_disagreement": disagreement,
                "accepted_public_disagreement": prior_disagreement,
                "selector": selector_details, "event_credit_feedback": credit_state,
                "proposal_tokens": proposal_tokens, "selector_tokens": selector_tokens,
                "proposal_trajectory": proposal, "accepted_trajectory": copy.deepcopy(incumbent),
                "public_only": True, "online_outcome_used": False,
            }
            write_json(checkpoint_path, stage)
            stages.append(stage)

        output = copy.deepcopy(incumbent)
        output.update({
            "state_trace_eds_eca_final_round": ROUND_NAMES[args.round_count - 1],
            "state_trace_eds_eca_final_origin": incumbent_origin,
            "state_trace_eds_eca_stages": stages,
            "state_trace_eds_eca_proposal_tokens": proposal_tokens_total,
            "state_trace_eds_eca_selector_tokens": selector_tokens_total,
            "state_trace_eds_eca_elapsed_seconds": round(time.monotonic() - started, 3),
            "method": "StateTrace-EDS-ECA", "online_gold_or_evaluator_used": False,
        })
        write_json(output_path, output)
        return {"task_key": key, "status": "ok", "final_origin": incumbent_origin}

    pending = [job for job in jobs if not (args.output_dir / f"{job[0]}__{job[1]}.json").exists()]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, *job): job for job in pending}
        for future in as_completed(futures):
            domain, task_id = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                result = {
                    "task_key": f"statebench::{domain}::{task_id}", "status": "error",
                    "error": str(error), "traceback": traceback.format_exc()[-8000:],
                }
                write_json(args.output_dir / f"{domain}__{task_id}.error.json", result)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(args.output_dir / "run_summary.json", {
        "method": "StateTrace-EDS-ECA", "split": args.split,
        "requested_tasks": len(jobs), "pending_tasks": len(pending),
        "workers": args.workers,
        "ok": sum(item["status"] == "ok" for item in results),
        "errors": sum(item["status"] == "error" for item in results),
        "results": results,
    })
    if any(item['status'] == 'error' for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
