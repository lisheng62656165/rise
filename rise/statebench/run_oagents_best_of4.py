"""Run official-ORM parallel Best-of-4 on StateBench.

For every task this runner creates four independent complete trajectories in
fresh environments, then makes one public-only four-way list-wise choice. The
selector sees visible messages, tool calls, tool results and public schemas;
benchmark scores are used only later by ``phase0_score.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from artifact_io import read_rows
from phase0_runner import DOMAINS, bundled_benchmark_root, public_mechanisms, read_split_ids, selected_jobs, write_json
from public_evidence import assert_public_payload, public_projection
from oagents_official_orm import PROTOCOL, build_messages, load_prompt, select_with_retries
from trace_ir import public_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official ORM Best-of-4 on StateBench")
    parser.add_argument("--statebench-root", type=Path, default=bundled_benchmark_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--workers", type=int, default=10, help="Total candidate-generation workers")
    parser.add_argument("--selector-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True, help="Base seed; candidates use seed+i")
    parser.add_argument("--selector-seed", type=int, default=77113)
    parser.add_argument("--tasks-per-domain", type=int, default=0)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--selector-retries", type=int, default=2)
    parser.add_argument('--anchor-dir', type=Path, help='Reuse the same Vanilla A as EDS-ECA candidate 0')
    parser.add_argument('--cohort-file', type=Path)
    return parser.parse_args()


def close_client(client: Any) -> None:
    transport = getattr(client, "_client", None)
    close = getattr(transport, "close", None)
    if close is not None:
        close()


def candidate_path(root: Path, candidate_index: int, domain: str, task_id: str) -> Path:
    return root / "candidates" / f"candidate_{candidate_index}" / f"{domain}__{task_id}.json"


def final_path(root: Path, domain: str, task_id: str) -> Path:
    return root / "final" / f"{domain}__{task_id}.json"


def generate_one(
    args: argparse.Namespace, domain_name: str, task_id: str, candidate_index: int, ordinal: int,
) -> dict[str, Any]:
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
    client = ReliableMiMoClient.from_env()
    simulator_client = ReliableMiMoClient.from_env()
    candidate_seed = args.seed + candidate_index
    client.seed = candidate_seed
    simulator_client.seed = candidate_seed
    agent = MiMoAgent(client, system_prompt, domain.tool_schemas, env.tool_handlers, runtime_context=None)
    started = time.monotonic()
    try:
        trajectory = run_task(
            task, env_data, task.user_id, client=None, domain=domain, agent=agent,
            env=env, simulator_client=simulator_client,
            trajectory_metadata={
                "method": "OAgents official ORM Best-of-4 (StateBench)",
                "split": args.split,
                "domain": domain_name,
                "agent_model": client.model_name,
                "generation_seed": candidate_seed,
                "task_key": f"statebench::{domain_name}::{task_id}",
                "candidate_index": candidate_index,
                "candidate_seed": candidate_seed,
                "candidate_generation_ordinal": ordinal,
                "inference_policy": "independent_parallel_candidate",
                "agent_runtime_context_exposed": False,
                "gold_fields_exposed_to_agent": False,
                "online_gold_or_evaluator_used": False,
            },
        )
    finally:
        close_client(client)
        close_client(simulator_client)
    row = trajectory.to_dict()
    row.update(public_mechanisms(trajectory.conversation))
    row["task_key"] = f"statebench::{domain_name}::{task_id}"
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


def build_packet(rows: list[Mapping[str, Any]], domain: Any, order: list[int]) -> dict[str, Any]:
    shown = []
    for shown_index, original_index in enumerate(order):
        shown.append({"candidate_index": shown_index, "trajectory": public_trajectory(rows[original_index])})
    packet = {
        "candidates": shown,
        "public_only": True,
    }
    assert_public_payload(packet)
    return packet


def select_one(args: argparse.Namespace, key: str, rows: list[Mapping[str, Any]], ordinal: int) -> dict[str, Any]:
    sys.path.insert(0, str(args.statebench_root))
    from reliable_mimo_client import ReliableMiMoClient

    if len(rows) != 4:
        raise ValueError("Official ORM Best-of-4 requires exactly four candidates")
    client = ReliableMiMoClient.from_env()
    client.seed = args.selector_seed + ordinal
    try:
        choice, attempts = select_with_retries(
            client, build_messages(load_prompt(), rows),
            args.output_dir / "selector_audit" / (key.replace("::", "__") + ".json"),
            attempts=args.selector_retries + 1,
        )
        return {
            "task_key": key, "selected_candidate": choice["index"], "shown_order": [0, 1, 2, 3],
            "reason": choice["analysis"], "decision": choice, "fallback": False, "attempt": attempts,
            "selector_seed": client.seed, "selector_protocol": PROTOCOL,
            "selector_model": client.model, "selector_temperature": 0.0,
            "selector_max_tokens": 2048, "public_only": True, "outcome_used": False,
        }
    finally:
        close_client(client)


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.selector_workers < 1:
        raise ValueError("worker counts must be positive")
    if args.tasks_per_domain < 0 or args.selector_retries < 0:
        raise ValueError("task and retry counts must be non-negative")
    if not (os.environ.get("MIMO_API_KEY") or os.environ.get("MIMO_API_KEYS_FILE")):
        raise ValueError("Set MIMO_API_KEY or MIMO_API_KEYS_FILE")
    jobs = selected_jobs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.anchor_dir:
        anchors = read_rows(args.anchor_dir)
        for domain, task_id in jobs:
            row = anchors[f'statebench::{domain}::{task_id}']
            if row.get('inference_policy') != 'vanilla' or row.get('generation_seed') != args.seed:
                raise ValueError('Shared anchor must be Vanilla with the same generation seed')
            destination = candidate_path(args.output_dir, 0, domain, task_id)
            if not destination.exists():
                write_json(destination, row)
    # Official reselection used sorted full task keys for the seed offset.
    all_keys = sorted(f"statebench::{domain}::{task_id}" for domain in DOMAINS
                      for task_id in read_split_ids(args.statebench_root, domain, args.split))
    task_ordinals = {key: ordinal for ordinal, key in enumerate(all_keys)}
    for domain, task_id in jobs:
        path = final_path(args.output_dir, domain, task_id)
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("best_of4_selection", {}).get("selector_protocol") != PROTOCOL:
                raise ValueError("Existing final uses an older selector; use a new output directory")

    generation_jobs = [
        (domain, task_id, candidate_index, ordinal)
        for ordinal, (domain, task_id) in enumerate(jobs)
        for candidate_index in range(4)
        if not candidate_path(args.output_dir, candidate_index, domain, task_id).exists()
    ]
    generation_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one, args, domain, task_id, candidate_index, ordinal):
            (domain, task_id, candidate_index)
            for domain, task_id, candidate_index, ordinal in generation_jobs
        }
        for future in as_completed(futures):
            domain, task_id, candidate_index = futures[future]
            path = candidate_path(args.output_dir, candidate_index, domain, task_id)
            try:
                row = future.result()
                write_json(path, row)
                result = {"task_key": row["task_key"], "candidate_index": candidate_index, "status": "ok"}
            except BaseException as error:
                result = {"task_key": f"statebench::{domain}::{task_id}", "candidate_index": candidate_index,
                          "status": "error", "error": str(error)[:2000], "traceback": traceback.format_exc()[-8000:]}
                write_json(path.with_suffix(".error.json"), result)
            generation_results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    selection_jobs: list[tuple[str, list[Mapping[str, Any]], int]] = []
    for ordinal, (domain, task_id) in enumerate(jobs):
        final = final_path(args.output_dir, domain, task_id)
        if final.exists():
            continue
        paths = [candidate_path(args.output_dir, i, domain, task_id) for i in range(4)]
        if not all(path.exists() for path in paths):
            continue
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        key = f"statebench::{domain}::{task_id}"
        selection_jobs.append((key, rows, task_ordinals[key]))

    selection_results: list[dict[str, Any]] = []
    os.environ["MIMO_STREAMING"] = os.environ.get("MIMO_ORM_SELECTOR_STREAMING", "0")
    with ThreadPoolExecutor(max_workers=args.selector_workers) as executor:
        futures = {
            executor.submit(select_one, args, key, rows, ordinal): (key, rows)
            for key, rows, ordinal in selection_jobs
        }
        for future in as_completed(futures):
            key, rows = futures[future]
            try:
                decision = future.result()
                selected = int(decision["selected_candidate"])
                final = copy.deepcopy(rows[selected])
                final.update({
                    "method": "OAgents official ORM Best-of-4 (StateBench)",
                    "best_of4_selected_candidate": selected,
                    "best_of4_selection": decision,
                    "best_of4_candidate_count": 4,
                    "online_gold_or_evaluator_used": False,
                })
                domain, task_id = key.split("::", 2)[1:]
                write_json(final_path(args.output_dir, domain, task_id), final)
                selection_results.append(decision)
                print(json.dumps({"task_key": key, "selected_candidate": selected,
                                  "fallback": decision.get("fallback", False)}, ensure_ascii=False), flush=True)
            except BaseException as error:
                selection_results.append({"task_key": key, "status": "error", "error": str(error)[:2000]})
                print(json.dumps(selection_results[-1], ensure_ascii=False), flush=True)

    report = {
        "method": "OAgents official ORM Best-of-4 (StateBench)",
        "selector_protocol": PROTOCOL,
        "split": args.split,
        "requested_tasks": len(jobs),
        "candidate_generation_jobs": len(generation_jobs),
        "final_trajectories": len(list((args.output_dir / "final").glob("*.json"))) if (args.output_dir / "final").exists() else 0,
        "candidate_count": 4,
        "workers": args.workers,
        "selector_workers": args.selector_workers,
        "selector_fallbacks": sum(bool(row.get("fallback")) for row in selection_results),
        "selection_counts": {str(index): sum(row.get("selected_candidate") == index for row in selection_results) for index in range(4)},
        "seed": args.seed,
        "selector_seed": args.selector_seed,
        "generation_results": generation_results,
        "selection_results": selection_results,
        "selector_information_boundary": {"public_only": True, "evaluator_scores_exposed": False},
    }
    write_json(args.output_dir / "best_of4_report.json", report)
    if report['final_trajectories'] != len(jobs) or any(r['status'] == 'error' for r in generation_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
