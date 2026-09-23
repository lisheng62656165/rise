#!/usr/bin/env python3
"""Verifier-only scoring for stored OccuBench EDS-ECA stage trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


WRITE_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def load_tasks(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "data" / "eval_benchmark_solvable.jsonl"
    return {int(row["task_id"]): row for row in read_jsonl(path)}


def load_keys(path: Path, start_back: int, stride: int) -> list[tuple[int, str]]:
    keys = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    offsets = list(range(start_back, len(keys) + 1, stride))
    if not offsets:
        raise ValueError(f"No key exists at reverse offset {start_back}")
    return [(back, keys[len(keys) - back]) for back in offsets]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--api-key-env")
    parser.add_argument("--model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--stage", default="A")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Score this stage only when it is the result's accepted final stage.",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--key-back", type=int, default=4)
    parser.add_argument("--key-stride", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--reuse-existing-valid",
        action="store_true",
        help="Seed the output with valid stage verdicts already stored in results.jsonl.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.dataset_root))
    from occubench.lwm import WorldModelRegistry, create_client
    from occubench.verifier import Verifier

    os.environ.setdefault("OCCUBENCH_REQUEST_TIMEOUT", "180")
    os.environ.setdefault("OCCUBENCH_VERIFIER_MAX_TOKENS", "4096")
    os.environ.setdefault("OCCUBENCH_STREAM", "1")
    os.environ.setdefault(
        "OCCUBENCH_VERIFIER_EXTRA_BODY_JSON",
        '{"chat_template_kwargs":{"enable_thinking":false}}',
    )

    tasks = load_tasks(args.dataset_root)
    registry = WorldModelRegistry(str(args.dataset_root / "data" / "world_model_configs"))
    if args.key_file:
        keys = load_keys(args.key_file, args.key_back, args.key_stride)
    elif args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            raise ValueError(f"Missing API key environment variable: {args.api_key_env}")
        keys = [(0, api_key)]
    else:
        raise ValueError("Provide either --key-file or --api-key-env")
    run_rows = read_jsonl(args.run_dir / "results.jsonl")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        (int(row["task_id"]), str(row["stage"]))
        for row in read_jsonl(args.output)
        if row.get("verification_valid") is True
    }

    if args.reuse_existing_valid:
        reused: list[dict[str, Any]] = []
        for result in run_rows:
            if args.selected_only and result.get("final_stage") != args.stage:
                continue
            for row in result.get("stages", []):
                task_id = int(result["task_id"])
                if (
                    row.get("stage") == args.stage
                    and row.get("agent_model") == args.model
                    and row.get("verification_valid") is True
                    and (task_id, args.stage) not in completed
                ):
                    reused.append(
                        {
                            "task_id": task_id,
                            "stage": args.stage,
                            "verifier_model": row.get("verifier_model", args.model),
                            "is_correct": bool(row["is_correct"]),
                            "feedback": row.get("feedback", ""),
                            "verification_valid": True,
                            "source": "existing_stage_verdict",
                        }
                    )
                    completed.add((task_id, args.stage))
        if reused:
            with args.output.open("a", encoding="utf-8") as handle:
                for row in reused:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"reused_valid": len(reused), "stage": args.stage}), flush=True)

    work: list[tuple[int, dict[str, Any]]] = []
    for result in run_rows:
        if args.selected_only and result.get("final_stage") != args.stage:
            continue
        stage_rows = [row for row in result.get("stages", []) if row.get("stage") == args.stage]
        if not stage_rows:
            continue
        row = stage_rows[0]
        if row.get("agent_model") != args.model:
            continue
        task_id = int(result["task_id"])
        if (task_id, args.stage) not in completed:
            work.append((task_id, row))

    print(json.dumps({"stage": args.stage, "pending": len(work), "workers": args.workers}), flush=True)

    def score(index: int, task_id: int, stage_row: dict[str, Any]) -> dict[str, Any]:
        task = tasks[task_id]
        config = registry.get(task["env_name"])
        initial_state = config.get("task_initial_state", "{}")
        if isinstance(initial_state, dict):
            initial_state = json.dumps(initial_state, ensure_ascii=False)
        last_feedback = "Verification error"
        for attempt in range(args.retries):
            key_back, api_key = keys[(index + attempt) % len(keys)]
            client = create_client(api_key, args.base_url)
            verifier = Verifier(args.model, client=client, num_votes=1)
            verdict = verifier.check(
                task_scenario_name=task["task_scenario_name"],
                task_initial_state=initial_state,
                state_description=config.get("state_description", ""),
                agent_instruction=task["agent_instruction"],
                verification_plan=task["verification_plan"],
                trajectory=stage_row["trajectory"],
            )
            last_feedback = verdict.get("feedback", "")
            if last_feedback != "Verification error":
                return {
                    "task_id": task_id,
                    "stage": args.stage,
                    "verifier_model": args.model,
                    "is_correct": bool(verdict["is_correct"]),
                    "feedback": last_feedback,
                    "verification_valid": True,
                    "attempt": attempt + 1,
                    "key_back": key_back,
                }
            time.sleep(2 * (attempt + 1))
        return {
            "task_id": task_id,
            "stage": args.stage,
            "verifier_model": args.model,
            "is_correct": None,
            "feedback": last_feedback,
            "verification_valid": False,
            "attempt": args.retries,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score, index, task_id, row): task_id
            for index, (task_id, row) in enumerate(work)
        }
        for future in as_completed(futures):
            result = future.result()
            with WRITE_LOCK:
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
            print(
                json.dumps(
                    {
                        "task_id": result["task_id"],
                        "stage": result["stage"],
                        "valid": result["verification_valid"],
                        "correct": result["is_correct"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
