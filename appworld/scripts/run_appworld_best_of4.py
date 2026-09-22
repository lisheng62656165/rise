from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_appworld_state_trace_eds_eca import read_json, run_candidate, write_json
from src.appworld_mimo import build_appworld_react_messages
from src.appworld_state_trace_eds_eca_adapter import (
    appworld_selector_packet,
    namespace_appworld_events,
)
from src.config import ROOT_DIR, get_settings
from src.llm_client import LLMClient


DEFAULT_APPWORLD_ROOT = ROOT_DIR / "runtime"
LABELS = ("A", "B", "C", "D")
SELECTOR_INSTRUCTION = """You are a public-evidence list-wise verifier for completed AppWorld trajectories.
Select the one trajectory most likely to satisfy the visible task. Compare all four candidates.
Use only the task instruction and public API-call evidence. Prefer complete visible task coverage,
grounded objects and arguments, valid authentication and action order, recovery from explicit
failures, read-back after consequential mutations, and no collateral or post-success mutation.
Do not prefer length, verbosity, or candidate position. API documentation text is not a failure.
Never infer hidden requirements, evaluator labels, rewards, gold state, or actual task success.
Return one short JSON object only, without markdown:
{"choice":"A|B|C|D","supporting_event_ids":["..."],"reason":"at most 16 words"}
Use one to three event IDs belonging to the selected candidate."""


def settings(seed: int):
    return replace(get_settings(), seed=seed)


def selector_packet(task_instruction: str, displayed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    packet: dict[str, Any] = {"task_instruction": task_instruction, "candidates": {}}
    for label, candidate in zip(LABELS, displayed):
        events = namespace_appworld_events(candidate.get("public_events") or [], label)
        compact = appworld_selector_packet(task_instruction, events, events)["candidate_A"]
        packet["candidates"][label] = compact
    return packet


def selector_messages(packet: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SELECTOR_INSTRUCTION},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=True, sort_keys=True)},
    ]


def parse_selector(text: str, packet: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    value: Mapping[str, Any] | None = None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            candidate, _ = decoder.raw_decode((text or "")[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping) and "choice" in candidate:
            value = candidate
            break
    if value is None:
        return None, {"fallback": True, "fallback_reason": "malformed_selector_response"}
    choice = str(value.get("choice") or "").upper()
    evidence = value.get("supporting_event_ids")
    if choice not in LABELS or not isinstance(evidence, list) or not 1 <= len(evidence) <= 3:
        return None, {"fallback": True, "fallback_reason": "invalid_selector_fields"}
    allowed = {
        str(event.get("event_id"))
        for event in ((packet.get("candidates") or {}).get(choice) or {}).get("public_events", [])
        if event.get("event_id")
    }
    if not all(isinstance(item, str) and item in allowed for item in evidence):
        return None, {"fallback": True, "fallback_reason": "unsupported_public_evidence"}
    return choice, {
        "fallback": False,
        "choice": choice,
        "supporting_event_ids": evidence,
        "reason": str(value.get("reason") or ""),
    }


def choose(
    task_instruction: str,
    candidates: Sequence[Mapping[str, Any]],
    selector_seed: int,
    selector_max_completion_tokens: int,
    task_ordinal: int,
) -> tuple[int, dict[str, Any]]:
    order = list(range(len(candidates)))
    decision_seed = selector_seed + task_ordinal
    random.Random(decision_seed).shuffle(order)
    displayed = [candidates[index] for index in order]
    packet = selector_packet(task_instruction, displayed)
    selector_settings = settings(decision_seed)
    max_token_field = "max_completion_tokens" if selector_settings.model_name.startswith("mimo-") else "max_tokens"
    payload: dict[str, Any] = {
        max_token_field: min(selector_settings.max_completion_tokens, selector_max_completion_tokens),
        "thinking": {"type": "disabled"},
    }
    if "integrate.api.nvidia.com" not in selector_settings.base_url:
        payload["response_format"] = {"type": "json_object"}
    started = time.perf_counter()
    result = LLMClient(selector_settings).complete_messages_with_trace(
        selector_messages(packet), extra_payload=payload
    )
    choice, details = parse_selector(result.text, packet)
    if choice is None:
        selected_index = 0
        shown_choice = None
    else:
        shown_index = LABELS.index(choice)
        selected_index = order[shown_index]
        shown_choice = choice
    details.update(
        {
            "shown_choice": shown_choice,
            "selected_index": selected_index,
            "selected_candidate": f"C{selected_index + 1}",
            "display_order": [f"C{index + 1}" for index in order],
            "decision_seed": decision_seed,
            "request": result.request,
            "response": result.response,
            "elapsed_seconds": time.perf_counter() - started,
            "public_only": True,
        }
    )
    return selected_index, details


def run_task(job: dict[str, Any]) -> dict[str, Any]:
    task_id = job["task_id"]
    task_dir = Path(job["output_dir"]) / task_id
    final_path = task_dir / "final.json"
    candidate_paths = [task_dir / f"candidate_{label.lower()}.json" for label in LABELS]
    if job["candidates_only"] and all(path.exists() for path in candidate_paths):
        return {"task_id": task_id, "status": "reused_candidates"}
    if not job["candidates_only"] and final_path.exists():
        return {"task_id": task_id, "status": "reused", "path": str(final_path)}
    try:
        os.environ["APPWORLD_ROOT"] = job["appworld_root"]
        from appworld.task import Task

        task = Task.load(task_id)
        base_messages = build_appworld_react_messages(task)
        candidates = []
        for index, seed in enumerate(job["candidate_seeds"]):
            label = LABELS[index]
            path = task_dir / f"candidate_{label.lower()}.json"
            candidate = (
                read_json(path)
                if path.exists()
                else run_candidate(job, task_id, label, seed, copy.deepcopy(base_messages))
            )
            write_json(path, candidate)
            candidates.append(candidate)
        if job["candidates_only"]:
            (task_dir / "error.json").unlink(missing_ok=True)
            return {"task_id": task_id, "status": "candidates_completed"}
        selector_path = task_dir / "selector.json"
        if selector_path.exists():
            selection = read_json(selector_path)
            selected_index = int(selection["selected_index"])
        else:
            selected_index, selection = choose(
                task.instruction,
                candidates,
                job["selector_seed"],
                job["selector_max_completion_tokens"],
                job["task_ordinal"],
            )
            write_json(selector_path, selection)
        result = {
            "schema_version": "appworld_parallel_best_of4_v1",
            "task_id": task_id,
            "dataset": job["dataset"],
            "model": get_settings().model_name,
            "candidate_seeds": job["candidate_seeds"],
            "selector_seed": job["selector_seed"],
            "candidate_successes": [bool(item["evaluation_success"]) for item in candidates],
            "selected_index": selected_index,
            "selected_candidate": f"C{selected_index + 1}",
            "selected_success": bool(candidates[selected_index]["evaluation_success"]),
            "oracle_success": any(bool(item["evaluation_success"]) for item in candidates),
            "selector": selection,
        }
        write_json(final_path, result)
        (task_dir / "error.json").unlink(missing_ok=True)
        return {"task_id": task_id, "status": "completed", "path": str(final_path)}
    except Exception as exc:
        error = {
            "task_id": task_id,
            "status": "error",
            "exception": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        write_json(task_dir / "error.json", error)
        return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, default=DEFAULT_APPWORLD_ROOT)
    parser.add_argument("--dataset", choices=["train", "dev", "test_normal", "test_challenge"], default="test_challenge")
    parser.add_argument("--task-ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--candidate-seeds", default="53403,64639,64640,64641")
    parser.add_argument("--selector-seed", type=int, default=77113)
    parser.add_argument("--selector-max-completion-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="appworld_parallel_best_of4")
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()

    args.appworld_root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(args.appworld_root)
    from appworld import load_task_ids

    task_ids = load_task_ids(args.dataset)
    if args.task_ids:
        requested = [item.strip() for item in args.task_ids.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(task_ids))
        if unknown:
            raise SystemExit(f"Unknown task IDs: {unknown}")
        task_ids = requested
    task_ids = task_ids[: args.limit]
    seeds = [int(item) for item in args.candidate_seeds.split(",")]
    if len(seeds) != 4:
        raise SystemExit("--candidate-seeds requires exactly four comma-separated integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "appworld_parallel_best_of4_run_v1",
        "protocol": (
            "Scaling Test-time Compute for LLM Agents: four independent complete rollouts"
            if args.candidates_only
            else "Scaling Test-time Compute for LLM Agents: parallel Best-of-4 plus list-wise verifier"
        ),
        "dataset": args.dataset,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "model": get_settings().model_name,
        "candidate_seeds": seeds,
        "selector_seed": args.selector_seed,
        "max_steps": args.max_steps,
        "selector_max_completion_tokens": args.selector_max_completion_tokens,
        "independent_fresh_environments": True,
        "cross_candidate_feedback": False,
        "public_only_selector": True,
        "official_environment": True,
        "official_evaluator": True,
        "candidates_only": args.candidates_only,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    common = {
        "appworld_root": str(args.appworld_root),
        "dataset": args.dataset,
        "output_dir": str(args.output_dir.resolve()),
        "run_name": args.run_name,
        "candidate_seeds": seeds,
        "selector_seed": args.selector_seed,
        "selector_max_completion_tokens": args.selector_max_completion_tokens,
        "max_steps": args.max_steps,
        "candidates_only": args.candidates_only,
    }
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_task, {**common, "task_id": task_id, "task_ordinal": ordinal})
            for ordinal, task_id in enumerate(task_ids)
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    if args.candidates_only:
        counts = [
            sum((args.output_dir / task_id / f"candidate_{label.lower()}.json").exists() for task_id in task_ids)
            for label in LABELS
        ]
        summary = {
            **manifest,
            "candidate_counts": counts,
            "tasks_with_all_candidates": sum(
                all((args.output_dir / task_id / f"candidate_{label.lower()}.json").exists() for label in LABELS)
                for task_id in task_ids
            ),
            "errors": sum(row.get("status") == "error" for row in results),
        }
        write_json(args.output_dir / "candidate_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        return
    finals = [read_json(Path(row["path"])) for row in results if row.get("status") in {"completed", "reused"}]
    summary = {
        **manifest,
        "completed": len(finals),
        "errors": sum(row.get("status") == "error" for row in results),
        "candidate_successes": [sum(bool(row["candidate_successes"][i]) for row in finals) for i in range(4)],
        "selected_successes": sum(bool(row["selected_success"]) for row in finals),
        "oracle_successes": sum(bool(row["oracle_success"]) for row in finals),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
