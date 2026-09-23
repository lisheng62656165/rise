from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_trajectory",
        "description": "Select the strongest public trajectory for the visible task.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_index": {"type": "integer", "enum": [0, 1, 2, 3]},
                "reason": {"type": "string"},
                "supporting_event_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidate_index", "reason", "supporting_event_ids"],
        },
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidate_functions(root: Path):
    sys.path.insert(0, str(root))
    from run_deepplanning_eds_eca import message_rows, public_candidate
    return message_rows, public_candidate


def query_for(root: Path, level: int, case_id: int) -> str:
    rows = load_json(root / "shoppingplanning" / f"data/level_{level}_query_meta.json")
    return next(str(row["query"]) for row in rows if int(row["id"]) == case_id)


def select_one(client: OpenAI, candidates: list[dict[str, Any]], ordinal: int, seed: int) -> dict[str, Any]:
    order = list(range(4))
    random.Random(seed + ordinal).shuffle(order)
    shown = []
    allowed: dict[int, set[str]] = {}
    for shown_index, original_index in enumerate(order):
        item = copy.deepcopy(candidates[original_index])
        for index, event in enumerate(item.get("events", [])):
            event["event_id"] = f"C{shown_index}-E{index:04d}"
        item["candidate_index"] = shown_index
        shown.append(item)
        allowed[shown_index] = {event["event_id"] for event in item.get("events", [])}
    system = (
        "Select one of four independently executed public trajectories for the same visible DeepPlanning task. "
        "Use only the request, public tool calls/results, and final answer. Prefer complete request coverage, "
        "grounded facts, valid action order, successful recovery, verification, and correct stopping. "
        "Do not use candidate order, trajectory length, evaluator output, hidden labels, or gold state."
    )
    response = client.chat.completions.create(
        model=os.environ.get("DEEPPLANNING_API_MODEL", "deepseek-v4.1-flash"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": canonical({"candidates": shown})}],
        tools=[SELECT_TOOL],
        tool_choice={"type": "function", "function": {"name": "select_trajectory"}},
        temperature=0,
        max_tokens=4096,
        timeout=300,
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        return {"selected": 0, "fallback": True, "reason": "missing selector tool call"}
    args = json.loads(calls[0].function.arguments)
    selected_shown = int(args.get("candidate_index", -1))
    if selected_shown not in range(4):
        return {"selected": 0, "fallback": True, "reason": "invalid candidate index"}
    supporting = list(args.get("supporting_event_ids") or [])
    if supporting and not set(supporting).issubset(allowed[selected_shown]):
        return {"selected": 0, "fallback": True, "reason": "invalid event provenance"}
    return {
        "selected": order[selected_shown],
        "fallback": False,
        "reason": str(args.get("reason") or ""),
        "supporting_event_ids": supporting,
    }


def run_shopping(args: argparse.Namespace, level: int, case_id: int, ordinal: int, client: OpenAI) -> dict[str, Any]:
    root = args.root
    shop = root / "shoppingplanning"
    message_rows, public_candidate = load_candidate_functions(root)
    sources = [shop / "database_infered" / f"database_deepseek_best4_c{i}_L{level}" / f"case_{case_id}" for i in range(4)]
    candidates = []
    for i, source in enumerate(sources):
        payload = load_json(source / "messages.json")
        candidates.append(public_candidate(query_for(root, level, case_id), message_rows(payload), f"C{i}-E"))
    decision = select_one(client, candidates, ordinal, args.selector_seed)
    selected_source = sources[int(decision["selected"])]
    target = shop / "database_infered" / f"database_deepseek_best4_selected_L{level}" / f"case_{case_id}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(selected_source, target)
    return {"task": f"shopping::L{level}::{case_id}", **decision}


def run_travel(args: argparse.Namespace, language: str, task_id: int, ordinal: int, client: OpenAI) -> dict[str, Any]:
    root = args.root
    message_rows, public_candidate = load_candidate_functions(root)
    model = "deepseek-v4.1-flash"
    sources = [root / "results" / "deepseek_dp_compare" / "best4" / f"c{i}" / "travel_runs" / f"deepseek-v4.1-flash_{language}" / "trajectories" / f"id_{task_id}.json" for i in range(4)]
    candidates = []
    for i, source in enumerate(sources):
        payload = load_json(source)
        candidates.append(public_candidate(str(payload["query"]), message_rows(payload.get("messages", [])), f"C{i}-E"))
    decision = select_one(client, candidates, ordinal, args.selector_seed)
    selected_source = sources[int(decision["selected"])]
    target_root = root / "results" / "deepseek_dp_compare" / "best4_selected" / "travel_runs"
    target = target_root / f"deepseek-v4.1-flash_{language}" / "trajectories" / f"id_{task_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(selected_source, target)
    reports = target.parent.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    selected_payload = load_json(selected_source)
    reports.joinpath(f"id_{task_id}.txt").write_text(str(selected_payload.get("final_plan", "")), encoding="utf-8")
    return {"task": f"travel-{language}::{task_id}", **decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort", choices=("shopping", "travel-zh", "travel-en"), required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--selector-seed", type=int, default=77113)
    args = parser.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required")
    client = OpenAI(api_key=key, base_url=os.environ.get("DEEPPLANNING_BASE_URL", "https://hgapi.dieqiyun.top/v1"), timeout=300, max_retries=2)
    if args.cohort == "shopping":
        jobs = [(level, case_id) for level, count in ((1, 50), (2, 50), (3, 20)) for case_id in range(1, count + 1)]
        runner = lambda job, ordinal: run_shopping(args, job[0], job[1], ordinal, client)
    else:
        language = args.cohort.rsplit("-", 1)[1]
        jobs = [(language, task_id) for task_id in range(120)]
        runner = lambda job, ordinal: run_travel(args, job[0], job[1], ordinal, client)
    output = args.root / "results" / "deepseek_dp_compare" / "best4_selected" / f"selection_{args.cohort.replace('-', '_')}"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(runner, job, ordinal): job for ordinal, job in enumerate(jobs)}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {"cohort": args.cohort, "tasks": len(rows), "fallbacks": sum(row.get("fallback", False) for row in rows), "selector_seed": args.selector_seed, "rows": rows}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
