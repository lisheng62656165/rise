from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


DSR_INSTRUCTION = (
    "StateTrace-DSR provides a symmetric public disagreement graph from two "
    "independent executions of this same request. Every value in the graph is "
    "an unresolved hypothesis. Complete the full task in this fresh environment. "
    "Reacquire needed facts with current legal tools; do not replay either prior "
    "trajectory, infer a winner, or assume a candidate result is correct."
)

SELECT_INSTRUCTION = (
    "Select the stronger completed public trajectory for the visible shopping "
    "request. Prefer verified request coverage, preserved cart effects, and fewer "
    "failed, repeated, contradictory, wrong-target, or unsupported actions. Do not "
    "use candidate length or order as quality evidence."
)

SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_trajectory",
        "description": "Select one completed public trajectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_index": {"type": "integer", "enum": [0, 1]},
                "reason": {"type": "string"},
                "supporting_event_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidate_index", "reason", "supporting_event_ids"],
        },
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_messages(case_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((case_dir / "messages.json").read_text(encoding="utf-8"))
    rows = payload.get("messages") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError(f"invalid messages: {case_dir}")
    return rows


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def entity_refs(value: Any, key: str = "") -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            refs.update(entity_refs(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            refs.update(entity_refs(child, key))
    elif value is not None and ("id" in key.lower() or key.lower() in {"brand", "color", "size"}):
        refs.add(str(value))
    return refs


def operation(tool: str) -> str:
    name = tool.lower()
    if name.startswith(("add_", "delete_", "remove_", "update_", "set_")):
        return "WRITE"
    return "READ"


def result_category(content: Any) -> str:
    text = json.dumps(content, ensure_ascii=False, sort_keys=True).lower()
    if any(token in text for token in ("error", "failed", "failure", "exception", "not found", "invalid")):
        return "FAILURE"
    value = parse_json(content)
    if value in ({}, [], None, ""):
        return "EMPTY"
    return "SUCCESS"


def compile_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: dict[str, tuple[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                pending[str(call.get("id") or len(pending))] = (
                    str(function.get("name") or ""), parse_json(function.get("arguments") or {}),
                )
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            tool, arguments = pending.get(call_id, ("unknown", {}))
            result = parse_json(message.get("content"))
            refs = entity_refs(arguments) | entity_refs(result)
            events.append({
                "event_id": f"E{len(events):04d}", "operation": operation(tool), "tool": tool,
                "arguments": arguments, "entity_refs": sorted(refs),
                "result_category": result_category(result),
            })
    return events


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def disagreement(left: list[dict[str, Any]], right: list[dict[str, Any]], task_key: str) -> dict[str, Any]:
    def grouped(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            key = canonical([event["operation"], event["tool"], event["entity_refs"]])
            output.setdefault(key, []).append({
                "operation": event["operation"], "tool": event["tool"],
                "entity_refs": event["entity_refs"], "arguments": event["arguments"],
                "result_category": event["result_category"],
            })
        return output

    groups = [grouped(left), grouped(right)]
    conflicts = []
    for key in sorted(set(groups[0]) | set(groups[1])):
        alternatives = [groups[0].get(key, []), groups[1].get(key, [])]
        if canonical(alternatives[0]) != canonical(alternatives[1]):
            conflicts.append({
                "component_id": f"C{len(conflicts):04d}", "kind": "TRANSACTION_CONTENT",
                "transaction_key": json.loads(key),
                "unlabeled_alternatives": sorted([alternatives[0], alternatives[1]], key=canonical),
            })
    orders = [[canonical([e["operation"], e["tool"], e["entity_refs"]]) for e in rows] for rows in (left, right)]
    if canonical(orders[0]) != canonical(orders[1]):
        conflicts.append({
            "component_id": f"C{len(conflicts):04d}", "kind": "TRANSACTION_ORDER",
            "unlabeled_alternatives": sorted(orders, key=canonical),
        })
    return {
        "schema_version": "deepplanning_state_trace_dsr_packet_v1", "task_key": task_key,
        "material_disagreement": bool(conflicts), "conflict_components": conflicts,
        "public_only": True, "candidate_identity_removed": True, "candidate_winner_included": False,
    }


def public_candidate(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "request": str(messages[1].get("content") or ""),
        "events": compile_events(messages),
        "final_answer": next((str(row.get("content") or "") for row in reversed(messages) if row.get("role") == "assistant" and row.get("content")), ""),
    }


def selector(client: OpenAI, candidates: list[dict[str, Any]], model: str, ordinal: int) -> dict[str, Any]:
    order = [0, 1] if ordinal % 2 == 0 else [1, 0]
    packet = {"candidates": [{"candidate_index": shown, **candidates[original]} for shown, original in enumerate(order)]}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SELECT_INSTRUCTION}, {"role": "user", "content": canonical(packet)}],
        tools=[SELECT_TOOL], tool_choice={"type": "function", "function": {"name": "select_trajectory"}}, temperature=0,
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        return {"selected": 0, "fallback": True, "reason": "missing selector tool call"}
    arguments = json.loads(calls[0].function.arguments)
    shown = int(arguments["candidate_index"])
    return {
        "selected": order[shown], "fallback": False, "reason": str(arguments.get("reason") or ""),
        "supporting_event_ids": list(arguments.get("supporting_event_ids") or []),
    }


def case_map(root: Path, arm: str) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for level, count in ((1, 50), (2, 50), (3, 20)):
        database = root / "shoppingplanning" / f"database_run_full_{arm}_L{level}_20260828"
        for case_id in range(1, count + 1):
            rows[f"L{level}::{case_id}"] = database / f"case_{case_id}"
    return rows


def select_stage(root: Path, left: dict[str, Path], right: dict[str, Path], name: str, workers: int) -> tuple[dict[str, Path], dict[str, Any]]:
    output = root / "selection" / name
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ["MIMO_API_KEY"], base_url=os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"))
    selected: dict[str, Path] = {}
    rows = []

    def run(ordinal: int, key: str) -> dict[str, Any]:
        path = output / f"{key.replace('::', '_')}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        candidates = [public_candidate(load_messages(left[key])), public_candidate(load_messages(right[key]))]
        decision = selector(client, candidates, "mimo-v2.5-pro", ordinal)
        row = {"task_key": key, **decision}
        write_json(path, row)
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, ordinal, key): key for ordinal, key in enumerate(sorted(left))}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            selected[row["task_key"]] = [left[row["task_key"]], right[row["task_key"]]][row["selected"]]
    report = {"tasks": len(rows), "selected_left": sum(r["selected"] == 0 for r in rows), "selected_right": sum(r["selected"] == 1 for r in rows), "fallbacks": sum(r["fallback"] for r in rows)}
    write_json(output / "summary.json", report)
    return selected, report


def run_d(root: Path, packets: dict[str, dict[str, Any]], workers: int, max_calls: int) -> dict[str, Path]:
    shopping_root = root / "shoppingplanning"
    sys.path.insert(0, str(shopping_root))
    from agent.prompts import prompt_lib
    from agent.shopping_agent import ShoppingFnAgent

    databases: dict[int, Path] = {}
    for level in (1, 2, 3):
        target = shopping_root / f"database_run_full_dsr_d_L{level}_20260828"
        if not target.exists():
            shutil.copytree(shopping_root / f"database_level{level}", target)
        databases[level] = target

    def run(key: str) -> tuple[str, Path]:
        level, case_id = int(key[1]), int(key.split("::")[1])
        case_dir = databases[level] / f"case_{case_id}"
        if (case_dir / "messages.json").exists():
            return key, case_dir
        agent = ShoppingFnAgent(model="mimo-v2.5-pro", sample_id=str(case_id), database_base_path=str(databases[level]), tool_schema_path=str(shopping_root / "tools/shopping_tool_schema.json"))
        query_rows = json.loads((shopping_root / f"data/level_{level}_query_meta.json").read_text(encoding="utf-8"))
        query = next(row["query"] for row in query_rows if int(row["id"]) == case_id)
        prompt = f"{getattr(prompt_lib, f'SYSTEM_PROMPT_level{level}')}\n\n{DSR_INSTRUCTION}\n\nPublic disagreement graph:\n{canonical(packets[key])}"
        agent.run(user_query=query, system_prompt=prompt, save_messages=True, sample_id=str(case_id), max_llm_calls=max_calls)
        return key, case_dir

    output: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, key): key for key in packets if packets[key]["material_disagreement"]}
        for future in as_completed(futures):
            key, path = future.result()
            output[key] = path
    return output


def material_packets(root: Path, left: dict[str, Path], right: dict[str, Path]) -> dict[str, dict[str, Any]]:
    output = root / "packets"
    output.mkdir(parents=True, exist_ok=True)
    packets = {}
    for key in sorted(left):
        path = output / f"{key.replace('::', '_')}.json"
        packet = disagreement(compile_events(load_messages(left[key])), compile_events(load_messages(right[key])), key)
        write_json(path, packet)
        packets[key] = packet
    write_json(output / "summary.json", {"tasks": len(packets), "material": sum(p["material_disagreement"] for p in packets.values())})
    return packets


def assemble_and_evaluate(root: Path, selected: dict[str, Path], name: str) -> Path:
    shopping_root = root / "shoppingplanning"
    database = shopping_root / "database_infered" / f"database_{name}"
    if database.exists():
        shutil.rmtree(database)
    database.mkdir(parents=True)
    for key, source in selected.items():
        level, case_id = key.split("::")
        shutil.copytree(source, database / f"case_{level}_{case_id}")
    subprocess.run([sys.executable, str(shopping_root / "evaluation/evaluation_pipeline.py"), "--database_dir", str(database)], check=True)
    return shopping_root / "result_report" / database.name / "summary_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-llm-calls", type=int, default=400)
    args = parser.parse_args()
    left, right = case_map(args.root, "vanilla_a"), case_map(args.root, "generic_g")
    missing = [str(path) for path in [*left.values(), *right.values()] if not (path / "messages.json").exists()]
    if missing:
        raise SystemExit(f"candidate trajectories are incomplete: {len(missing)} missing")
    packets = material_packets(args.root, left, right)
    base, base_report = select_stage(args.root, left, right, "state_trace_base", args.workers)
    baseline_report = assemble_and_evaluate(args.root, left, "full_vanilla_a_20260828")
    d = run_d(args.root, packets, args.workers, args.max_llm_calls)
    final_left, final_right = {}, {}
    for key in base:
        final_left[key] = base[key]
        final_right[key] = d.get(key, base[key])
    final, final_report = select_stage(args.root, final_left, final_right, "state_trace_dsr", args.workers)
    dsr_report = assemble_and_evaluate(args.root, final, "full_state_trace_dsr_20260828")
    write_json(args.root / "run_summary.json", {
        "cohort": "full_120_direct_evaluation", "baseline_report": str(baseline_report),
        "dsr_report": str(dsr_report), "material_tasks": len(d),
        "base_selector": base_report, "dsr_selector": final_report,
    })


if __name__ == "__main__":
    main()
