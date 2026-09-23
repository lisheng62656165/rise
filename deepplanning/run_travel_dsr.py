from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TRAVEL = ROOT / "travelplanning"
MODEL = "mimo-v2.5-pro"

SELECT_SYSTEM = (
    "Select the stronger completed public travel-planning trajectory. "
    "Use only visible request, tool calls, and tool results. Prefer exact tool-grounded "
    "coverage of requested constraints, consistent dates/routes, and fewer failed, "
    "repeated, contradictory, unsupported, or unverified recovery actions. "
    "Do not use hidden labels, evaluator output, candidate order, or trajectory length."
)
DSR_SYSTEM = (
    "StateTrace-DSR: the following public disagreement graph contains unresolved "
    "hypotheses extracted from two prior independent executions. Do not replay either "
    "trajectory or assume either hypothesis is correct. Re-query all needed facts with "
    "the legal tools in this fresh environment, resolve the visible travel request, and "
    "produce a complete plan in <plan></plan>. All plan facts must come from tool results."
)
SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_trajectory",
        "description": "Select one public trajectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_index": {"type": "integer", "enum": [0, 1]},
                "reason": {"type": "string"},
                "supporting_event_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["candidate_index", "reason", "supporting_event_ids"],
        },
    }
}


def canon(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def parse(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except Exception:
        return x


def refs(x: Any, key: str = "") -> set[str]:
    if isinstance(x, dict):
        return {r for k, v in x.items() for r in refs(v, str(k))}
    if isinstance(x, list):
        return {r for v in x for r in refs(v, key)}
    if x is not None and ("id" in key.lower() or key.lower() in {"brand", "color", "size"}):
        return {str(x)}
    return set()


def category(x: Any) -> str:
    text = canon(x).lower()
    if any(t in text for t in ("error", "failed", "failure", "exception", "not found", "invalid")):
        return "FAILURE"
    return "EMPTY" if x in ({}, [], None, "") else "SUCCESS"


def events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending: dict[str, tuple[str, Any]] = {}
    out = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                pending[str(call.get("id") or len(pending))] = (
                    str(fn.get("name") or ""), parse(fn.get("arguments") or {})
                )
        elif msg.get("role") == "tool":
            cid = str(msg.get("tool_call_id") or "")
            tool, args = pending.get(cid, ("unknown", {}))
            result = parse(msg.get("content"))
            out.append({
                "event_id": f"E{len(out):04d}", "tool": tool,
                "arguments": args, "entity_refs": sorted(refs(args) | refs(result)),
                "result_category": category(result),
            })
    return out


def public(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    return {"request": payload.get("query", ""), "events": events(messages),
            "final_answer": payload.get("final_plan", "")}


def disagreement(a: dict[str, Any], b: dict[str, Any], task_id: str) -> dict[str, Any]:
    def key(e: dict[str, Any]) -> str:
        return canon([e["tool"], e["entity_refs"]])
    ga: dict[str, list] = {}
    gb: dict[str, list] = {}
    for e in a["events"]: ga.setdefault(key(e), []).append(e)
    for e in b["events"]: gb.setdefault(key(e), []).append(e)
    conflicts = []
    for k in sorted(set(ga) | set(gb)):
        if canon(ga.get(k, [])) != canon(gb.get(k, [])):
            conflicts.append({"component_id": f"C{len(conflicts):04d}",
                              "kind": "PUBLIC_TRANSACTION_DISAGREEMENT",
                              "alternatives": [ga.get(k, []), gb.get(k, [])]})
    oa = [key(e) for e in a["events"]]
    ob = [key(e) for e in b["events"]]
    if oa != ob:
        conflicts.append({"component_id": f"C{len(conflicts):04d}",
                          "kind": "PUBLIC_TRANSACTION_ORDER_DISAGREEMENT",
                          "alternatives": [oa, ob]})
    return {"schema_version": "deepplanning_state_trace_dsr_travel_v1",
            "task_id": task_id, "material_disagreement": bool(conflicts),
            "conflict_components": conflicts, "public_only": True,
            "candidate_winner_included": False}


def selector(client: Any, candidates: list[dict[str, Any]], ordinal: int) -> dict[str, Any]:
    order = [0, 1] if ordinal % 2 == 0 else [1, 0]
    packet = {"candidates": [{"candidate_index": i, **candidates[j]} for i, j in enumerate(order)]}
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SELECT_SYSTEM},
                  {"role": "user", "content": canon(packet)}],
        tools=[SELECT_TOOL], tool_choice={"type": "function", "function": {"name": "select_trajectory"}},
        temperature=0,
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        return {"selected": 0, "fallback": True, "reason": "missing selector call", "supporting_event_ids": []}
    arg = json.loads(calls[0].function.arguments)
    shown = int(arg.get("candidate_index", 0))
    return {"selected": order[shown if shown in (0, 1) else 0], "fallback": False,
            "reason": str(arg.get("reason", "")),
            "supporting_event_ids": list(arg.get("supporting_event_ids", []))}


def load_agent(language: str):
    sys.path.insert(0, str(TRAVEL))
    from agent.tools_fn_agent import ToolsFnAgent
    from agent.prompts import get_system_prompt
    return ToolsFnAgent, get_system_prompt


def run_fresh_d(language: str, task_id: str, query: str, packet: dict[str, Any], out: Path, max_calls: int) -> None:
    ToolsFnAgent, get_prompt = load_agent(language)
    db = TRAVEL / "database" / f"database_{language}"
    agent = ToolsFnAgent(model=MODEL, sample_id=task_id, database_base_path=str(db), language=language)
    final, messages = agent.run(query, system_prompt=get_system_prompt(language) + "\n\n" + DSR_SYSTEM +
                                "\n\nPublic disagreement graph:\n" + canon(packet), max_llm_calls=max_calls)
    out.mkdir(parents=True, exist_ok=True)
    result = {"id": f"id_{task_id}", "query": query, "model": MODEL, "language": language,
              "final_plan": final, "messages": agent._serialize_messages(messages), "success": True,
              "method": "StateTrace-DSR-fresh-D"}
    (out / f"id_{task_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / "reports" / f"id_{task_id}.txt").parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "reports" / f"id_{task_id}.txt").write_text(final, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["zh", "en"], required=True)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--g", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-llm-calls", type=int, default=400)
    args = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["MIMO_API_KEY"], base_url=os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"))
    queries = {str(x["id"]): x["query"] for x in json.loads((TRAVEL / "data" / f"travelplanning_query_{args.language}.json").read_text(encoding="utf-8"))}
    a_files = {p.stem.removeprefix("id_"): p for p in (args.a / "trajectories").glob("id_*.json")}
    g_files = {p.stem.removeprefix("id_"): p for p in (args.g / "trajectories").glob("id_*.json")}
    ids = sorted(set(queries) & set(a_files) & set(g_files), key=lambda x: int(x))
    (args.out / "selection").mkdir(parents=True, exist_ok=True)
    records = []
    for ordinal, task_id in enumerate(ids):
        left, right = public(a_files[task_id]), public(g_files[task_id])
        sel = selector(client, [left, right], ordinal)
        packet = disagreement(left, right, task_id)
        records.append({"id": task_id, "first_selector": sel, "material_disagreement": packet["material_disagreement"]})
        (args.out / "selection" / f"id_{task_id}.json").write_text(json.dumps({"task_id": task_id, "selection": sel, "packet": packet}, ensure_ascii=False, indent=2), encoding="utf-8")
    d_ids = [r["id"] for r in records if r["material_disagreement"]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for task_id in d_ids:
            packet = json.loads((args.out / "selection" / f"id_{task_id}.json").read_text(encoding="utf-8"))["packet"]
            futs[pool.submit(run_fresh_d, args.language, task_id, queries[task_id], packet,
                             args.out / "dsr_d" / "trajectories", args.max_llm_calls)] = task_id
        for f in as_completed(futs):
            f.result()
    # Non-triggered cases use the first selector's selected candidate as the final public trajectory.
    for r in records:
        task_id = r["id"]
        if r["material_disagreement"]:
            continue
        source = [a_files[task_id], g_files[task_id]][r["first_selector"]["selected"]]
        dst = args.out / "final" / "trajectories" / f"id_{task_id}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)
        (args.out / "final" / "reports").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.parent.parent / "reports" / f"id_{task_id}.txt", args.out / "final" / "reports" / f"id_{task_id}.txt")
    for p in (args.out / "dsr_d" / "trajectories").glob("id_*.json"):
        dst = args.out / "final" / "trajectories" / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    summary = {"language": args.language, "candidate_a": len(a_files), "candidate_g": len(g_files),
               "paired": len(ids), "material_disagreement": len(d_ids),
               "selector_fallbacks": sum(r["first_selector"]["fallback"] for r in records),
               "final": len(list((args.out / "final" / "trajectories").glob("id_*.json"))),
               "method": "StateTrace-DSR", "full_cohort": True}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
