from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


MODEL = os.environ.get("DEEPPLANNING_MODEL", "mimo-v2.5-pro")
API_MODEL = os.environ.get("DEEPPLANNING_API_MODEL", MODEL)
ISSUE_TYPES = (
    "NONE", "FAILURE_RECOVERY", "TARGET_IDENTITY", "PARAMETER_GROUNDING",
    "ACTION_ORDER", "VERIFICATION", "TASK_COVERAGE", "STOPPING", "OTHER",
)
ISSUE_DIRECTIVES = {
    "NONE": "",
    "FAILURE_RECOVERY": "Do not repeat a rejected action; reacquire its target and preconditions before recovery.",
    "TARGET_IDENTITY": "Reacquire target identity and bind actions only to publicly supported objects.",
    "PARAMETER_GROUNDING": "Recompute arguments from current public facts and visible constraints.",
    "ACTION_ORDER": "Re-establish public preconditions and execute dependent actions in a valid order.",
    "VERIFICATION": "Read back or otherwise verify consequential actions before stopping.",
    "TASK_COVERAGE": "Complete every visible user request and constraint.",
    "STOPPING": "Stop only after visible obligations are complete; avoid post-success actions.",
    "OTHER": "Avoid the rejected event structure and re-ground the next action from current evidence.",
}
SELECT_SYSTEM = (
    "StateTrace-EDS-ECA compares two completed public trajectories for the same visible task. "
    "Select the stronger trajectory, then assign event-level credit for the next independent rollout. "
    "Prefer exact tool-grounded request coverage, valid targets and parameters, coherent action order, "
    "successful recovery, verification, and correct stopping. Preserve IDs must belong to the selected "
    "candidate and avoid IDs to the rejected candidate. Do not use hidden labels, evaluator output, "
    "candidate order, or trajectory length. Call select_and_credit_events exactly once."
)
DEEPPLANNING_SELECT_ADDENDUM = (
    " For DeepPlanning travel candidates, also inspect the provided public schema and grounding signals: "
    "parseable daily-plan structure, time-entry presence, successful visible travel lookups, and visible "
    "tool failures. Use them as evidence, never as evaluator labels or hidden-state claims."
)
SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_and_credit_events",
        "description": "Select one public trajectory and assign local event credit.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_index": {"type": "integer", "enum": [0, 1]},
                "reason": {"type": "string"},
                "supporting_event_ids": {"type": "array", "items": {"type": "string"}},
                "preserve_event_ids": {"type": "array", "items": {"type": "string"}},
                "avoid_event_ids": {"type": "array", "items": {"type": "string"}},
                "unresolved_issue_type": {"type": "string", "enum": list(ISSUE_TYPES)},
                "unresolved_issue": {"type": "string"},
            },
            "required": [
                "candidate_index", "reason", "supporting_event_ids", "preserve_event_ids",
                "avoid_event_ids", "unresolved_issue_type", "unresolved_issue",
            ],
        },
    },
}


def collect_stream_tool_calls(stream: Any) -> list[dict[str, Any]]:
    """Collect fragmented OpenAI-compatible tool calls from a streamed response."""
    calls: dict[int, dict[str, Any]] = {}
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        for call in getattr(chunk.choices[0].delta, "tool_calls", None) or []:
            index = getattr(call, "index", 0) or 0
            target = calls.setdefault(
                index,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if getattr(call, "id", None):
                target["id"] += call.id
            function = getattr(call, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    target["function"]["name"] += function.name
                if getattr(function, "arguments", None):
                    target["function"]["arguments"] += function.arguments
    return [calls[index] for index in sorted(calls)]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def message_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("messages", []) if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def result_category(value: Any) -> str:
    text = canonical(value).lower()
    if any(term in text for term in ("error", "failed", "failure", "exception", "not found", "invalid")):
        return "FAILURE"
    return "EMPTY" if value in ({}, [], None, "") else "SUCCESS"


def public_events(messages: list[dict[str, Any]], prefix: str = "E") -> list[dict[str, Any]]:
    pending: dict[str, tuple[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                pending[str(call.get("id") or len(pending))] = (
                    str(function.get("name") or ""), parse(function.get("arguments") or {}),
                )
        elif message.get("role") == "tool":
            call_id = str(message.get("tool_call_id") or "")
            tool, arguments = pending.get(call_id, (str(message.get("name") or "unknown"), {}))
            result = parse(message.get("content"))
            events.append({
                "event_id": f"{prefix}{len(events):04d}", "tool": tool,
                "arguments": arguments, "argument_keys": sorted(arguments) if isinstance(arguments, dict) else [],
                "result": result, "result_category": result_category(result),
            })
    return events


def final_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(message.get("content") or "").strip():
            return str(message["content"])
    return ""


def public_candidate(query: str, messages: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "request": query,
        "events": public_events(messages, prefix),
        "final_answer": final_text(messages),
    }


def structural_frontier(candidate: dict[str, Any], credit: dict[str, Any] | None) -> dict[str, Any]:
    ranked = sorted(
        candidate["events"],
        key=lambda event: (event["result_category"] == "FAILURE", bool(event["argument_keys"])),
        reverse=True,
    )[:12]
    return {
        "schema_version": "deepplanning_state_trace_eds_eca_v1",
        "high_value_events": [
            {
                "source_event_id": event["event_id"], "tool": event["tool"],
                "argument_keys": event["argument_keys"], "result_category": event["result_category"],
            }
            for event in ranked
        ],
        "event_credit_state": copy.deepcopy(credit or {"available": False}),
        "instruction": (
            "Complete the entire visible task in a fresh environment. Treat prior structures as hypotheses. "
            "Re-query facts with legal tools; never replay old identifiers, values, results, or calls. "
            "Preserve positively credited structure, avoid negatively credited structure, recover from public "
            "failures, verify consequential effects, and stop only after all visible requirements are handled."
        ),
        "public_only": True,
        "outcome_used": False,
    }


def deepplanning_frontier(candidate: dict[str, Any], credit: dict[str, Any] | None) -> dict[str, Any]:
    from deepplanning_adapter import adapter_frontier
    return adapter_frontier(candidate, credit)


def material_disagreement(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_shape = [(e["tool"], e["argument_keys"], e["result_category"]) for e in left["events"]]
    right_shape = [(e["tool"], e["argument_keys"], e["result_category"]) for e in right["events"]]
    return left_shape != right_shape or left["final_answer"].strip() != right["final_answer"].strip()


def select_and_credit(
    client: OpenAI, left: dict[str, Any], right: dict[str, Any], ordinal: int, seed: int,
    adapter: bool = False,
) -> tuple[int, dict[str, Any]]:
    order = [0, 1]
    random.Random(seed + ordinal).shuffle(order)
    originals = [left, right]
    shown = []
    allowed: dict[int, set[str]] = {}
    for shown_index, original_index in enumerate(order):
        candidate = copy.deepcopy(originals[original_index])
        remapped = []
        for index, event in enumerate(candidate["events"]):
            event = copy.deepcopy(event)
            event["event_id"] = f"C{shown_index}-E{index:04d}"
            remapped.append(event)
        candidate["events"] = remapped
        if adapter:
            from deepplanning_adapter import public_plan_evidence
            candidate["deepplanning_public_evidence"] = public_plan_evidence(candidate)
        candidate["candidate_index"] = shown_index
        shown.append(candidate)
        allowed[shown_index] = {event["event_id"] for event in remapped}
    try:
        response = client.chat.completions.create(
            model=API_MODEL,
            messages=[{"role": "system", "content": SELECT_SYSTEM + (DEEPPLANNING_SELECT_ADDENDUM if adapter else "")},
                      {"role": "user", "content": canonical({"candidates": shown})}],
            tools=[SELECT_TOOL],
            tool_choice={"type": "function", "function": {"name": "select_and_credit_events"}},
            temperature=0,
            max_tokens=16384,
            stream=True,
            extra_body={
                "seed": seed + ordinal,
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 16384,
            },
            timeout=600,
        )
        calls = collect_stream_tool_calls(response)
        arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        selected_shown = int(arguments.get("candidate_index", -1))
        if selected_shown not in (0, 1):
            raise ValueError("invalid candidate index")
        rejected_shown = 1 - selected_shown
        supporting = list(arguments.get("supporting_event_ids") or [])
        preserve = list(arguments.get("preserve_event_ids") or [])
        avoid = list(arguments.get("avoid_event_ids") or [])
        if not supporting or not set(supporting + preserve).issubset(allowed[selected_shown]):
            raise ValueError("invalid selected-candidate event provenance")
        if not set(avoid).issubset(allowed[rejected_shown]):
            raise ValueError("invalid rejected-candidate event provenance")
        issue_type = str(arguments.get("unresolved_issue_type") or "NONE")
        if issue_type not in ISSUE_TYPES:
            raise ValueError("invalid issue type")
        selected_original = order[selected_shown]
        credit = {
            "available": True,
            "preserve_events": [event for event in shown[selected_shown]["events"] if event["event_id"] in preserve][:6],
            "avoid_events": [event for event in shown[rejected_shown]["events"] if event["event_id"] in avoid][:6],
            "unresolved_issue_type": issue_type,
            "unresolved_issue": ISSUE_DIRECTIVES[issue_type],
            "reason": str(arguments.get("reason") or ""),
            "supporting_event_ids": supporting,
            "public_only": True,
            "outcome_used": False,
        }
        return selected_original, {"fallback": False, "selected": selected_original, "credit": credit}
    except Exception as error:
        return 0, {
            "fallback": True, "selected": 0, "fallback_reason": str(error),
            "credit": {"available": False, "public_only": True, "outcome_used": False},
        }


def install_seed(module: Any, seed: int) -> None:
    original = module.load_model_config

    def seeded(model_name: str) -> dict[str, Any]:
        config = dict(original(model_name))
        extra = dict(config.get("extra_body") or {})
        extra["seed"] = seed
        config["extra_body"] = extra
        return config

    module.load_model_config = seeded


def load_shopping_anchor(root: Path, anchor_tag: str, level: int, case_id: int) -> tuple[str, list[dict[str, Any]], Path]:
    source = root / "shoppingplanning" / "database_infered" / f"database_{anchor_tag}_L{level}" / f"case_{case_id}"
    payload = json.loads((source / "messages.json").read_text(encoding="utf-8"))
    query_rows = json.loads((root / "shoppingplanning" / f"data/level_{level}_query_meta.json").read_text(encoding="utf-8"))
    query = next(str(row["query"]) for row in query_rows if int(row["id"]) == case_id)
    return query, message_rows(payload), source


def run_shopping_task(args: argparse.Namespace, level: int, case_id: int, ordinal: int) -> dict[str, Any]:
    output = args.output / "shopping" / "tasks" / f"L{level}_{case_id}.json"
    if output.exists():
        return {"task": f"shopping::L{level}::{case_id}", "status": "existing"}
    shopping = args.root / "shoppingplanning"
    sys.path.insert(0, str(shopping))
    from agent.prompts import prompt_lib
    from agent.shopping_agent import ShoppingFnAgent
    query, anchor_messages, anchor_source = load_shopping_anchor(args.root, args.anchor_tag, level, case_id)
    incumbent = public_candidate(query, anchor_messages, "A-E")
    incumbent_source = anchor_source
    incumbent_origin = "A"
    credit: dict[str, Any] | None = None
    stages = []
    for stage_number, stage in enumerate(("B", "C", "D"), 1):
        checkpoint = args.output / "shopping" / "checkpoints" / f"L{level}_{case_id}_{stage}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            incumbent = row["accepted_candidate"]
            incumbent_source = Path(row["accepted_source"])
            incumbent_origin = row["accepted_origin"]
            credit = row["selector"]["credit"]
            stages.append(row)
            continue
        frontier = structural_frontier(incumbent, credit)
        stage_root = args.output / "shopping" / "stage_databases" / stage / f"L{level}"
        case_dir = stage_root / f"case_{case_id}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        if not case_dir.exists():
            case_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(shopping / f"database_level{level}" / f"case_{case_id}", case_dir)
        agent = ShoppingFnAgent(
            model=MODEL, sample_id=str(case_id), database_base_path=str(stage_root),
            tool_schema_path=str(shopping / "tools/shopping_tool_schema.json"),
        )
        prompt = (
            f"{getattr(prompt_lib, f'SYSTEM_PROMPT_level{level}')}\n\n"
            f"StateTrace-EDS-ECA round {stage_number + 1}.\n{canonical(frontier)}"
        )
        messages = agent.run(
            user_query=query, system_prompt=prompt, save_messages=True,
            sample_id=str(case_id), max_llm_calls=args.max_llm_calls,
        )
        proposal = public_candidate(query, message_rows(messages), f"{stage}'-E")
        if material_disagreement(incumbent, proposal):
            selected, selector = select_and_credit(
                args.selector_client, incumbent, proposal, ordinal * 3 + stage_number, args.selector_seed,
            )
        else:
            selected, selector = 0, {
                "fallback": True, "selected": 0, "fallback_reason": "no_material_disagreement",
                "credit": {"available": False, "public_only": True, "outcome_used": False},
            }
        if selected == 1:
            incumbent, incumbent_source, incumbent_origin = proposal, case_dir, f"{stage}'"
        credit = selector["credit"]
        row = {
            "stage": stage, "proposal_source": str(case_dir), "proposal_candidate": proposal,
            "accepted_source": str(incumbent_source), "accepted_origin": incumbent_origin,
            "accepted_candidate": incumbent, "selector": selector, "frontier": frontier,
        }
        write_json(checkpoint, row)
        stages.append(row)
    final_case = args.output / "shopping" / "final_database" / f"case_L{level}_{case_id}"
    if not final_case.exists():
        final_case.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(incumbent_source, final_case)
    write_json(output, {
        "task": f"shopping::L{level}::{case_id}", "anchor_origin": "A",
        "final_origin": incumbent_origin, "final_source": str(final_case), "stages": stages,
        "method": "StateTrace-EDS-ECA", "public_only": True, "online_outcome_used": False,
    })
    return {"task": f"shopping::L{level}::{case_id}", "status": "ok", "final_origin": incumbent_origin}


def load_travel_anchor(root: Path, anchor_tag: str, anchor_model_slug: str, language: str, task_id: int) -> tuple[str, list[dict[str, Any]], Path]:
    source = root / "travel_runs" / anchor_tag / f"{anchor_model_slug}_{language}" / "trajectories" / f"id_{task_id}.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    return str(payload["query"]), message_rows(payload), source


def run_travel_task(args: argparse.Namespace, language: str, task_id: int, ordinal: int) -> dict[str, Any]:
    output = args.output / f"travel_{language}" / "tasks" / f"id_{task_id}.json"
    if output.exists():
        return {"task": f"travel-{language}::{task_id}", "status": "existing"}
    travel = args.root / "travelplanning"
    sys.path.insert(0, str(travel))
    from agent.prompts import get_system_prompt
    from agent.tools_fn_agent import ToolsFnAgent
    query, anchor_messages, anchor_source = load_travel_anchor(
        args.root, args.anchor_tag, args.anchor_model_slug, language, task_id,
    )
    incumbent = public_candidate(query, anchor_messages, "A-E")
    incumbent_source = anchor_source
    incumbent_origin = "A"
    credit: dict[str, Any] | None = None
    stages = []
    for stage_number, stage in enumerate(("B", "C", "D"), 1):
        checkpoint = args.output / f"travel_{language}" / "checkpoints" / f"id_{task_id}_{stage}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            incumbent = row["accepted_candidate"]
            incumbent_source = Path(row["accepted_source"])
            incumbent_origin = row["accepted_origin"]
            credit = row["selector"]["credit"]
            stages.append(row)
            continue
        frontier = structural_frontier(incumbent, credit)
        if args.deepplanning_adapter:
            frontier["deepplanning_public_evidence"] = deepplanning_frontier(incumbent, credit)
        stage_path = args.output / f"travel_{language}" / "stage_trajectories" / stage / f"id_{task_id}.json"
        if stage_path.exists():
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
            messages = message_rows(payload)
        else:
            agent = ToolsFnAgent(
                model=MODEL, sample_id=str(task_id),
                database_base_path=str(travel / "database" / f"database_{language}"), language=language,
            )
            prompt = (
                f"{get_system_prompt(language)}\n\nStateTrace-EDS-ECA round {stage_number + 1}.\n"
                f"{canonical(frontier)}"
            )
            final, raw_messages = agent.run(query, system_prompt=prompt, max_llm_calls=args.max_llm_calls)
            messages = agent._serialize_messages(raw_messages)
            write_json(stage_path, {
                "id": f"id_{task_id}", "query": query, "final_plan": final,
                "messages": messages, "language": language, "model": MODEL,
                "method": (
                    f"StateTrace-EDS-ECA-DeepPlanning-{stage}-proposal"
                    if args.deepplanning_adapter else f"StateTrace-EDS-ECA-{stage}-proposal"
                ),
            })
        proposal = public_candidate(query, messages, f"{stage}'-E")
        if material_disagreement(incumbent, proposal):
            selected, selector = select_and_credit(
                args.selector_client, incumbent, proposal, ordinal * 3 + stage_number, args.selector_seed,
                adapter=args.deepplanning_adapter,
            )
        else:
            selected, selector = 0, {
                "fallback": True, "selected": 0, "fallback_reason": "no_material_disagreement",
                "credit": {"available": False, "public_only": True, "outcome_used": False},
            }
        if selected == 1:
            incumbent, incumbent_source, incumbent_origin = proposal, stage_path, f"{stage}'"
        credit = selector["credit"]
        row = {
            "stage": stage, "proposal_source": str(stage_path), "proposal_candidate": proposal,
            "accepted_source": str(incumbent_source), "accepted_origin": incumbent_origin,
            "accepted_candidate": incumbent, "selector": selector, "frontier": frontier,
        }
        write_json(checkpoint, row)
        stages.append(row)
    final_path = args.output / f"travel_{language}" / "final" / "trajectories" / f"id_{task_id}.json"
    if not final_path.exists():
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(incumbent_source, final_path)
        reports = final_path.parent.parent / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        reports.joinpath(f"id_{task_id}.txt").write_text(incumbent["final_answer"], encoding="utf-8")
    write_json(output, {
        "task": f"travel-{language}::{task_id}", "anchor_origin": "A",
        "final_origin": incumbent_origin, "final_source": str(final_path), "stages": stages,
        "method": (
            "StateTrace-EDS-ECA-DeepPlanning" if args.deepplanning_adapter else "StateTrace-EDS-ECA"
        ),
    })
    return {"task": f"travel-{language}::{task_id}", "status": "ok", "final_origin": incumbent_origin}


def jobs(args: argparse.Namespace) -> list[tuple[Any, ...]]:
    if args.cohort == "shopping":
        return [(level, case_id) for level, count in ((1, 50), (2, 50), (3, 20)) for case_id in range(1, count + 1)]
    language = args.cohort.rsplit("-", 1)[1]
    anchor_dir = args.root / "travel_runs" / args.anchor_tag / f"{args.anchor_model_slug}_{language}" / "trajectories"
    return [(language, task_id) for task_id in range(120) if (anchor_dir / f"id_{task_id}.json").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("shopping", "travel-zh", "travel-en"), required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=400)
    parser.add_argument("--proposal-seed", type=int, default=64639)
    parser.add_argument("--selector-seed", type=int, default=77113)
    parser.add_argument("--anchor-tag", default="full_vanilla_a_20260828",
                        help="Vanilla anchor tag: shopping run tag or travel_runs subdirectory.")
    parser.add_argument("--anchor-model-slug", default="mimo-v2.5-pro",
                        help="Filesystem-safe model slug under a travel anchor tag.")
    parser.add_argument("--deepplanning-adapter", action="store_true",
                        help="Add public DeepPlanning schema/grounding evidence to travel selection.")
    parser.add_argument("--job-shard-count", type=int, default=1)
    parser.add_argument("--job-shard-index", type=int, default=0)
    args = parser.parse_args()
    api_key_env = os.environ.get("DEEPPLANNING_API_KEY_ENV", "MIMO_API_KEY")
    api_key = os.environ.get(api_key_env)
    if args.workers < 1 or not api_key:
        raise ValueError(f"positive workers and {api_key_env} are required")
    args.selector_client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPPLANNING_BASE_URL", "https://api.xiaomimimo.com/v1"),
        max_retries=2,
        timeout=60.0,
    )
    domain_root = args.root / ("shoppingplanning" if args.cohort == "shopping" else "travelplanning")
    sys.path.insert(0, str(domain_root))
    import agent.call_llm as call_llm_module
    install_seed(call_llm_module, args.proposal_seed)
    all_jobs = jobs(args)
    if len(all_jobs) != 120:
        raise RuntimeError(f"{args.cohort} anchor coverage is {len(all_jobs)}/120")
    if args.job_shard_count < 1 or not 0 <= args.job_shard_index < args.job_shard_count:
        raise ValueError("job shard index must be within the positive shard count")
    selected_jobs = list(enumerate(all_jobs))[args.job_shard_index::args.job_shard_count]
    runner = run_shopping_task if args.cohort == "shopping" else run_travel_task
    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(runner, args, *job, ordinal): job
            for ordinal, job in selected_jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except BaseException as error:
                row = {"task": canonical(job), "status": "error", "error": str(error), "traceback": traceback.format_exc()[-6000:]}
                error_path = args.output / args.cohort.replace("-", "_") / "errors" / f"{canonical(job).replace('/', '_')}.json"
                write_json(error_path, row)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "method": (
            "StateTrace-EDS-ECA-DeepPlanning" if args.deepplanning_adapter else "StateTrace-EDS-ECA"
        ), "cohort": args.cohort, "full_cohort_test": True,
        "tasks": len(selected_jobs), "ok": sum(row["status"] in {"ok", "existing"} for row in results),
        "errors": sum(row["status"] == "error" for row in results), "workers": args.workers,
        "job_shard_count": args.job_shard_count, "job_shard_index": args.job_shard_index,
        "proposal_seed": args.proposal_seed, "selector_seed": args.selector_seed,
        "elapsed_seconds": round(time.monotonic() - started, 3), "public_only": True,
        "online_outcome_used": False, "results": results,
    }
    summary_name = (
        "run_summary.json" if args.job_shard_count == 1
        else f"run_summary_shard_{args.job_shard_index}_of_{args.job_shard_count}.json"
    )
    write_json(args.output / args.cohort.replace("-", "_") / summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
