from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.appworld_mimo import build_appworld_react_messages, calculate_appworld_loop_metrics, extract_python_code
from src.appworld_state_trace_eds_eca_adapter import (
    appworld_selector_packet,
    compile_appworld_public_events,
    appworld_event_frontier,
    appworld_public_disagreement,
    namespace_appworld_events,
)
from src.appworld_state_trace_eds_eca import (
    build_credit_state, parse_credit_decision, proposal_messages, public_event_ids,
    selector_messages,
)
from src.config import ROOT_DIR, get_settings
from src.llm_client import LLMClient

DEFAULT_APPWORLD_ROOT = ROOT_DIR / "runtime"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def settings(seed: int):
    return replace(get_settings(), seed=seed)


def run_react(
    world: Any,
    llm: LLMClient,
    messages: list[dict[str, Any]],
    max_steps: int,
    progress_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    trace, termination = [], "max_steps"
    for step in range(1, max_steps + 1):
        result = llm.complete_messages_with_trace(messages)
        code = extract_python_code(result.text)
        output = world.execute(code) if code else "No executable Python code block found."
        trace.append({"step": step, "assistant_text": result.text, "code": code, "execution_output": output, "llm_request": result.request, "llm_response": result.response})
        if progress_path is not None:
            write_json(progress_path, {"step": step, "updated_at": time.time()})
        messages.append(copy.deepcopy(result.message))
        messages.append({"role": "user", "content": f"Output:\n```\n{output}\n```\n\nContinue with one Python code block."})
        if world.task_completed():
            termination = "task_completed"
            break
    return trace, termination


def run_candidate(job: dict[str, Any], task_id: str, stage: str, seed: int, messages: list[dict[str, Any]]) -> dict[str, Any]:
    os.environ["APPWORLD_ROOT"] = job["appworld_root"]
    from appworld import AppWorld
    started = time.perf_counter()
    with AppWorld(task_id=task_id, experiment_name=f"{job['run_name']}_{stage}", timeout_seconds=None) as world:
        progress_path = Path(job["output_dir"]) / task_id / f"progress_{stage.lower()}.json"
        trace, termination = run_react(
            world,
            LLMClient(settings(seed)),
            messages,
            job["max_steps"],
            progress_path,
        )
        progress_path.unlink(missing_ok=True)
        evaluation = world.evaluate().to_dict()
        return {
            "task_id": task_id, "stage": stage, "seed": seed,
            "termination_reason": termination, "task_completed": world.task_completed(),
            "evaluation_success": bool(evaluation.get("success")), "evaluation": evaluation,
            "trace": trace, "public_events": compile_appworld_public_events(trace),
            "loop_metrics": calculate_appworld_loop_metrics(trace),
            "elapsed_seconds": time.perf_counter() - started,
        }


def choose(
    task_instruction: str,
    incumbent: dict[str, Any],
    proposal: dict[str, Any],
    seed: int,
    selector_max_completion_tokens: int,
    task_ordinal: int,
    stage_index: int,
    prior_disagreement: dict[str, Any] | None,
    prior_credit: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    disagreement = appworld_public_disagreement(incumbent["public_events"], proposal["public_events"])
    if not disagreement["material_disagreement"]:
        return incumbent, {"available": False, "preserve_events": [], "avoid_events": [], "unresolved_issue_type": "NONE", "active_intervention": None, "public_only": True}, {
            "choice": "A", "fallback": True, "reason": "no material public disagreement", "selector_skipped": True,
            "public_disagreement": disagreement,
        }
    display_order = ["incumbent", "proposal"]
    if (seed + task_ordinal * 3 + stage_index) % 2:
        display_order.reverse()
    displayed = [incumbent if x == "incumbent" else proposal for x in display_order]
    display_labels = ["A", "B"]
    display_events = [namespace_appworld_events(item["public_events"], label) for item, label in zip(displayed, display_labels)]
    packet = appworld_selector_packet(task_instruction, display_events[0], display_events[1])
    started = time.perf_counter()
    decision_seed = seed + task_ordinal * 3 + stage_index
    selector_settings = settings(decision_seed)
    max_token_field = "max_completion_tokens" if selector_settings.model_name.startswith("mimo-") else "max_tokens"
    selector_limit = min(selector_settings.max_completion_tokens, selector_max_completion_tokens)
    selector_payload = {max_token_field: selector_limit, "thinking": {"type": "disabled"}}
    # The NVIDIA proxy can stall on server-side JSON schema enforcement; the
    # existing parser already validates and falls back safely client-side.
    if "integrate.api.nvidia.com" not in selector_settings.base_url:
        selector_payload["response_format"] = {"type": "json_object"}
    result = LLMClient(selector_settings).complete_messages_with_trace(
        selector_messages(packet), extra_payload=selector_payload
    )
    decision = parse_credit_decision(result.text, public_event_ids(display_events[0]), public_event_ids(display_events[1]))
    selected_display = displayed[0] if decision.choice == "A" else displayed[1]
    selected = selected_display
    credit = build_credit_state(display_events[0], display_events[1], decision)
    details = {
        "choice": decision.choice, "fallback": decision.fallback, "reason": decision.reason,
        "supporting_event_ids": list(decision.supporting_event_ids),
        "preserve_event_ids": list(decision.preserve_event_ids), "avoid_event_ids": list(decision.avoid_event_ids),
        "unresolved_issue_type": decision.unresolved_issue_type,
        "request": result.request, "response": result.response,
        "public_disagreement": disagreement, "display_order": display_order,
        "decision_seed": decision_seed,
        "selected_candidate": "incumbent" if selected_display is incumbent else "proposal",
        "elapsed_seconds": time.perf_counter() - started,
    }
    return selected, credit, details


def run_task(job: dict[str, Any]) -> dict[str, Any]:
    task_id, task_dir = job["task_id"], Path(job["output_dir"]) / job["task_id"]
    final_path = task_dir / "final.json"
    if final_path.exists():
        return {"task_id": task_id, "status": "reused", "path": str(final_path)}
    try:
        os.environ["APPWORLD_ROOT"] = job["appworld_root"]
        from appworld.task import Task
        task = Task.load(task_id)
        base_messages = build_appworld_react_messages(task)
        a_path = task_dir / "candidate_a.json"
        incumbent = read_json(a_path) if a_path.exists() else run_candidate(job, task_id, "A", job["seed_a"], copy.deepcopy(base_messages))
        write_json(a_path, incumbent)
        credit: dict[str, Any] = {"available": False, "preserve_events": [], "avoid_events": [], "unresolved_issue_type": "NONE", "active_intervention": None, "public_only": True}
        prior_disagreement: dict[str, Any] | None = None
        stages = []
        for stage_index, (stage, seed) in enumerate(zip(("B", "C", "D"), job["proposal_seeds"]), start=1):
            proposal_path = task_dir / f"candidate_{stage.lower()}.json"
            checkpoint_path = task_dir / f"checkpoint_{stage.lower()}.json"
            frontier = appworld_event_frontier(incumbent["public_events"], prior_disagreement, credit)
            proposal_prompt = proposal_messages(base_messages, {"frontier": frontier, "event_credit": credit}, stage)
            proposal = read_json(proposal_path) if proposal_path.exists() else run_candidate(job, task_id, stage, seed, proposal_prompt)
            write_json(proposal_path, proposal)
            if checkpoint_path.exists():
                checkpoint = read_json(checkpoint_path)
                selection = checkpoint["selection"]
                credit = checkpoint["event_credit"]
                incumbent = incumbent if selection.get("selected_candidate", "incumbent") == "incumbent" else proposal
                prior_disagreement = selection.get("public_disagreement")
            else:
                incumbent, credit, selection = choose(
                    task.instruction,
                    incumbent,
                    proposal,
                    job["selector_seed"],
                    job["selector_max_completion_tokens"],
                    job["task_ordinal"], stage_index, prior_disagreement, credit,
                )
                checkpoint = {"stage": stage, "proposal_success": proposal["evaluation_success"], "accepted_success": incumbent["evaluation_success"], "selection": selection, "event_credit": credit}
                write_json(checkpoint_path, checkpoint)
            stages.append(checkpoint)
            prior_disagreement = selection.get("public_disagreement") if isinstance(selection, dict) else None
        result = {
            "schema_version": "appworld_state_trace_eds_eca_task_v3_faithful", "task_id": task_id,
            "dataset": job["dataset"], "model": get_settings().model_name,
            "seeds": {"a": job["seed_a"], "proposals": job["proposal_seeds"], "selector": job["selector_seed"]},
            "vanilla_success": read_json(a_path)["evaluation_success"],
            "selected_success": incumbent["evaluation_success"], "stages": stages,
        }
        write_json(final_path, result)
        (task_dir / "error.json").unlink(missing_ok=True)
        return {"task_id": task_id, "status": "completed", "path": str(final_path)}
    except Exception as exc:
        error = {"task_id": task_id, "status": "error", "exception": type(exc).__name__, "traceback": traceback.format_exc()}
        write_json(task_dir / "error.json", error)
        return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appworld-root", type=Path, default=DEFAULT_APPWORLD_ROOT)
    parser.add_argument("--dataset", choices=["train", "dev", "test_normal", "test_challenge"], default="test_normal")
    parser.add_argument("--task-ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed-a", type=int, default=53403)
    parser.add_argument("--proposal-seeds", default="64639,64640,64641")
    parser.add_argument("--selector-seed", type=int, default=77113)
    parser.add_argument("--selector-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="appworld_state_trace_eds_eca")
    args = parser.parse_args()
    args.appworld_root = args.appworld_root.resolve()
    os.environ["APPWORLD_ROOT"] = str(args.appworld_root)
    from appworld import load_task_ids
    task_ids = load_task_ids(args.dataset)
    if args.task_ids:
        requested = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        unknown = sorted(set(requested) - set(task_ids))
        if unknown:
            raise SystemExit(f"Unknown task IDs: {unknown}")
        task_ids = requested
    if args.limit is not None:
        task_ids = task_ids[:args.limit]
    proposal_seeds = [int(x) for x in args.proposal_seeds.split(",")]
    if len(proposal_seeds) != 3:
        raise SystemExit("--proposal-seeds requires exactly three comma-separated integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "appworld_state_trace_eds_eca_run_v3_faithful", "dataset": args.dataset, "task_count": len(task_ids), "model": get_settings().model_name, "seeds": {"a": args.seed_a, "proposals": proposal_seeds, "selector": args.selector_seed}, "max_steps": args.max_steps, "selector_max_completion_tokens": args.selector_max_completion_tokens, "event_adapter": "appworld_eds_eca_v3_faithful", "official_environment": True, "official_evaluator": True}
    write_json(args.output_dir / "manifest.json", manifest)
    common = {"appworld_root": str(args.appworld_root), "dataset": args.dataset, "output_dir": str(args.output_dir.resolve()), "run_name": args.run_name, "seed_a": args.seed_a, "proposal_seeds": proposal_seeds, "selector_seed": args.selector_seed, "selector_max_completion_tokens": args.selector_max_completion_tokens, "max_steps": args.max_steps}
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, {**common, "task_id": task_id, "task_ordinal": ordinal}) for ordinal, task_id in enumerate(task_ids)]
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(json.dumps(row, ensure_ascii=True), flush=True)
    finals = [read_json(Path(row["path"])) for row in results if row.get("status") in {"completed", "reused"}]
    summary = {**manifest, "completed": len(finals), "errors": sum(row.get("status") == "error" for row in results), "vanilla_success": sum(row["vanilla_success"] for row in finals), "eds_eca_success": sum(row["selected_success"] for row in finals)}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
