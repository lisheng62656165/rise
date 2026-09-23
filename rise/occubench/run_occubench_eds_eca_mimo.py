#!/usr/bin/env python3
"""OccuBench StateTrace-EDS-ECA runner with a MIMO agent.

Each stage runs in a fresh LWM environment. The primary selector follows the
original StateBench EDS-ECA protocol: an LLM compares source-blind public
trajectories and assigns event credit. Verifier outcomes are never exposed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import run_occubench_base as base
from occubench_eds_eca_selector import (
    SELECTOR_INSTRUCTION,
    build_selector_packet,
    credit_guidance,
    resolve_selector_response,
    selector_tool,
)
from occubench_statebench_eds_eca import (
    SELECT_INSTRUCTION as STATEBENCH_EDS_ECA_INSTRUCTION,
    analyze_stage as analyze_statebench_stage,
    build_credit_state as build_statebench_credit_state,
    build_generation_guidance as build_statebench_generation_guidance,
    build_selector_packet as build_statebench_selector_packet,
    compare_stages as compare_statebench_stages,
    openai_event_credit_tool,
    resolve_selector as resolve_statebench_selector,
    statebench_tool_schemas,
)


STAGES = ("A", "B", "C", "D")
RISK_WORDS = (
    "error", "failed", "failure", "timeout", "rejected", "denied",
    "not found", "invalid", "try again", "unable", "missing",
)
VERIFY_WORDS = ("verify", "verified", "confirm", "confirmed", "check", "read back", "status")
COMPLETION_WORDS = (
    "task complete", "task completed", "successfully finalized", "certified safe",
    "final status", "session closed", "session finalized",
)


def public_events(trajectory: str) -> Dict[str, Any]:
    """Extract coarse, provenance-preserving events from public text only."""
    actions: List[str] = []
    observations: List[str] = []
    responses: List[str] = []
    for line in trajectory.splitlines():
        line = line.strip()
        if line.startswith("[Agent Action]"):
            actions.append(line[:280])
        elif line.startswith("[Environment Observation]"):
            observations.append(line[:420])
        elif line.startswith("[Agent Response]"):
            responses.append(line[:600])
    failures = [
        item for item in observations
        if any(word in item.lower() for word in RISK_WORDS)
    ]
    verification = [
        item for item in actions + observations
        if any(word in item.lower() for word in VERIFY_WORDS)
    ]
    completion = [
        item for item in responses + observations
        if any(word in item.lower() for word in COMPLETION_WORDS)
    ]
    return {
        "action_count": len(actions),
        "observation_count": len(observations),
        "failure_events": failures[-4:],
        "verification_events": verification[-4:],
        "completion_events": completion[-4:],
        "last_actions": actions[-6:],
    }


def evidence_score(trajectory: str) -> float:
    """A deterministic public-only score used for incumbent updates."""
    ledger = public_events(trajectory)
    # Action count is only a weak tie-breaker; longer trajectories are not
    # inherently better than concise, explicitly verified completion.
    score = 0.25 * float(ledger["action_count"])
    score += 1.5 * len(ledger["verification_events"])
    score += 6.0 * len(ledger["completion_events"])
    score -= 3.0 * len(ledger["failure_events"])
    if ledger["action_count"] == 0:
        score -= 10.0
    return score


def guidance(ledger: Dict[str, Any], credit: Dict[str, Any] | None = None) -> str:
    credit = credit or {}
    return (
        "\n\nPUBLIC EVENT GUIDANCE FROM THE PREVIOUS FRESH ROLLOUT:\n"
        "Treat this as a hypothesis, not ground truth. Preserve useful public evidence; "
        "re-ground every object and argument in the current environment.\n"
        f"Action events: {ledger['action_count']}; observations: {ledger['observation_count']}.\n"
        f"Public failure/risk events: {json.dumps(ledger['failure_events'], ensure_ascii=False)}\n"
        f"Public verification events: {json.dumps(ledger['verification_events'], ensure_ascii=False)}\n"
        f"Recent public actions: {json.dumps(ledger['last_actions'], ensure_ascii=False)}\n"
        "Recovery rules: do not repeat a rejected action unchanged; reacquire the target and "
        "arguments before mutation; verify consequential changes before stopping; avoid extra "
        "mutation after apparent success.\n"
        f"Accepted-comparison issue type: {credit.get('issue_type', 'NONE')}.\n"
        f"Accepted-comparison instruction: {credit.get('instruction', '')}\n"
        "Do not replay identifiers, values, or results from a prior fresh environment."
    )


def select_semantically(
    ns: argparse.Namespace,
    task: Dict[str, Any],
    incumbent: Dict[str, Any],
    proposal: Dict[str, Any],
    stage_index: int,
) -> tuple[int, Dict[str, Any]]:
    """Run the original EDS-ECA style public semantic comparison."""
    decision_seed = ns.selector_seed + int(task["task_id"]) * 3 + stage_index
    packet, order, allowed = build_selector_packet(
        task, incumbent, proposal, decision_seed,
    )
    key = os.environ.get(ns.selector_api_key_env, "")
    if not key:
        return 0, {
            "fallback": True,
            "fallback_reason": f"missing_selector_key:{ns.selector_api_key_env}",
        }
    client = base.make_client(
        key,
        ns.selector_base_url or ns.agent_base_url,
        base.parse_extra_body(
            ns.selector_extra_body_json
            or ns.agent_extra_body_json
            or os.environ.get("OCCUBENCH_SELECTOR_EXTRA_BODY_JSON", "")
        ),
    )
    try:
        semaphore = getattr(ns, "selector_semaphore", None)
        if semaphore is not None:
            semaphore.acquire()
        try:
            response = base.create_chat_completion_with_retry(
                client,
                max_retries=ns.selector_retries,
                retry_delay=ns.selector_retry_delay,
                model=ns.selector_model or ns.agent_model,
                messages=[
                    {"role": "system", "content": SELECTOR_INSTRUCTION},
                    {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
                ],
                tools=[selector_tool()],
                tool_choice={
                    "type": "function",
                    "function": {"name": "select_and_credit_events"},
                },
                max_tokens=ns.selector_max_tokens,
                seed=decision_seed,
            )
        finally:
            if semaphore is not None:
                semaphore.release()
        base.log_response(response, stage=f"semantic_selector_{stage_index}")
    except Exception as exc:
        return 0, {
            "fallback": True,
            "fallback_reason": "selector_call_error",
            "selector_error": f"{type(exc).__name__}: {exc}"[:600],
        }
    finally:
        transport = getattr(getattr(client, "_client", None), "close", None)
        if callable(transport):
            transport()
    return resolve_selector_response(response, order, allowed)


def select_with_statebench_eds_eca(
    ns: argparse.Namespace,
    task: Dict[str, Any],
    incumbent: Dict[str, Any],
    proposal: Dict[str, Any],
    schemas: List[Dict[str, Any]],
    stage_index: int,
) -> tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    """Run the original StateBench EDS-ECA event-credit selector contract."""
    decision_seed = ns.selector_seed + int(task["task_id"]) * 3 + stage_index
    packet, order, allowed, public_candidates = build_statebench_selector_packet(
        task, incumbent, proposal, schemas, decision_seed,
    )
    key = os.environ.get(ns.selector_api_key_env, "")
    if not key:
        raise RuntimeError(f"missing_selector_key:{ns.selector_api_key_env}")
    client = base.make_client(
        key,
        ns.selector_base_url or ns.agent_base_url,
        base.parse_extra_body(
            ns.selector_extra_body_json
            or ns.agent_extra_body_json
            or os.environ.get("OCCUBENCH_SELECTOR_EXTRA_BODY_JSON", "")
        ),
    )
    attempts = []
    selector_design = getattr(ns, "selector_design", "original")
    instruction = STATEBENCH_EDS_ECA_INSTRUCTION
    if selector_design == "atts-eca-pairwise-v1":
        from state_trace_pairwise_selector import INSTRUCTION
        instruction = INSTRUCTION
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    try:
        semaphore = getattr(ns, "selector_semaphore", None)
        if semaphore is not None:
            semaphore.acquire()
        try:
            for attempt in range(1 + getattr(ns, "selector_validation_retries", 2)):
                if attempt and len(messages[1]["content"]) > 200000:
                    from selector_public_text_refs import deduplicate_public_text
                    messages[1]["content"] = json.dumps(deduplicate_public_text(packet), ensure_ascii=False)
                response = base.create_chat_completion_with_retry(
                    client,
                    max_retries=ns.selector_retries,
                    retry_delay=ns.selector_retry_delay,
                    model=ns.selector_model or ns.agent_model,
                    messages=messages,
                    tools=[openai_event_credit_tool(2)],
                    tool_choice={
                        "type": "function",
                        "function": {"name": "select_and_credit_events"},
                    },
                    max_tokens=ns.selector_max_tokens,
                    seed=decision_seed + attempt,
                    **({"stream": False} if attempt else {}),
                )
                base.log_response(response, stage=f"statebench_eds_eca_selector_{stage_index}_attempt_{attempt}")
                selected, details = resolve_statebench_selector(response, order, allowed)
                attempts.append({"attempt": attempt + 1,
                                 "transport": "non_stream" if attempt else "configured",
                                 "fallback_reason": details.get("fallback_reason"),
                                 "valid": not details.get("fallback", False)})
                if not details.get("fallback", False):
                    details["validation_attempts"] = attempts
                    details["selector_design"] = selector_design
                    return selected, details, public_candidates
                messages.append({"role": "user", "content": (
                    "Your previous selection failed validation: "
                    + str(details.get("fallback_reason"))
                    + ". Reconsider the same candidates and call select_and_credit_events. "
                    "Provide every required field, a nonempty reason and supporting_event_ids. "
                    "Supporting and preserve IDs must belong to the selected candidate; "
                    "avoid IDs must belong to the rejected candidate. Empty preserve/avoid "
                    "lists are allowed when no justified credit exists. Use an allowed issue type "
                    "and describe the unresolved issue unless type is NONE. "
                    "Allowed event IDs by displayed candidate index: "
                    + json.dumps({str(k): sorted(v) for k, v in allowed.items()})
                )})
            raise RuntimeError("selector_validation_exhausted: " + json.dumps(attempts))
        finally:
            if semaphore is not None:
                semaphore.release()
    finally:
        transport = getattr(getattr(client, "_client", None), "close", None)
        if callable(transport):
            transport()


def make_args(ns: argparse.Namespace) -> argparse.Namespace:
    args = argparse.Namespace(
        env_mode=ns.env_mode,
        fault_count=ns.fault_count,
        fault_duration=ns.fault_duration,
        extra_body_json="",
        agent_extra_body_json=ns.agent_extra_body_json or os.environ.get("OCCUBENCH_AGENT_EXTRA_BODY_JSON", ""),
        world_extra_body_json=ns.world_extra_body_json or os.environ.get("OCCUBENCH_WORLD_EXTRA_BODY_JSON", ""),
        verifier_extra_body_json=ns.verifier_extra_body_json or os.environ.get("OCCUBENCH_VERIFIER_EXTRA_BODY_JSON", ""),
        agent_model=ns.agent_model,
        world_model=ns.world_model,
        verifier_model=ns.verifier_model,
        agent_reasoning_effort=None,
        agent_request_retries=ns.agent_request_retries,
        agent_retry_delay=ns.agent_retry_delay,
        persist_max_interventions=0,
        persist_min_steps_before_stop=0,
        persist_policy="strict_v1",
        persist_repeat_threshold=3,
        persist_max_length_continuations=0,
        max_steps=ns.max_steps,
        max_tokens=ns.max_tokens,
        verifier_votes=ns.verifier_votes,
        verifier_retries=ns.verifier_retries,
        agent_api_key=os.environ.get(ns.agent_api_key_env, ""),
        world_api_key=os.environ.get(ns.world_api_key_env, ""),
        verifier_api_key=os.environ.get(ns.verifier_api_key_env, ""),
        agent_base_url=ns.agent_base_url,
        world_base_url=ns.world_base_url,
        verifier_base_url=ns.verifier_base_url,
    )
    if not args.agent_api_key or not args.world_api_key or not args.verifier_api_key:
        raise SystemExit("Agent, world, and verifier API key environment variables are required")
    return args


def run_task(ns: argparse.Namespace, task: Dict[str, Any], root: Path,
             partial: Dict[str, Dict[str, Any]] | None = None,
             anchor: Dict[str, Any] | None = None) -> Dict[str, Any]:
    args = make_args(ns)
    registry = root / "data" / "world_model_configs"
    from occubench.lwm import WorldModelRegistry, LWMEnvironment
    from occubench.verifier import Verifier
    from occubench.fault_injection import build_fault_prompt

    modules = {"LWMEnvironment": LWMEnvironment, "Verifier": Verifier, "build_fault_prompt": build_fault_prompt}
    world_registry = WorldModelRegistry(str(registry))
    environment_config = world_registry.get(task["env_name"])
    sb_schemas = statebench_tool_schemas(environment_config)
    incumbent: Dict[str, Any] | None = copy.deepcopy(anchor) if anchor else None
    latest_credit: Dict[str, Any] = {}
    statebench_credit_state: Dict[str, Any] = {}
    statebench_prior_disagreement: Dict[str, Any] | None = None
    rows: List[Dict[str, Any]] = [copy.deepcopy(anchor)] if anchor else []
    if incumbent is not None:
        incumbent["accepted"] = True
        incumbent["selector_policy"] = "vanilla-anchor"
        incumbent["selector"] = {
            "fallback": False,
            "reason": "fixed_external_vanilla_anchor",
            "selected_original_index": 0,
        }
    for stage_index, stage in enumerate(ns.stages):
        if partial and stage in partial and not partial[stage].get("selection_pending"):
            row = partial[stage]
            rows.append(row)
            if row.get("accepted"):
                incumbent = row
            if row.get("selector_credit"):
                latest_credit = dict(row["selector_credit"])
            if ns.selector_policy == "statebench-eds-eca":
                statebench_credit_state = dict(
                    row.get("statebench_event_credit") or statebench_credit_state
                )
                statebench_prior_disagreement = (
                    dict(row["statebench_public_disagreement"])
                    if row.get("accepted") and row.get("statebench_public_disagreement")
                    else None
                )
            continue
        stage_task = copy.deepcopy(task)
        if incumbent is not None:
            if ns.selector_policy == "statebench-eds-eca":
                incumbent_ledger, generation_frontier, injected = (
                    build_statebench_generation_guidance(
                        task,
                        incumbent,
                        sb_schemas,
                        statebench_prior_disagreement,
                        statebench_credit_state,
                        stage_index + 1,
                        ns.max_event_groups,
                    )
                )
                stage_task["agent_instruction"] += "\n\n" + injected
            else:
                stage_task["agent_instruction"] += guidance(
                    public_events(incumbent["trajectory"]), latest_credit,
                )
        # Verification is only needed for the final incumbent.  Earlier
        # stages affect proposal guidance through public events, not labels.
        pending = (partial or {}).get(stage, {})
        row = copy.deepcopy(pending) if pending.get("selection_pending") else base.evaluate_one(
            args, stage_task, "raw", world_registry, modules,
            verify=(stage == ns.stages[-1]),
        )
        row["stage"] = stage
        row["run_id"] = ns.run_id
        row["numeric_seed"] = ns.seed
        row["public_event_ledger"] = public_events(row["trajectory"])
        row["selector_policy"] = ns.selector_policy
        # Save the expensive proposal before selection, so retries reuse it.
        if incumbent is not None and ns.selector_policy == "statebench-eds-eca":
            row["selection_pending"] = True
            callback = getattr(ns, "stage_callback", None)
            if callback is not None:
                callback(int(task["task_id"]), stage, copy.deepcopy(row))
        if incumbent is None:
            row["accepted"] = True
            row["selector"] = {
                "fallback": False,
                "reason": "vanilla_anchor",
                "selected_original_index": 1,
            }
            incumbent = row
        elif ns.selector_policy == "legacy-heuristic":
            row["public_evidence_score"] = evidence_score(row["trajectory"])
            incumbent_score = incumbent.get("public_evidence_score")
            if incumbent_score is None:
                incumbent_score = evidence_score(incumbent["trajectory"])
                incumbent["public_evidence_score"] = incumbent_score
            selected_index = int(row["public_evidence_score"] > incumbent_score)
            row["accepted"] = selected_index == 1
            row["selector"] = {
                "fallback": False,
                "reason": "legacy_keyword_evidence_score",
                "selected_original_index": selected_index,
            }
            if row["accepted"]:
                incumbent = row
        elif ns.selector_policy == "statebench-eds-eca":
            proposal_ledger = analyze_statebench_stage(task, row, sb_schemas)
            disagreement = compare_statebench_stages(
                task, incumbent, row, sb_schemas,
            )
            public_candidates = None
            if disagreement.get("material_disagreement"):
                selected_index, selector_details, public_candidates = (
                    select_with_statebench_eds_eca(
                        ns, task, incumbent, row, sb_schemas, stage_index,
                    )
                )
            else:
                selected_index = 0
                selector_details = {
                    "fallback": True,
                    "decision": None,
                    "fallback_reason": "no_material_public_transaction_disagreement",
                }
                _, _, _, public_candidates = build_statebench_selector_packet(
                    task, incumbent, row, sb_schemas,
                    ns.selector_seed + int(task["task_id"]) * 3 + stage_index,
                )
            row["selector"] = selector_details
            row["accepted"] = selected_index == 1
            statebench_credit_state = build_statebench_credit_state(
                public_candidates,
                selected_index,
                selector_details,
                sb_schemas,
                statebench_credit_state,
            )
            row["statebench_event_ledger"] = proposal_ledger
            row["statebench_generation_frontier"] = generation_frontier
            row["statebench_public_disagreement"] = disagreement
            row["statebench_event_credit"] = statebench_credit_state
            statebench_prior_disagreement = disagreement if row["accepted"] else None
            if row["accepted"]:
                incumbent = row
        else:
            selected_index, selector_details = select_semantically(
                ns, task, incumbent, row, stage_index,
            )
            row["selector"] = selector_details
            row["accepted"] = selected_index == 1
            latest_credit = credit_guidance(selector_details)
            row["selector_credit"] = latest_credit
            if row["accepted"]:
                incumbent = row
        row.pop("selection_pending", None)
        rows.append(row)
        callback = getattr(ns, "stage_callback", None)
        if callback is not None:
            callback(int(task["task_id"]), stage, row)
    assert incumbent is not None
    return {
        "task_id": int(task["task_id"]),
        "run_id": ns.run_id,
        "numeric_seed": ns.seed,
        "stages": rows,
        "final_stage": incumbent["stage"],
        "final_trajectory": incumbent["trajectory"],
        "final_is_correct": incumbent.get("is_correct"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=53403)
    p.add_argument("--agent-model", default="mimo-v2.5-pro")
    p.add_argument("--world-model", default="deepseek-v4-flash")
    p.add_argument("--verifier-model", default="deepseek-v4-flash")
    p.add_argument("--agent-base-url", required=True)
    p.add_argument("--world-base-url", required=True)
    p.add_argument("--verifier-base-url", required=True)
    p.add_argument("--agent-api-key-env", default="MIMO_API_KEY")
    p.add_argument("--world-api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--verifier-api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--selector-model")
    p.add_argument("--selector-base-url")
    p.add_argument("--selector-api-key-env", default="MIMO_API_KEY")
    p.add_argument("--selector-extra-body-json", default="")
    p.add_argument(
        "--selector-policy",
        choices=["semantic", "legacy-heuristic", "statebench-eds-eca"],
        default="semantic",
    )
    p.add_argument("--selector-seed", type=int, default=77113)
    p.add_argument("--selector-max-tokens", type=int, default=1200)
    p.add_argument("--selector-retries", type=int, default=2)
    p.add_argument("--selector-validation-retries", type=int, default=2)
    p.add_argument("--selector-design", choices=["original", "atts-eca-pairwise-v1"], default="original")
    p.add_argument("--selector-retry-delay", type=float, default=5.0)
    p.add_argument("--selector-active-requests", type=int, default=20)
    p.add_argument("--max-event-groups", type=int, default=12)
    p.add_argument("--agent-extra-body-json", default="")
    p.add_argument("--world-extra-body-json", default="")
    p.add_argument("--verifier-extra-body-json", default="")
    p.add_argument("--env-mode", choices=["E0", "E1", "E2", "E3"], default="E0")
    p.add_argument("--fault-count", type=int, default=2)
    p.add_argument("--fault-duration", type=int, default=2)
    p.add_argument("--task-ids", type=int, nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--agent-request-retries", type=int, default=3)
    p.add_argument("--agent-retry-delay", type=float, default=1.0)
    p.add_argument("--verifier-votes", type=int, default=3)
    p.add_argument("--verifier-retries", type=int, default=2)
    p.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    p.add_argument(
        "--anchor-results",
        default="",
        help="JSONL from a Vanilla A run. Reuse its stage-A trajectory as a fixed incumbent.",
    )
    p.add_argument("--resume", action="store_true", help="Skip task IDs already present in results.jsonl")
    p.add_argument("--fresh-stages", action="store_true", help="Ignore saved stage checkpoints for the selected tasks")
    ns = p.parse_args()
    random.seed(ns.seed)
    root = Path(ns.dataset_root).resolve()
    sys.path.insert(0, str(root))
    tasks = base.load_tasks(root, ns.task_ids, None, None)
    if ns.limit is not None:
        tasks = tasks[:ns.limit]
    anchors: Dict[int, Dict[str, Any]] = {}
    requested_ids = {int(task['task_id']) for task in tasks}
    if ns.anchor_results:
        anchor_path = Path(ns.anchor_results).resolve()
        if not anchor_path.exists():
            raise FileNotFoundError(f"anchor-results does not exist: {anchor_path}")
        with anchor_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = payload.get("task_id")
                if tid is None or int(tid) not in requested_ids:
                    continue
                # Accept either the runner's top-level result (with stages)
                # or a direct stage-A row exported by a scoring utility.
                if isinstance(payload.get("stages"), list):
                    candidates = [
                        row for row in payload["stages"]
                        if isinstance(row, dict) and row.get("stage") == "A"
                    ]
                    if candidates:
                        anchors[int(tid)] = copy.deepcopy(candidates[0])
                elif payload.get("stage") == "A":
                    anchors[int(tid)] = copy.deepcopy(payload)
        missing = [int(task["task_id"]) for task in tasks
                   if int(task["task_id"]) not in anchors]
        if missing:
            preview = ",".join(str(item) for item in missing[:12])
            raise ValueError(
                f"Missing Vanilla A anchors for {len(missing)} task(s): {preview}"
            )
        print(json.dumps({"anchor_results": str(anchor_path),
                          "anchors_loaded": len(anchors)}, ensure_ascii=False), flush=True)
    out = Path(ns.output_dir).resolve() / ns.run_id
    out.mkdir(parents=True, exist_ok=True)
    stage_path = out / "stage_checkpoints.jsonl"
    stage_records: Dict[tuple[int, str], Dict[str, Any]] = {}
    if stage_path.exists():
        with stage_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if int(item['task_id']) in requested_ids:
                        stage_records[(int(item["task_id"]), str(item["stage"]))] = item["row"]
                except (ValueError, KeyError, json.JSONDecodeError):
                    continue
    (out / "config.json").write_text(json.dumps(vars(ns), indent=2), encoding="utf-8")
    ns.selector_semaphore = threading.BoundedSemaphore(ns.selector_active_requests)
    if ns.resume:
        completed_ids = set()
        results_path = out / "results.jsonl"
        if results_path.exists():
            # The shared result file grows large during multi-process runs.
            # Stream it line-by-line so every worker does not materialize the
            # entire JSONL file and exhaust memory at startup.
            with results_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        completed_ids.add(int(json.loads(line)["task_id"]))
                    except (ValueError, KeyError, json.JSONDecodeError):
                        continue
        tasks = [task for task in tasks if int(task["task_id"]) not in completed_ids]
        print(json.dumps({"resume": True, "skipped_task_ids": len(completed_ids), "remaining_tasks": len(tasks)}), flush=True)
    stage_lock = threading.Lock()
    def persist_stage(task_id: int, stage: str, row: Dict[str, Any]) -> None:
        item = {"task_id": task_id, "stage": stage, "row": row}
        with stage_lock:
            with stage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
            stage_records[(task_id, stage)] = row
            print(json.dumps({"task_id": task_id, "stage": stage,
                              "checkpoint": True}), flush=True)
    ns.stage_callback = persist_stage

    def run_one(task: Dict[str, Any]) -> Dict[str, Any]:
        tid = int(task["task_id"])
        partial = {}
        if not ns.fresh_stages:
            partial = {
                stage: row
                for (saved_tid, stage), row in stage_records.items()
                if saved_tid == tid
                and row.get("agent_model") == ns.agent_model
                and row.get("world_model") == ns.world_model
                and row.get("verifier_model") == ns.verifier_model
                and row.get("selector_policy") == ns.selector_policy
            }
        anchor = anchors.get(tid)
        return run_task(ns, task, root, partial=partial, anchor=anchor)

    results: List[Dict[str, Any]] = []
    errors_path = out / "errors.jsonl"
    with ThreadPoolExecutor(max_workers=ns.workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                # Keep one transient API failure from aborting the whole batch.
                # The task remains absent from results.jsonl and will be retried
                # by the next --resume batch using its saved stage checkpoints.
                task = next((t for t, f in zip(tasks, futures) if f is future), None)
                error = {
                    "task_id": int(task["task_id"]) if task else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                with errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                    handle.flush()
                print(json.dumps({"task_id": error["task_id"], "error": error["error_type"]}), flush=True)
                continue
            results.append(row)
            with (out / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"task_id": row["task_id"], "final_stage": row["final_stage"]}), flush=True)
    (out / "summary.json").write_text(json.dumps({"rows": len(results), "run_id": ns.run_id}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
