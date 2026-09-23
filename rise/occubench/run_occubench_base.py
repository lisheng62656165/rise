#!/usr/bin/env python3
"""
Run OccuBench with paired Raw / PERSIST-ACE agent loops.

The strict controller observes only the online trajectory, audits every repeat
gate before tool execution, protects productive retries and dynamic polling,
and abstains when no grounded recovery card is available. A separate one-shot
card can continue an agent generation truncated by the token limit. Verification
rubrics and final outcomes are never visible to the controller.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from tqdm import tqdm

from occubench_persist_controller import StrictPersistController


EXECUTOR_SYSTEM_PROMPT = """You are a professional AI assistant executing tasks in a simulated environment.
You have access to a set of tools that interact with the environment.
Your goal is to complete the given task by making appropriate tool calls.

Task Scenario: {task_scenario_name}

Please remember the current actual time: {current_time}

IMPORTANT:
- Think step by step before acting.
- Use the available tools to gather information and take actions.
- Do not hallucinate tool responses - always call the tool to get real data.
- Complete the task as thoroughly as possible.
"""

EXECUTOR_SYSTEM_PROMPT_SHORT = """You execute the task in a simulated environment using the available tools.
Use real tool results, complete the requested state changes, and verify consequential changes before stopping.
Scenario: {task_scenario_name}. Current date: {current_time}.
"""


PERSIST_EARLY_STOP_PROMPT = """You appear to have stopped after very little environment interaction.
Continue the task if any required state, constraint, or confirmation is still missing.
Use available tools to gather or verify the necessary information before giving a final answer."""


PERSIST_ERROR_RECOVERY_PROMPT = """The latest environment observation indicates a recoverable tool or service failure.
Do not abandon the task solely because of this observation. Retry when appropriate, or use a different available tool to gather equivalent evidence, then continue toward task completion."""


PERSIST_REPEAT_PROMPT = """The last action repeated a previous tool call without adding visible new evidence.
Avoid spending more calls on the identical request. Use another available tool, a narrower query, or a state-verification action that can advance the task."""


PERSIST_LENGTH_CONTINUATION_PROMPT = """Your previous generation was truncated by the output token limit.
Continue exactly from the current task state. Do not restart the analysis or repeat completed tool calls. Use the available tools if further environment actions are required."""


EXPLICIT_ERROR_PATTERNS = [
    "http 500",
    "500 error",
    "timeout",
    "timed out",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "rate limit",
    "internal error",
    "exception",
    "try again",
]


@dataclass
class Intervention:
    step: int
    kind: str
    detail: str


class ExtraBodyClient:
    """Small proxy that injects extra_body into OpenAI-compatible calls."""

    class _Completions:
        def __init__(self, base: Any, extra_body: Optional[Dict[str, Any]]):
            self._base = base
            self._extra_body = extra_body

        def create(self, **kwargs: Any) -> Any:
            if self._extra_body:
                merged = dict(kwargs.get("extra_body") or {})
                merged.update(self._extra_body)
                kwargs["extra_body"] = merged
            return self._base.create(**kwargs)

    class _Chat:
        def __init__(self, base: Any, extra_body: Optional[Dict[str, Any]]):
            self.completions = ExtraBodyClient._Completions(base.completions, extra_body)

    def __init__(self, client: OpenAI, extra_body: Optional[Dict[str, Any]]):
        self._client = client
        self.chat = ExtraBodyClient._Chat(client.chat, extra_body)


def parse_extra_body(value: str) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    return json.loads(value)


def make_client(api_key: str, base_url: Optional[str], extra_body: Optional[Dict[str, Any]]) -> ExtraBodyClient:
    timeout = float(os.environ.get("OCCUBENCH_REQUEST_TIMEOUT", "45"))
    return ExtraBodyClient(
        OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0),
        extra_body,
    )


def resolve_role_runtime(args: argparse.Namespace) -> None:
    """Resolve per-role credentials and endpoints without changing legacy defaults."""
    for role in ("agent", "world", "verifier"):
        key_env = getattr(args, f"{role}_api_key_env", None)
        api_key = os.getenv(key_env) if key_env else args.api_key
        if not api_key:
            source = key_env or args.api_key_env
            raise SystemExit(f"Missing {role} API key. Set {source}.")
        setattr(args, f"{role}_api_key", api_key)
        setattr(
            args,
            f"{role}_base_url",
            getattr(args, f"{role}_base_url", None) or args.base_url,
        )


def create_chat_completion_with_retry(
    client: Any,
    *,
    max_retries: int,
    retry_delay: float,
    **kwargs: Any,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            if kwargs.get("stream", os.environ.get("OCCUBENCH_STREAM", "0") == "1"):
                kwargs = dict(kwargs)
                kwargs["stream"] = True
                return collect_stream_response(client.chat.completions.create(**kwargs))
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt >= max_retries:
                raise
            delay = retry_delay * (2 ** min(attempt, 4))
            print(
                f"Agent request failed ({type(exc).__name__}); "
                f"retry {attempt + 1}/{max_retries} in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def collect_stream_response(stream: Any) -> Any:
    """Reconstruct one completion from streamed text and tool-call deltas."""
    content: List[str] = []
    reasoning: List[str] = []
    tool_calls: Dict[int, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    usage: Any = None
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if getattr(delta, "content", None):
            content.append(delta.content)
        if getattr(delta, "reasoning_content", None):
            reasoning.append(delta.reasoning_content)
        for delta_call in getattr(delta, "tool_calls", None) or []:
            index = int(getattr(delta_call, "index", 0) or 0)
            item = tool_calls.setdefault(
                index,
                {"id": None, "type": "function", "name": [], "arguments": []},
            )
            if getattr(delta_call, "id", None):
                item["id"] = delta_call.id
            function = getattr(delta_call, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    item["name"].append(function.name)
                if getattr(function, "arguments", None):
                    item["arguments"].append(function.arguments)

    calls = []
    for index in sorted(tool_calls):
        item = tool_calls[index]
        calls.append(
            SimpleNamespace(
                id=item["id"],
                type=item["type"],
                function=SimpleNamespace(
                    name="".join(item["name"]),
                    arguments="".join(item["arguments"]),
                ),
            )
        )
    message = SimpleNamespace(
        content="".join(content) or None,
        reasoning_content="".join(reasoning) or None,
        reasoning="".join(reasoning) or None,
        tool_calls=calls or None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


_RESPONSE_LOG_LOCK = threading.Lock()


def log_response(response: Any, *, stage: str = "") -> None:
    """Write a compact raw-response audit record when a log path is configured."""
    path = os.environ.get("OCCUBENCH_RESPONSE_LOG")
    if not path:
        return
    choice = response.choices[0]
    message = choice.message
    record = {
        "stage": stage,
        "finish_reason": getattr(choice, "finish_reason", None),
        "content": getattr(message, "content", None),
        "reasoning": getattr(message, "reasoning", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": [
            {
                "id": getattr(call, "id", None),
                "name": getattr(getattr(call, "function", None), "name", None),
                "arguments": getattr(getattr(call, "function", None), "arguments", None),
            }
            for call in (getattr(message, "tool_calls", None) or [])
        ],
        "usage": str(getattr(response, "usage", None)),
    }
    with _RESPONSE_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def strict_majority_verification(
    verifier: Any,
    *,
    task_scenario_name: str,
    task_initial_state: str,
    state_description: str,
    agent_instruction: str,
    verification_plan: str,
    trajectory: str,
    max_retries: int,
) -> Dict[str, Any]:
    """Run all verifier votes without treating request/parse failures as false."""

    vote_args = (
        task_scenario_name,
        task_initial_state,
        state_description,
        agent_instruction,
        verification_plan,
        trajectory,
    )

    def run_vote() -> Dict[str, Any]:
        invalid_attempts = 0
        for _ in range(max_retries + 1):
            result = verifier._single_check(*vote_args)
            if result.get("feedback") != "Verification error":
                return {"result": result, "invalid_attempts": invalid_attempts}
            invalid_attempts += 1
        return {"result": None, "invalid_attempts": invalid_attempts}

    with ThreadPoolExecutor(max_workers=verifier.num_votes) as pool:
        futures = [pool.submit(run_vote) for _ in range(verifier.num_votes)]
        vote_records = [future.result() for future in futures]

    valid_results = [record["result"] for record in vote_records if record["result"] is not None]
    true_results = [result for result in valid_results if result["is_correct"]]
    false_results = [result for result in valid_results if not result["is_correct"]]
    metadata = {
        "verification_valid": len(valid_results) == verifier.num_votes,
        "verification_valid_votes": len(valid_results),
        "verification_true_votes": len(true_results),
        "verification_false_votes": len(false_results),
        "verification_invalid_attempts": sum(record["invalid_attempts"] for record in vote_records),
    }

    if len(valid_results) != verifier.num_votes:
        return {
            **metadata,
            "is_correct": None,
            "feedback": "Verifier request or parse failure after retries",
        }

    majority = true_results if len(true_results) > len(false_results) else false_results
    return {**metadata, **majority[0]}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(dataset_root: Path, task_ids: Optional[List[int]], categories: Optional[List[str]], difficulties: Optional[List[int]]) -> List[Dict[str, Any]]:
    eval_path = dataset_root / "data" / "eval_benchmark_solvable.jsonl"
    scenario_path = dataset_root / "data" / "task_scenario_all_pool.jsonl"
    tasks = load_jsonl(eval_path)
    scenario_meta = {
        row["task_scenario_name"]: row
        for row in load_jsonl(scenario_path)
    }

    for task in tasks:
        meta = scenario_meta.get(task["task_scenario_name"], {})
        task["category"] = meta.get("category", "Unknown")
        task["domain"] = meta.get("domain", "Unknown")

    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if int(t["task_id"]) in wanted]
    if categories:
        wanted_categories = set(categories)
        tasks = [t for t in tasks if t.get("category") in wanted_categories]
    if difficulties:
        wanted_difficulties = set(difficulties)
        tasks = [t for t in tasks if int(t.get("difficulty_level", -1)) in wanted_difficulties]
    return tasks


def select_tasks(tasks: List[Dict[str, Any]], limit: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if not limit or limit >= len(tasks):
        return tasks
    rng = random.Random(seed)
    sampled = list(tasks)
    rng.shuffle(sampled)
    return sorted(sampled[:limit], key=lambda t: int(t["task_id"]))


def completed_keys(results_path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((int(row["task_id"]), row["arm"]))
            except Exception:
                continue
    return done


def observation_has_explicit_error(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None

    if isinstance(parsed, dict):
        if parsed.get("error") not in (None, "", False):
            return True
        status_code = parsed.get("status_code", parsed.get("http_status"))
        if isinstance(status_code, int) and status_code >= 500:
            return True
        status = str(parsed.get("status", "")).strip().lower().replace(" ", "_")
        return status in {"service_unavailable", "internal_error", "timeout"}

    lower = text.lower()
    return any(pattern in lower for pattern in EXPLICIT_ERROR_PATTERNS)


def normalize_observation(text: str) -> str:
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return re.sub(r"\s+", " ", text).strip()
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compact_observation(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def execute_agent(
    *,
    arm: str,
    client: Any,
    model: str,
    env: Any,
    task_scenario_name: str,
    agent_instruction: str,
    max_steps: int,
    max_tokens: int,
    reasoning_effort: Optional[str],
    agent_request_retries: int,
    agent_retry_delay: float,
    persist_max_interventions: int,
    persist_min_steps_before_stop: int,
    persist_policy: str,
    persist_repeat_threshold: int,
    persist_max_length_continuations: int,
) -> Dict[str, Any]:
    env.reset()
    current_time = datetime.now().strftime("%A, %B %d, %Y")
    system_template = (
        EXECUTOR_SYSTEM_PROMPT_SHORT
        if os.environ.get("OCCUBENCH_SHORT_INITIAL_CONTEXT", "0") == "1"
        else EXECUTOR_SYSTEM_PROMPT
    )
    system_msg = system_template.format(
        task_scenario_name=task_scenario_name,
        current_time=current_time,
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Task Instruction:\n{agent_instruction}\n\nPlease solve this task."},
    ]
    tools = env.get_tool_schemas()
    strict_controller = (
        StrictPersistController(
            tools,
            repeat_threshold=persist_repeat_threshold,
            max_length_continuations=min(
                persist_max_interventions,
                persist_max_length_continuations,
            ),
        )
        if arm == "persist_ace" and persist_policy == "strict_v1"
        else None
    )
    trajectory_parts: List[str] = []
    interventions: List[Intervention] = []
    step_count = 0
    finish_reasons: List[Optional[str]] = []
    reasoning_chars = 0
    prompt_tokens = 0
    completion_tokens = 0
    seen_action_observations: Dict[str, str] = {}
    length_continuations = 0
    max_length_continuations = int(os.environ.get("OCCUBENCH_LENGTH_CONTINUATIONS", "1"))

    for step in range(max_steps):
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools if tools else None,
            "max_tokens": max_tokens,
        }
        if tools:
            request_kwargs["tool_choice"] = "auto"
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
        response = create_chat_completion_with_retry(
            client,
            max_retries=agent_request_retries,
            retry_delay=agent_retry_delay,
            **request_kwargs,
        )
        log_response(response)
        choice = response.choices[0]
        message = choice.message
        finish_reasons.append(choice.finish_reason)
        legacy_reasoning_content = getattr(message, "reasoning_content", None)
        reasoning_content = legacy_reasoning_content
        if reasoning_content is None:
            reasoning_content = getattr(message, "reasoning", None)
        if reasoning_content:
            reasoning_chars += len(reasoning_content)
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if legacy_reasoning_content is not None:
            assistant_msg["reasoning_content"] = legacy_reasoning_content
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if message.content:
            trajectory_parts.append(f"\n[Agent Response]: {message.content}\n")

        if not message.tool_calls:
            should_continue_length = (
                choice.finish_reason == "length"
                and length_continuations < max_length_continuations
            )
            if strict_controller and strict_controller.allow_length_continuation(
                choice.finish_reason,
                has_tool_calls=False,
            ):
                should_continue_length = True
            if should_continue_length:
                length_continuations += 1
                interventions.append(
                    Intervention(
                        step=step_count,
                        kind="generation_length_continue",
                        detail=f"finish_reason=length; continuation={length_continuations}",
                    )
                )
                messages.append({"role": "user", "content": PERSIST_LENGTH_CONTINUATION_PROMPT})
                trajectory_parts.append("\n[PERSIST-ACE Intervention]: generation_length_continue\n")
                continue
            if (
                arm == "persist_ace"
                and persist_policy == "legacy_prompt_v0"
                and step_count < persist_min_steps_before_stop
                and len(interventions) < persist_max_interventions
            ):
                interventions.append(Intervention(step=step_count, kind="early_stop_continue", detail=f"step_count={step_count}"))
                messages.append({"role": "user", "content": PERSIST_EARLY_STOP_PROMPT})
                trajectory_parts.append("\n[PERSIST-ACE Intervention]: early_stop_continue\n")
                continue
            break

        pending_persist_prompt: Optional[str] = None
        pending_persist_label: Optional[str] = None

        for tc in message.tool_calls:
            tool_name = tc.function.name
            tool_args = tc.function.arguments
            action_key = json.dumps({"name": tool_name, "arguments": tool_args}, sort_keys=True, ensure_ascii=False)
            previous_observation = seen_action_observations.get(action_key)
            if strict_controller:
                strict_controller.inspect_proposal(
                    step=step_count + 1,
                    tool_name=tool_name,
                    tool_arguments=tool_args,
                )
            step_count += 1

            trajectory_parts.append(
                f"\n[Agent Action]: Call tool `{tool_name}` with arguments: {tool_args}\n"
            )
            observation = env.simulate(tool_name, tool_args)
            normalized_observation = normalize_observation(observation)
            repeated_without_evidence = (
                previous_observation is not None
                and previous_observation == normalized_observation
            )
            seen_action_observations[action_key] = normalized_observation
            if strict_controller:
                strict_controller.record_effect(tool_name, tool_args, observation)
            trajectory_parts.append(
                f"\n[Environment Observation] (from {tool_name}): {observation}\n"
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": observation,
            })

            if (
                arm == "persist_ace"
                and persist_policy == "legacy_prompt_v0"
                and len(interventions) < persist_max_interventions
            ):
                if observation_has_explicit_error(observation):
                    interventions.append(
                        Intervention(
                            step=step_count,
                            kind="explicit_error_recovery",
                            detail=compact_observation(observation),
                        )
                    )
                    pending_persist_prompt = PERSIST_ERROR_RECOVERY_PROMPT
                    pending_persist_label = "explicit_error_recovery"
                elif repeated_without_evidence:
                    interventions.append(
                        Intervention(step=step_count, kind="repeat_avoidance", detail=f"{tool_name}({tool_args[:120]})")
                    )
                    pending_persist_prompt = PERSIST_REPEAT_PROMPT
                    pending_persist_label = "repeat_avoidance"

        if pending_persist_prompt:
            messages.append({"role": "user", "content": pending_persist_prompt})
            trajectory_parts.append(f"\n[PERSIST-ACE Intervention]: {pending_persist_label}\n")

    return {
        "trajectory": "".join(trajectory_parts),
        "step_count": step_count,
        "interventions": [i.__dict__ for i in interventions],
        "agent_finish_reasons": finish_reasons,
        "agent_reasoning_chars": reasoning_chars,
        "agent_prompt_tokens": prompt_tokens,
        "agent_completion_tokens": completion_tokens,
        "persist_policy": persist_policy if arm == "persist_ace" else None,
        "controller_audit": (
            [decision.to_dict() for decision in strict_controller.audit]
            if strict_controller
            else []
        ),
    }


def evaluate_one(args: argparse.Namespace, task: Dict[str, Any], arm: str, registry: Any, modules: Dict[str, Any], verify: bool = True) -> Dict[str, Any]:
    config = registry.get(task["env_name"])
    if args.env_mode != "E0":
        config = copy.deepcopy(config)
        config["world_model_system_prompt"] += "\n\n" + modules["build_fault_prompt"](
            args.env_mode,
            args.fault_count,
            args.fault_duration,
        )

    legacy_extra_body = parse_extra_body(args.extra_body_json)
    agent_extra_body = parse_extra_body(args.agent_extra_body_json) or legacy_extra_body
    world_extra_body = parse_extra_body(args.world_extra_body_json) or legacy_extra_body
    verifier_extra_body = parse_extra_body(args.verifier_extra_body_json) or legacy_extra_body
    agent_client = make_client(args.agent_api_key, args.agent_base_url, agent_extra_body)
    world_client = make_client(args.world_api_key, args.world_base_url, world_extra_body)
    verifier_client = make_client(
        args.verifier_api_key,
        args.verifier_base_url,
        verifier_extra_body,
    )

    env = modules["LWMEnvironment"](config, world_client, args.world_model)
    result = execute_agent(
        arm=arm,
        client=agent_client,
        model=args.agent_model,
        env=env,
        task_scenario_name=task["task_scenario_name"],
        agent_instruction=task["agent_instruction"],
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        reasoning_effort=args.agent_reasoning_effort,
        agent_request_retries=args.agent_request_retries,
        agent_retry_delay=args.agent_retry_delay,
        persist_max_interventions=args.persist_max_interventions,
        persist_min_steps_before_stop=args.persist_min_steps_before_stop,
        persist_policy=args.persist_policy,
        persist_repeat_threshold=args.persist_repeat_threshold,
        persist_max_length_continuations=args.persist_max_length_continuations,
    )

    if verify:
        verifier = modules["Verifier"](args.verifier_model, verifier_client, num_votes=args.verifier_votes)
        initial_state = config.get("task_initial_state", "{}")
        if isinstance(initial_state, dict):
            initial_state = json.dumps(initial_state, ensure_ascii=False)
        verification = strict_majority_verification(
            verifier,
            task_scenario_name=task["task_scenario_name"],
            task_initial_state=initial_state,
            state_description=config.get("state_description", ""),
            agent_instruction=task["agent_instruction"],
            verification_plan=task["verification_plan"],
            trajectory=result["trajectory"],
            max_retries=args.verifier_retries,
        )
    else:
        verification = {
            "is_correct": None,
            "feedback": "intermediate stage not verified",
            "verification_valid": False,
            "verification_valid_votes": 0,
            "verification_true_votes": 0,
            "verification_false_votes": 0,
            "verification_invalid_attempts": 0,
        }

    return {
        "task_id": int(task["task_id"]),
        "arm": arm,
        "task_scenario_name": task["task_scenario_name"],
        "category": task.get("category", "Unknown"),
        "domain": task.get("domain", "Unknown"),
        "difficulty_level": task.get("difficulty_level"),
        "env_name": task["env_name"],
        "agent_model": args.agent_model,
        "world_model": args.world_model,
        "verifier_model": args.verifier_model,
        "env_mode": args.env_mode,
        "fault_count": args.fault_count if args.env_mode != "E0" else 0,
        "fault_duration": args.fault_duration if args.env_mode != "E0" else 0,
        "is_correct": (
            None if verification["is_correct"] is None else bool(verification["is_correct"])
        ),
        "feedback": verification["feedback"],
        "verification_valid": verification["verification_valid"],
        "verification_valid_votes": verification["verification_valid_votes"],
        "verification_true_votes": verification["verification_true_votes"],
        "verification_false_votes": verification["verification_false_votes"],
        "verification_invalid_attempts": verification["verification_invalid_attempts"],
        "step_count": result["step_count"],
        "agent_finish_reasons": result["agent_finish_reasons"],
        "agent_reasoning_chars": result["agent_reasoning_chars"],
        "agent_prompt_tokens": result["agent_prompt_tokens"],
        "agent_completion_tokens": result["agent_completion_tokens"],
        "intervention_count": len(result["interventions"]),
        "interventions": result["interventions"],
        "persist_policy": result["persist_policy"],
        "controller_audit_count": len(result["controller_audit"]),
        "controller_audit": result["controller_audit"],
        "trajectory": result["trajectory"],
    }


def summarize(results_path: Path) -> Dict[str, Any]:
    rows = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    summary: Dict[str, Any] = {"rows": len(rows), "valid_rows": 0, "invalid_rows": 0, "by_arm": {}}
    for row in rows:
        arm = row["arm"]
        bucket = summary["by_arm"].setdefault(
            arm,
            {"rows": 0, "valid": 0, "invalid": 0, "correct": 0, "interventions": 0},
        )
        bucket["rows"] += 1
        if row.get("is_correct") is None:
            bucket["invalid"] += 1
            summary["invalid_rows"] += 1
            continue
        bucket["valid"] += 1
        summary["valid_rows"] += 1
        bucket["correct"] += int(bool(row.get("is_correct")))
        bucket["interventions"] += int(row.get("intervention_count", 0) or 0)
    for bucket in summary["by_arm"].values():
        bucket["accuracy"] = bucket["correct"] / bucket["valid"] if bucket["valid"] else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--agent-model", default="deepseek-chat")
    parser.add_argument("--world-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--agent-api-key-env", default=None)
    parser.add_argument("--world-api-key-env", default=None)
    parser.add_argument("--verifier-api-key-env", default=None)
    parser.add_argument("--agent-base-url", default=None)
    parser.add_argument("--world-base-url", default=None)
    parser.add_argument("--verifier-base-url", default=None)
    parser.add_argument("--extra-body-json", default="")
    parser.add_argument("--agent-extra-body-json", default="")
    parser.add_argument("--world-extra-body-json", default="")
    parser.add_argument("--verifier-extra-body-json", default="")
    parser.add_argument("--agent-reasoning-effort", choices=["high", "max"], default=None)
    parser.add_argument("--env-mode", default="E0", choices=["E0", "E1", "E2", "E3"])
    parser.add_argument("--fault-count", type=int, default=2)
    parser.add_argument("--fault-duration", type=int, default=2)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--difficulties", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=["raw", "persist_ace"], choices=["raw", "persist_ace"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--agent-request-retries", type=int, default=5)
    parser.add_argument("--agent-retry-delay", type=float, default=1.0)
    parser.add_argument("--verifier-votes", type=int, default=3)
    parser.add_argument("--verifier-retries", type=int, default=2)
    parser.add_argument("--persist-max-interventions", type=int, default=2)
    parser.add_argument("--persist-min-steps-before-stop", type=int, default=3)
    parser.add_argument(
        "--persist-policy",
        choices=["strict_v1", "legacy_prompt_v0"],
        default="strict_v1",
    )
    parser.add_argument("--persist-repeat-threshold", type=int, default=3)
    parser.add_argument("--persist-max-length-continuations", type=int, default=0)
    args = parser.parse_args()
    # Local Random instances control sampling; seed the global RNG as well so
    # retry jitter and any environment-side randomness are reproducible.
    random.seed(args.seed)

    args.dataset_root = str(Path(args.dataset_root).resolve())
    dataset_root = Path(args.dataset_root)
    sys.path.insert(0, str(dataset_root))

    from occubench.lwm import WorldModelRegistry, LWMEnvironment
    from occubench.verifier import Verifier
    from occubench.fault_injection import build_fault_prompt

    modules = {
        "LWMEnvironment": LWMEnvironment,
        "Verifier": Verifier,
        "build_fault_prompt": build_fault_prompt,
    }

    args.api_key = args.api_key or os.getenv(args.api_key_env)
    resolve_role_runtime(args)
    args.world_model = args.world_model or args.agent_model
    args.verifier_model = args.verifier_model or args.world_model

    run_dir = Path(args.output_dir).resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"

    safe_args = vars(args).copy()
    for key in ("api_key", "agent_api_key", "world_api_key", "verifier_api_key"):
        safe_args[key] = "<redacted>"
    (run_dir / "args.json").write_text(json.dumps(safe_args, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = load_tasks(dataset_root, args.task_ids, args.categories, args.difficulties)
    tasks = select_tasks(tasks, args.limit, args.seed)
    pairs = [(task, arm) for task in tasks for arm in args.arms]
    done = completed_keys(results_path)
    pending = [(task, arm) for task, arm in pairs if (int(task["task_id"]), arm) not in done]

    registry = WorldModelRegistry(str(dataset_root / "data" / "world_model_configs"))
    write_lock = threading.Lock()

    print(f"Run dir: {run_dir}")
    print(f"Loaded tasks: {len(tasks)}; pending rows: {len(pending)}; env_mode={args.env_mode}; arms={args.arms}")

    def run_pair(item: tuple[Dict[str, Any], str]) -> Dict[str, Any]:
        task, arm = item
        return evaluate_one(args, task, arm, registry, modules)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_pair, item): item for item in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="OccuBench"):
            task, arm = futures[future]
            try:
                row = future.result()
                with write_lock:
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                err = {
                    "task_id": int(task["task_id"]),
                    "arm": arm,
                    "task_scenario_name": task.get("task_scenario_name"),
                    "execution_error": str(exc),
                    "is_correct": None,
                }
                with write_lock:
                    with results_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(err, ensure_ascii=False) + "\n")

    summary = summarize(results_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
