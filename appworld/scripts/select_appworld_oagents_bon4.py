from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings
from src.llm_client import LLMClient


# Verbatim prompt content from OPPO-PersonalAI/OAgents (Apache-2.0),
# OAgents/src/oagents/prompts/ORM_list_wise.yaml.
ORM_LIST_WISE_PROMPT = """Evaluation_Guidelines:
  Objective:
    description: >
      You will evaluate N candidate trajectories, each representing a series of nodes in a search tree. Each trajectory contains the following:
      - step_number: Depth of the node in the trajectory.
      - observations: Observations recorded at each step of the trajectory.
      - action_output: Direct action output at each step.
      - model_output: Raw model output (LLM).
      - error: Any errors encountered (can be None).
      - previous_steps: The history of earlier steps, including TaskStep and PlanningStep, with the trajectory of ActionSteps leading to the current state.
    goal: >
      Your goal is to evaluate each trajectory holistically, considering how well it progresses toward solving the user's task. Select the trajectory that most effectively achieves this goal.

  Evaluation_Criteria:
    Progress_Toward_Goal:
      description: >
        Assess how well each trajectory advances the task at hand, considering both the individual node's progress and the overall progression of the entire trajectory.
      key_points:
        - Reward trajectories that demonstrate tangible and meaningful progress toward the goal.
        - Penalize trajectories with weak actions or minimal/no advancement.

    Trajectory_Efficiency:
      description: >
        Evaluate how efficiently each trajectory progresses toward the goal, considering the depth and complexity of the steps.
      key_points:
        - Favor trajectories that achieve significant progress with fewer steps.
        - Consider the overall value-to-depth ratio when comparing trajectories of different lengths.
        - Reward efficient exploration of the search space.

    Loop_Detection:
      description: >
        Detect loops or repetitions within each trajectory, especially those related to previous steps.
      loop_types:
        - Real Loops: Identical nodes (observations, action output, and model output) that do not add value to the trajectory.
        - Benign Repetitions: Similar strategies with variations yielding additional progress.
      key_points:
        - Heavily penalize trajectories with real loops.
        - Slight penalties for benign repetitions if they lead to meaningful improvements.

    Error_and_Stability:
      description: >
        Evaluate the severity of errors encountered in each trajectory and penalize based on their impact on progression.
      error_severity:
        - Fatal/Blocking Errors: Major penalty.
        - Significant Errors: Moderate penalty.
        - Minor/Recoverable Issues: Minor penalty.
      key_points:
        - Penalize unstable or unclear model outputs.
        - Consider how errors affect the overall trajectory's ability to move toward the goal.

    Overall_Trajectory_Quality:
      description: >
        Evaluate the coherence and overall quality of the trajectory.
      key_points:
        - Consider the logical sequence of steps and the exploration-exploitation balance.
        - Evaluate the final node's closeness to achieving the goal.
        - Reward trajectories that make consistent progress and demonstrate coherent planning.

  Evaluation_Variables:
    description: >
      Instead of a single score, evaluate each trajectory based on the following dimensions. Compare all dimensions of the trajectories and select the most optimal one.
    dimensions:
      - dimension_name: "Progress Toward Goal"
        description: "The degree to which the trajectory progresses toward solving the task."
        score_range: "0-10"
      - dimension_name: "Efficiency"
        description: "The efficiency of the trajectory in terms of the number of steps and exploration depth."
        score_range: "0-10"
      - dimension_name: "Loop Detection"
        description: "The number and severity of loops detected in the trajectory."
        score_range: "0-10"
      - dimension_name: "Error Severity"
        description: "The severity of errors in the trajectory and their impact on progression."
        score_range: "0-10"
      - dimension_name: "Coherence and Quality"
        description: "The overall quality of the trajectory, including its logical flow and consistency."
        score_range: "0-10"

  Final_Output_Format:
    description: >
      Provide your evaluation in the following JSON format. Select the best trajectory and provide a detailed analysis explaining why it is the most promising trajectory.
    output_format:
      json:
        - index: [integer]  # Index of the best trajectory
        - analysis: "Detailed analysis addressing progress, efficiency, reflection usage, loop detection, error severity, and overall trajectory quality."
        - selected_trajectory_dimensions: 
            - dimension_name: "Progress Toward Goal"
              score: [x.xx]
            - dimension_name: "Efficiency"
              score: [x.xx]
            - dimension_name: "Loop Detection"
              score: [x.xx]
            - dimension_name: "Error Severity"
              score: [x.xx]
            - dimension_name: "Coherence and Quality"
              score: [x.xx]
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temp, path)


def task_instruction(candidate: Mapping[str, Any]) -> str:
    trace = candidate.get("trace") or []
    for step in trace:
        messages = ((step.get("llm_request") or {}).get("json") or {}).get("messages") or []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            match = re.search(r"\nTask:\s*(.+?)(?:\n\n|$)", content, re.S)
            if match:
                return match.group(1).strip()
    return "Complete the visible AppWorld supervisor task."


def trajectory_text(candidate: Mapping[str, Any]) -> str:
    parts = [f"TaskStep:\n{task_instruction(candidate)}"]
    previous = "TaskStep"
    for ordinal, step in enumerate(candidate.get("trace") or [], start=1):
        model_output = str(step.get("assistant_text") or "")
        action_output = str(step.get("code") or model_output)
        observation = str(step.get("execution_output") or "")
        error = observation if re.search(r"execution failed|traceback|\berror\b|exception", observation, re.I) else "None"
        parts.append(
            "\n".join(
                [
                    f"ActionStep {ordinal}:",
                    f"step_number: {ordinal}",
                    f"observations: {observation}",
                    f"action_output: {action_output}",
                    f"model_output: {model_output}",
                    f"error: {error}",
                    f"previous_steps: {previous}",
                ]
            )
        )
        previous = f"TaskStep, ActionSteps 1-{ordinal}"
    parts.append(f"Final_Answer: {str((candidate.get('trace') or [{}])[-1].get('assistant_text') or '')}")
    return "\n\n".join(parts)


def selector_input(candidates: list[Mapping[str, Any]]) -> str:
    text = ""
    for index, candidate in enumerate(candidates):
        text += f"---Trajectory - {index}---\n{trajectory_text(candidate)}\n"
    return text + "you can start!"


def parse_first_json(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode((text or "")[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return {}


def select_task(task_dir: Path, output_dir: Path, max_tokens: int) -> dict[str, Any]:
    output_path = output_dir / f"{task_dir.name}.json"
    if output_path.exists():
        return {"task_id": task_dir.name, "status": "reused"}
    candidates = [read_json(task_dir / f"candidate_{label}.json") for label in "abcd"]
    prompt = selector_input(candidates)
    current = get_settings()
    max_token_field = "max_completion_tokens" if current.model_name.startswith("mimo-") else "max_tokens"
    payload: dict[str, Any] = {max_token_field: max_tokens, "temperature": 0.0}
    if "integrate.api.nvidia.com" in current.base_url:
        payload["thinking"] = {"type": "disabled"}
    result = LLMClient(replace(current, temperature=0.0, max_completion_tokens=max_tokens)).complete_messages_with_trace(
        [
            {"role": "system", "content": ORM_LIST_WISE_PROMPT},
            {"role": "user", "content": prompt},
        ],
        extra_payload=payload,
    )
    parsed = parse_first_json(result.text)
    try:
        index = int(parsed.get("index", -1))
    except (TypeError, ValueError):
        index = -1
    fallback = not 0 <= index < 4
    if fallback:
        index = 0
    record = {
        "schema_version": "appworld_oagents_bon4_listwise_v1",
        "task_id": task_dir.name,
        "source": "OPPO-PersonalAI/OAgents",
        "search_type": "BON",
        "n_rollouts": 4,
        "result_merging_type": "list-wise",
        "full_trajectory_input": True,
        "selected_index": index,
        "selected_candidate": f"C{index + 1}",
        "selected_success": bool(candidates[index]["evaluation_success"]),
        "candidate_successes": [bool(candidate["evaluation_success"]) for candidate in candidates],
        "oracle_success": any(bool(candidate["evaluation_success"]) for candidate in candidates),
        "fallback": fallback,
        "analysis": str(parsed.get("analysis") or ""),
        "selected_trajectory_dimensions": parsed.get("selected_trajectory_dimensions"),
        "request": result.request,
        "response": result.response,
    }
    write_json(output_path, record)
    (output_dir / f"{task_dir.name}.error.json").unlink(missing_ok=True)
    return {"task_id": task_dir.name, "status": "completed"}


def select_task_safe(task_dir: Path, output_dir: Path, max_tokens: int) -> dict[str, Any]:
    try:
        return select_task(task_dir, output_dir, max_tokens)
    except Exception as exc:
        error = {"task_id": task_dir.name, "status": "error", "exception": type(exc).__name__, "message": str(exc)}
        write_json(output_dir / f"{task_dir.name}.error.json", error)
        return error


def scenario_metrics(values: Mapping[str, bool]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = {}
    for task_id, success in values.items():
        groups.setdefault(task_id.rsplit("_", 1)[0], []).append(success)
    complete_groups = [rows for rows in groups.values() if len(rows) == 3]
    scenario_successes = sum(all(rows) for rows in complete_groups)
    return {
        "task_successes": sum(values.values()),
        "num_tasks": len(values),
        "tgc": round(100 * sum(values.values()) / len(values), 1) if values else 0.0,
        "scenario_successes": scenario_successes,
        "num_scenarios": len(complete_groups),
        "sgc": round(100 * scenario_successes / len(complete_groups), 1) if complete_groups else None,
    }


def paired(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict[str, Any]:
    keys = sorted(set(a) & set(b))
    wins = sum(a[key] and not b[key] for key in keys)
    losses = sum(not a[key] and b[key] for key in keys)
    return {"coverage": len(keys), "wins": wins, "losses": losses, "ties": len(keys) - wins - losses}


def summarize(output_dir: Path, task_ids: list[str], eds_eca_dir: Path | None) -> dict[str, Any]:
    rows = [read_json(output_dir / f"{task_id}.json") for task_id in task_ids if (output_dir / f"{task_id}.json").exists()]
    c1_values = {row["task_id"]: bool(row["candidate_successes"][0]) for row in rows}
    selected_values = {row["task_id"]: bool(row["selected_success"]) for row in rows}
    oracle_values = {row["task_id"]: bool(row["oracle_success"]) for row in rows}
    c1 = sum(c1_values.values())
    selected = sum(selected_values.values())
    oracle = sum(oracle_values.values())
    wins = sum(bool(row["selected_success"]) and not bool(row["candidate_successes"][0]) for row in rows)
    losses = sum(not bool(row["selected_success"]) and bool(row["candidate_successes"][0]) for row in rows)
    summary = {
        "schema_version": "appworld_oagents_bon4_summary_v1",
        "completed": len(rows),
        "task_count": len(task_ids),
        "c1_successes": c1,
        "selected_successes": selected,
        "oracle_successes": oracle,
        "paired_vs_c1": {"wins": wins, "losses": losses, "ties": len(rows) - wins - losses},
        "fallbacks": sum(bool(row.get("fallback")) for row in rows),
        "selection_counts": {f"C{i + 1}": sum(row["selected_index"] == i for row in rows) for i in range(4)},
        "c1_metrics": scenario_metrics(c1_values),
        "selected_metrics": scenario_metrics(selected_values),
        "oracle_metrics": scenario_metrics(oracle_values),
    }
    if eds_eca_dir:
        eds_values = {}
        for task_id in task_ids:
            path = eds_eca_dir / task_id / "final.json"
            if path.exists():
                eds_values[task_id] = bool(read_json(path)["selected_success"])
        summary["eds_eca_metrics"] = scenario_metrics(eds_values)
        summary["selected_vs_eds_eca"] = paired(selected_values, eds_values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--eds-eca-dir", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.candidate_dir / "manifest.json")
    task_ids = list(manifest["task_ids"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs = [args.candidate_dir / task_id for task_id in task_ids]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(select_task_safe, task_dir, args.output_dir, args.max_tokens) for task_dir in task_dirs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=True), flush=True)
    summary = summarize(args.output_dir, task_ids, args.eds_eca_dir)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
