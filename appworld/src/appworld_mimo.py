from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any

from .llm_client import LLMClient


CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Extract the first Python block, matching AppWorld's official ReAct behavior."""

    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else ""


def build_appworld_react_messages(task: Any, memory_note: str = "") -> list[dict[str, str]]:
    app_descriptions = json.dumps(
        [{"name": name, "description": description} for name, description in task.app_descriptions.items()],
        ensure_ascii=False,
        indent=2,
    )
    supervisor = task.supervisor
    prompt = f"""You are an autonomous ReAct coding agent operating AppWorld.

Complete the supervisor's task by writing one small Python code block per turn. The code is
executed in a persistent Python REPL, so variables remain available in later turns.

Use only the provided `apis` object to interact with apps. Do not use OS, filesystem, process,
network, or third-party package operations. Always inspect an API specification with
`apis.api_docs.show_api_doc` before calling it. Process every page of paginated APIs. Never
invent IDs, credentials, or values. Avoid unrelated changes.

Useful discovery calls:
```python
print(apis.api_docs.show_app_descriptions())
```
```python
print(apis.api_docs.show_api_descriptions(app_name="spotify"))
```
```python
print(apis.api_docs.show_api_doc(app_name="spotify", api_name="login"))
```

When the task is complete, you must call `apis.supervisor.complete_task()`. For action tasks
(send, create, update, move, like, etc.), call it without an `answer`; do not pass a completion
summary. Only pass the minimal direct `answer` when the supervisor explicitly asks for
information to be returned. If the task cannot be completed, call it with `status="fail"`.

Available apps:
{app_descriptions}

Supervisor:
- Name: {supervisor.first_name} {supervisor.last_name}
- Email: {supervisor.email}
- Phone: {supervisor.phone_number}

Task: {task.instruction}

{memory_note}

Respond with reasoning followed by exactly one Python code block. Do not emit multiple code
blocks in one turn."""
    return [{"role": "user", "content": prompt}]


def run_appworld_react(
    world: Any,
    llm: LLMClient,
    *,
    max_steps: int = 50,
    memory_note: str = "",
) -> tuple[list[dict[str, Any]], str]:
    messages: list[dict[str, Any]] = build_appworld_react_messages(world.task, memory_note)
    trace: list[dict[str, Any]] = []
    termination_reason = "max_steps"

    for step in range(1, max_steps + 1):
        result = llm.complete_messages_with_trace(messages)
        assistant_text = result.text
        code = extract_python_code(assistant_text)
        execution_output = world.execute(code) if code else "No executable Python code block found."
        trace.append(
            {
                "step": step,
                "assistant_text": assistant_text,
                "code": code,
                "execution_output": execution_output,
                "llm_request": result.request,
                "llm_response": result.response,
            }
        )
        messages.append(copy.deepcopy(result.message))
        messages.append(
            {
                "role": "user",
                "content": f"Output:\n```\n{execution_output}\n```\n\nContinue with one Python code block.",
            }
        )
        if world.task_completed():
            termination_reason = "task_completed"
            break

    return trace, termination_reason


def calculate_appworld_loop_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    codes = [entry.get("code", "").strip() for entry in trace if entry.get("code", "").strip()]
    outputs = [str(entry.get("execution_output", "")).strip() for entry in trace]
    code_counts = Counter(codes)
    output_counts = Counter(output for output in outputs if output)
    repeated_exact_code_calls = sum(count - 1 for count in code_counts.values() if count > 1)
    repeated_outputs = sum(count - 1 for count in output_counts.values() if count > 1)
    max_same_output_streak = 0
    current_streak = 0
    previous_output: str | None = None
    for output in outputs:
        if output and output == previous_output:
            current_streak += 1
        else:
            current_streak = 1 if output else 0
        max_same_output_streak = max(max_same_output_streak, current_streak)
        previous_output = output
    return {
        "steps": len(trace),
        "code_executions": len(codes),
        "empty_code_responses": len(trace) - len(codes),
        "repeated_exact_code_calls": repeated_exact_code_calls,
        "repeated_outputs": repeated_outputs,
        "max_same_output_streak": max_same_output_streak,
        "loop_candidate": repeated_exact_code_calls >= 2 or max_same_output_streak >= 3,
    }
