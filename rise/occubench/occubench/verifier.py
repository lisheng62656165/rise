"""
Verifier - Evaluates agent trajectories against verification rubrics.
Uses 3-vote majority for noise reduction.
"""

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from openai import OpenAI

from .lwm import create_client
from .debug import debug_print

logger = logging.getLogger(__name__)
_RESPONSE_LOG_LOCK = threading.Lock()


def _log_raw_response(response) -> None:
    """Persist a compact verifier response for timeout/parse diagnosis."""
    path = os.environ.get("OCCUBENCH_RESPONSE_LOG")
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    record = {
        "role": "verifier",
        "finish_reason": getattr(choice, "finish_reason", None),
        "content": getattr(message, "content", None),
        "usage": str(getattr(response, "usage", None)) if getattr(response, "usage", None) else None,
    }
    with _RESPONSE_LOG_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_json_object(content: str):
    """Parse the first valid JSON object instead of using a greedy regex."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(content or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(content[index:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None

VERIFIER_PROMPT = """You are a strict verifier for an AI agent evaluation benchmark.

## Task Scenario
{task_scenario_name}

## Initial State
{task_initial_state}

## State Description
{state_description}

## Agent Instruction
{agent_instruction}

## Verification Plan
{verification_plan}

## Agent Trajectory
{trajectory}

---

Based on the verification plan above, evaluate whether the agent successfully completed the task.

You must:
1. Check each criterion in the verification plan against the agent's trajectory.
2. Verify that the agent's actions and observations satisfy all required conditions.
3. Be strict: if any required criterion is not met, the task is a failure.

Respond in the following JSON format:
```json
{{
    "is_correct": true/false,
    "feedback": "Detailed explanation of why the task passed or failed."
}}
```
"""


class Verifier:
    """Evaluates agent trajectories using 3-vote majority verification."""

    def __init__(
        self,
        verifier_model: str,
        client: OpenAI = None,
        api_key: str = None,
        base_url: str = None,
        num_votes: int = 3,
    ):
        self.model = verifier_model
        self.client = client or create_client(api_key, base_url)
        self.num_votes = num_votes

    def _single_check(
        self,
        task_scenario_name: str,
        task_initial_state: str,
        state_description: str,
        agent_instruction: str,
        verification_plan: str,
        trajectory: str,
    ) -> Dict:
        """Run a single verification check."""
        prompt = VERIFIER_PROMPT.format(
            task_scenario_name=task_scenario_name,
            task_initial_state=task_initial_state,
            state_description=state_description,
            agent_instruction=agent_instruction,
            verification_plan=verification_plan,
            trajectory=trajectory,
        )

        try:
            request = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(os.environ.get("OCCUBENCH_VERIFIER_MAX_TOKENS", "4096")),
            }
            if os.environ.get("OCCUBENCH_STREAM", "0") == "1":
                request["stream"] = True
                parts = []
                for chunk in self.client.chat.completions.create(**request):
                    for choice in getattr(chunk, "choices", None) or []:
                        delta = getattr(choice, "delta", None)
                        if delta is not None and getattr(delta, "content", None):
                            parts.append(delta.content)
                content = "".join(parts)
            else:
                response = self.client.chat.completions.create(**request)
                _log_raw_response(response)
                content = response.choices[0].message.content

            # Parse JSON from response
            result = _extract_json_object(content)
            if result is not None and type(result.get("is_correct")) is bool:
                return {
                    "is_correct": result["is_correct"],
                    "feedback": result.get("feedback", ""),
                }
        except Exception as e:
            logger.error(f"Verification failed: {e}")

        return {"is_correct": False, "feedback": "Verification error"}

    def check(
        self,
        task_scenario_name: str,
        task_initial_state: str,
        state_description: str,
        agent_instruction: str,
        verification_plan: str,
        trajectory: str,
    ) -> Dict:
        """
        Run majority-vote verification.

        Returns:
            Dict with 'is_correct' (bool) and 'feedback' (str)
        """
        with ThreadPoolExecutor(max_workers=self.num_votes) as pool:
            futures = [
                pool.submit(
                    self._single_check,
                    task_scenario_name,
                    task_initial_state,
                    state_description,
                    agent_instruction,
                    verification_plan,
                    trajectory,
                )
                for _ in range(self.num_votes)
            ]
            results = [f.result() for f in futures]

        true_votes = [r for r in results if r["is_correct"]]
        false_votes = [r for r in results if not r["is_correct"]]

        debug_print(f"[VERIFIER] Votes: {len(true_votes)} TRUE, {len(false_votes)} FALSE")
        for i, r in enumerate(results):
            debug_print(f"[VERIFIER] Vote {i+1}: correct={r['is_correct']}, feedback={r['feedback'][:200]}")

        if len(true_votes) > len(false_votes):
            return true_votes[0]
        else:
            return false_votes[0]
