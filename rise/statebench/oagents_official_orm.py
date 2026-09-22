"""Official OAgents ORM prompt with StateBench public-conversation serialization.

Matches the project's official ORM reselection protocol, not the OAgents
CodeAgent runtime. No StateTrace event features or evaluator scores are added.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from phase0_runner import write_json
from trace_ir import public_trajectory

PROTOCOL = "oagents_official_orm_listwise_statebench_v1"
PROMPT_PATH = Path(__file__).resolve().parent / "third_party/OAgents/ORM_list_wise.yaml"


def load_prompt():
    prompt = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing official ORM prompt")
    return prompt


def build_messages(prompt, candidates):
    text = ""
    for index, candidate in enumerate(candidates):
        trajectory = public_trajectory(candidate)
        text += f"---Trajectory - {index}---\n"
        text += json.dumps(trajectory, ensure_ascii=False) + "\n"
    text += "you can start!"
    return [{"role": "system", "content": prompt}, {"role": "user", "content": text}]


def parse_choice(text):
    decoder = json.JSONDecoder()
    choices = []
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            position = start + 1
            continue
        position = start + length
        if isinstance(value, dict) and "index" in value:
            index = value["index"]
            if type(index) is not int or index not in range(4):
                raise ValueError("Out-of-range official ORM index")
            if not isinstance(value.get("analysis"), str) or not value["analysis"].strip():
                raise ValueError("Missing official ORM analysis")
            choices.append(value)
    if len(choices) != 1:
        raise ValueError("Expected one unambiguous official ORM choice")
    return choices[0]


def select_with_retries(client, messages, audit_path, attempts=3):
    history = []
    for attempt in range(1, attempts + 1):
        text = client.complete_chat(messages=messages, max_tokens=2048, temperature=0.0)
        history.append({"attempt": attempt, "response": text})
        try:
            choice = parse_choice(text)
        except ValueError:
            write_json(audit_path, {"attempts": history, "valid": False})
            continue
        write_json(audit_path, {"attempts": history, "valid": True, "choice": choice})
        return choice, len(history)
    raise ValueError("Official ORM format retries exhausted; no fallback selection")
