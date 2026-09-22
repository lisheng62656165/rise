"""DeepSeek Flash OpenAI-compatible client for STATE-Bench."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

import sys
from pathlib import Path


_benchmark = Path(__file__).resolve().parents[1] / "benchmark"
if str(_benchmark) not in sys.path:
    sys.path.insert(0, str(_benchmark))

from state_bench.client import BaseLLMClient


@dataclass(slots=True)
class DeepSeekFlashResponse:
    text: str
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None = None
    usage: Any | None = None


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


class DeepSeekFlashClient(BaseLLMClient):
    """Small Chat Completions wrapper for DeepSeek Flash-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        thinking_enabled: bool = False,
        seed: int | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.thinking_enabled = thinking_enabled
        self.seed = seed
        self._client = OpenAI(
            api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout_seconds
        )

    @classmethod
    def from_env(cls) -> "DeepSeekFlashClient":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("STATE_BENCH_AGENT_API_KEY")
        if not api_key:
            raise ValueError("Set DEEPSEEK_API_KEY or STATE_BENCH_AGENT_API_KEY.")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("DEEPSEEK_MODEL", "gpt-4.1"),
            max_tokens=int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("DEEPSEEK_TEMPERATURE", "0")),
            reasoning_effort=os.environ.get("DEEPSEEK_REASONING_EFFORT") or None,
            thinking_enabled=os.environ.get("DEEPSEEK_THINKING", "").lower() in {"1", "true", "yes", "on"},
        )

    @property
    def provider_name(self) -> str:
        return "deepseek-openai-compatible"

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> DeepSeekFlashResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        if self.seed is not None:
            kwargs["seed"] = self.seed

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            calls.append(
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                }
            )
        return DeepSeekFlashResponse(
            text=message.content or "",
            tool_calls=calls,
            reasoning_content=getattr(message, "reasoning_content", None),
            usage=getattr(response, "usage", None),
        )

    def complete_chat(
        self, messages: list[dict[str, str]], max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete_json(
        self, prompt: str, system_prompt: str | None = None,
        max_tokens: int | None = None, reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt.rstrip() + "\n\nReturn only one valid JSON object.",
            })
        messages.append({"role": "user", "content": prompt})
        last_error: Exception | None = None
        for attempt in range(3):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            if reasoning_effort or self.reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort or self.reasoning_effort
            if self.seed is not None:
                kwargs["seed"] = self.seed
            try:
                response = self._client.chat.completions.create(**kwargs)
                return _parse_json_object(response.choices[0].message.content or "")
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise ValueError(f"JSON response failed after 3 attempts: {last_error}")
