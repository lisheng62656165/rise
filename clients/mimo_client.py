"""MiMo OpenAI-compatible client for StateBench simulation and judging."""

from __future__ import annotations

import os

from clients.deepseek_flash_client import DeepSeekFlashClient


class MiMoClient(DeepSeekFlashClient):
    @classmethod
    def from_env(cls) -> "MiMoClient":
        api_key = (os.environ.get("MIMO_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("Set MIMO_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("MIMO_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("MIMO_MODEL", "gpt-4.1"),
            max_tokens=int(os.environ.get("MIMO_MAX_TOKENS", "4096")),
            temperature=float(os.environ.get("MIMO_TEMPERATURE", "0")),
            seed=int(os.environ["MIMO_SEED"]) if os.environ.get("MIMO_SEED") else None,
            reasoning_effort=os.environ.get("MIMO_REASONING_EFFORT") or None,
            thinking_enabled=os.environ.get("MIMO_THINKING", "").lower()
            in {"1", "true", "yes", "on"},
            timeout_seconds=float(os.environ.get("MIMO_TIMEOUT_SECONDS", "180")),
        )

    @property
    def provider_name(self) -> str:
        return "xiaomi-mimo-openai-compatible"
