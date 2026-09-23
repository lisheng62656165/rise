"""Sidecar auditing for clean DeepPlanning model calls."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _message_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def install_call_audit(module: Any, output_path: Path) -> Callable[..., Any]:
    """Wrap a domain module's imported call_llm without changing its behavior."""
    original = module.call_llm
    lock = threading.Lock()

    def audited_call(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages")
        if messages is None and len(args) > 1:
            messages = args[1]
        response = original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        row = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "thread_id": threading.get_ident(),
            "prefix_sha256": _message_hash(messages or []),
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with lock, output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return response

    module.call_llm = audited_call
    return original


def wrap_openai_client(client: Any, output_path: Path, extra_body: dict[str, Any]) -> Any:
    """Proxy Chat Completions to apply frozen reasoning settings and audit usage."""
    lock = threading.Lock()
    original_create = client.chat.completions.create

    def audited_create(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("extra_body", extra_body)
        response = original_create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        row = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "travel_parser",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with lock, output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return response

    client.chat.completions.create = audited_create
    return client
